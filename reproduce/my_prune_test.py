import argparse
from datetime import timedelta
from importlib import reload
import json
from logging import Logger
import logging
import math
import os
import pickle
import pprint
import random
import time
from scipy.optimize import fsolve

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp

import engine.utils as utils
import registry

import optuna

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "prune", "test"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="cifar100", choices=['cifar10', 'cifar100', 'modelnet40'])
parser.add_argument('--dataroot', default='data', help='path to your datasets')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--total-epochs", type=int, default=100)
parser.add_argument("--lr-decay-milestones", default="60,80", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument('--output-dir', default='run', help='path where to save')
parser.add_argument("--finetune", action="store_true", default=False, help='whether finetune or not')

# For pruning
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--speed-up", type=float, default=2)
parser.add_argument("--max-pruning-ratio", type=float, default=1.0)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=5e-4)
parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
parser.add_argument("--weight-decay", type=float, default=5e-4)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=False)
# parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
# parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
# parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
# parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
# parser.add_argument("--sl-restore", type=str, default=None)
parser.add_argument("--iterative-steps", default=400, type=int)

#args = parser.parse_args()

def progressive_pruning(pruner, 
                        model, 
                        prune_rate, 
                        example_inputs, 
                        logger: Logger
                        # device=None, 
                        # train_loader=None, 
                        # test_loader=None
                        ):
    model.eval()
    base_ops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_prune_rate = 0.0
    while current_prune_rate < prune_rate:
        # if test_loader:
        #     pass
            # eval(model=model, test_loader=test_loader)
            # acc, val_loss = eval(model, test_loader, device=device)
            # args.logger.info(
            #     "Speedup {:.2f}/{:.2f}, Acc={:.4f}, Val Loss={:.4f}".format(
            #         current_speed_up, speed_up, acc, val_loss
            #     )
            # )
        # if args.method == "obdc":
        #     model.zero_grad()
        #     imp=pruner.importance
        #     imp._prepare_model(model, pruner)
        #     for k, (imgs, lbls) in enumerate(train_loader):
        #         if k>=10: break
        #         imgs = imgs.to(args.device)
        #         lbls = lbls.to(args.device)
        #         output = model(imgs)
        #         sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),
        #                                           1).squeeze().to(args.device)
        #         loss_sample = F.cross_entropy(output, sampled_y)
        #         loss_sample.backward()
        #         imp.step()
        #     pruner.step()
        #     imp._rm_hooks(model)
        #     imp._clear_buffer()
        # else:
        pruner.step()

        pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_prune_rate = (float(base_params) - pruned_params) / base_params
        # logger.debug(f"Current Prune Rate: {current_prune_rate:.4f}")
        if pruner.current_step == pruner.iterative_steps:
            logger.info("Reached max iterative steps in progressive pruner.")
            break
    return current_prune_rate
    
def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()

def get_pruner(model, 
               example_inputs,
               dataset_num_classes,
               iterative_steps : int = 400,
               max_pruning_ratio : float = 1.0,
               global_pruning : bool = True):
    imp = tp.importance.MagnitudeImportance(p=2) # L2 Norm Pruning
    #pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)

    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == dataset_num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == dataset_num_classes:
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=max_pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        global_pruning=global_pruning
    )

    return pruner

def setup_logger(dataset: str = "cifar10",
                 model: str = "resnet56",
                 method: str = "l2",
                 mode: str = "prune",
                 output_dir: str = "run",
                 global_pruning : bool = True,
                 ) -> Logger:
    prefix = 'global' if global_pruning else 'local'
    logger_name = "{}-{}-{}-{}".format(dataset, prefix, method, model)
    output_dir = os.path.join(output_dir, logger_name)
    log_file = "{}/{}.txt".format(output_dir, logger_name)
    
    logger = utils.get_logger(logger_name, output=log_file)
    return logger, output_dir

def setup_model_and_dataset(logger: Logger,
                            dataroot: str = "data",
                            dataset: str = "cifar10",
                            model_name: str = "resnet56",
                            batch_size: int = 128,
                            restore: str = "models/cifar10_resnet56.pth"):
    # Model & Dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, train_dst, val_dst, input_size = registry.get_dataset(
        dataset, data_root=dataroot
    )
    #args.num_classes = num_classes
    model = registry.get_model(model_name, num_classes=num_classes, pretrained=True, target_dataset=dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=batch_size, num_workers=4
    )
    
    # for k, v in utils.utils.flatten_dict(vars(args)).items():  # print args
    #     logger.info("%s: %s" % (k, v))

    if restore is not None:
        loaded = torch.load(restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        logger.info("Loading model from {restore}".format(restore=restore))
    model = model.to(device)

    example_inputs = train_dst[0][0].unsqueeze(0).to(device)

    return model, train_loader, test_loader, example_inputs, num_classes

def prune(model, 
          example_inputs, 
          pruner,
          test_loader,
          logger: Logger,
          prune_rate: float = 0.5,
          device: str = "cuda")-> tuple:
    model.eval()
    ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    ori_acc, ori_val_loss = eval(model, test_loader, device=device)
    logger.info("Pruning...")

    progressive_pruning(pruner, 
                        model, 
                        prune_rate=prune_rate, 
                        example_inputs=example_inputs,
                        logger=logger)
    
    del pruner # remove reference
    # logger.info(model) # Print whole model architecture
    pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    pruned_acc, pruned_val_loss = eval(model, test_loader, device=device)
    
    logger.info(
        "Params: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
            ori_size / 1e6, 
            pruned_size / 1e6, 
            pruned_size / ori_size * 100,
            ori_size / pruned_size
        )
    )
    logger.info(
        "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
            ori_ops / 1e6,
            pruned_ops / 1e6,
            pruned_ops / ori_ops * 100,
            ori_ops / pruned_ops,
        )
    )
    logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
    logger.info(
        "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
    )
    return pruned_size, pruned_ops
    
def train_model(
    model,
    train_loader,
    test_loader,
    logger: Logger,
    output_dir: str,
    dataset: str = "cifar10",
    model_name: str = "resnet56",
    method: str = "l2",
    epochs: int = 100,
    lr: float = 0.01,
    lr_decay_milestones: str = "60,80",
    lr_decay_gamma: float = 0.1,
    save_as = None,
    verbose: bool = False,
    
    # For pruning
    weight_decay = 5e-4,
    save_state_dict_only = True,
    #pruner = None,
    device = None,
) -> tuple:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay, #if pruner is None else 0,
    )
    milestones = None if lr_decay_milestones is None else [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    model.to(device)
    best_acc = -1

    # Starting point
    model.eval()
    acc, val_loss = eval(model, test_loader, device=device)
    logger.info(
        "Starting point: Acc={:.4f}, Val Loss={:.4f}".format(
            acc, val_loss
        )
    )
    
    for epoch in range(epochs):
        model.train()
    
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            # if pruner is not None:
            #     pruner.regularize(model) # for sparsity learning
            optimizer.step()
            if i % 10 == 0 and verbose:
                logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        # if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner):
        #     pruner.update_reg() # increase the strength of regularization
            #print(pruner.group_reg[pruner._groups[0]])
        
        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(output_dir, exist_ok=True)
            # if mode == "prune":
            if save_as is None:
                save_as = os.path.join( output_dir, "{}_{}_{}.pth".format(dataset, model_name, method) )

            if save_state_dict_only:
                torch.save(model.state_dict(), save_as)
            else:
                torch.save(model, save_as)
            # elif args.mode == "pretrain":
            #     if save_as is None:
            #         save_as = os.path.join( args.output_dir, "{}_{}.pth".format(args.dataset, args.model) )
            #     torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    logger.info("Best Acc=%.4f" % (best_acc))
    logger.info("Last Acc=%.4f" % (acc))
    return best_acc, acc

def retrain(model,
            train_loader,
            test_loader,
            logger: Logger,
            output_dir: str,
            device: str = "cuda",
            total_epochs: int = 100,
            lr: float = 0.01,
            lr_decay_milestones: str = "60,80"
            ) -> tuple:
    
    logger.info("Re-training...")
    best_acc, last_acc = train_model(
        model,
        train_loader=train_loader,
        test_loader=test_loader,
        logger=logger,
        output_dir=output_dir,
        epochs=total_epochs,
        lr=lr,
        lr_decay_milestones=lr_decay_milestones,
        save_state_dict_only=False,
        device=device,
    )
    return best_acc, last_acc

def save_results(output_dir: str,
                 results: dict,
                 file_name: str = "results"):
    save_as = os.path.join(output_dir, file_name)

    # Save results in pretty text format
    with open('{}.txt'.format(save_as), 'w') as fp:
        pprint.pprint(results, fp)

    # Save results in json format
    with open('{}.json'.format(save_as), "w") as fp:
        json.dump(results, fp)

# Define the function to solve
def static_equations(vars):
    E1, E2, E3, E4, E5 = vars
    eq1 = E1 + 0.9 * E2 + 0.81 * E3 + 0.729 * E4 + 0.6561 * E5 - 100
    return [eq1, 0, 0, 0, 0]  # Return an array with appropriate size

def static_allocation():
    # Initial guess
    initial_guess = [20, 20, 20, 20, 30]

    # Solve
    solution = fsolve(static_equations, initial_guess)
    E1, E2, E3, E4, E5 = solution

    print(f"Epochs for each iteration: E1={E1}, E2={E2}, E3={E3}, E4={E4}, E5={E5}")

def dynamic_epoch_allocation_pruning(
    seed: int = None,
    retraining_epochs = 60,
    finetuning_epochs = 40,
    finetuning_lr: float = 0.001,
    finetuning_lr_decay_milestones: str = "20",
    #initial_flops = 1e9,  # example initial FLOPs of unpruned model
    target_prune_ratio = 0.5,
    one_shot_final_flops = 1e9 * 0.6, # FLOPs of the final model obtained through one-shot pruning
    num_iterations = 5, # Pruning iterations
):
    
    # Set torch random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Setup logger
    logger, output_dir = setup_logger(
        output_dir="my_runs/" + time.strftime("%Y-%m-%d-%H:%M") + "/dynamic-iterative/" + str(target_prune_ratio) 
    )

    # Load model and dataset
    model, train_loader, test_loader, example_inputs, dataset_num_classes = setup_model_and_dataset(logger=logger)
    
    # Measure original model flops and parameter counts (before pruning)
    model.eval()
    initial_flops, initial_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    initial_acc, initial_val_loss = eval(model, test_loader)

    # Total computational budget
    total_budget = retraining_epochs * one_shot_final_flops # This is the cost of one-shot pruning

    # Initialize variables
    remaining_budget = total_budget
    current_flops = initial_flops

    # Store epochs for each iteration
    epochs_list = [0]
    flops_list = [current_flops]
    params_list = [initial_params]
    raw_accuracy_list = [initial_acc] # Model accuracy just after pruning
    regained_accuracy_list = [initial_acc] # Model accuracy after pruning and retraining
    # TODO also save loss...?

    # Calculate constant pruning ratio required to achieve target compression in the desired number of iterations
    pruning_ratio = 1 - math.pow((1 - target_prune_ratio), 1 / num_iterations)

    # Save start time (for runtime calculation)
    start_time = time.time()

    for i in range(num_iterations):
        # Set up pruner for each iteration (it is deleted at the end of each iteration too)
        pruner = get_pruner(model=model,
                            example_inputs=example_inputs,
                            dataset_num_classes=dataset_num_classes)
        
        # Prune the model and get new FLOPs value 
        pruned_params, new_flops = prune(model=model,
                                            example_inputs=example_inputs,
                                            pruner=pruner,
                                            test_loader=test_loader,
                                            logger=logger,
                                            prune_rate=pruning_ratio)
        
        # Evaluate pruned model performance
        pruned_acc, pruned_val_loss = eval(model, test_loader)
        #aux = random.uniform(pruning_ratio * 0.6, pruning_ratio)
        #print(f"{pruning_ratio} - {aux}")
        #new_flops = current_flops - current_flops * aux

        # Append results to list
        flops_list.append(new_flops)
        params_list.append(pruned_params)
        raw_accuracy_list.append(pruned_acc)
        
        # Calculate remaining budget after current iteration
        if i > 0:
            remaining_budget -= epochs_list[-1] * current_flops
        
        # Calculate the epochs for this iteration
        if i < num_iterations - 1:
            epochs = remaining_budget / new_flops / (num_iterations - i)

            # Floor epochs variable 
            epochs = int(epochs)

        else:
            # Last iteration, use up the remaining budget
            epochs = remaining_budget / new_flops

            # Round epochs variable on the last iteration
            epochs = round(epochs)

        # Retrain Pruned Model for the dynamically calculated number of epochs
        best_acc, last_acc = retrain(model=model,
                                     train_loader=train_loader,
                                     test_loader=test_loader,
                                     logger=logger,
                                     output_dir=output_dir,
                                     total_epochs=epochs,
                                     lr_decay_milestones=None)
        
        # Append results to list
        epochs_list.append(epochs)
        regained_accuracy_list.append(last_acc)
        
        # Update current FLOPs
        current_flops = new_flops

    # Finally, fine-tune the model, with lower lr
    best_acc, last_acc = retrain(model=model,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 logger=logger,
                                 output_dir=output_dir,
                                 total_epochs=finetuning_epochs,
                                 lr=finetuning_lr,
                                 lr_decay_milestones=finetuning_lr_decay_milestones)

    # Calculate Runtime
    runtime = str(timedelta( seconds=(time.time() - start_time) ))

    # Append results to list
    epochs_list.append(finetuning_epochs)
    flops_list.append(flops_list[-1])
    params_list.append(params_list[-1])
    raw_accuracy_list.append(best_acc)
    regained_accuracy_list.append(best_acc)

    # Aux variable
    budget_accumulator = 0

    # Print results
    for idx, epochs in enumerate(epochs_list[1:-1]): # Note, last element corresponds to finetuning iteration, and must not be considered in budget calculation
        logger.info(f"Iteration {idx + 1}: {epochs:.2f} epochs with {flops_list[idx + 1]:.3e} FLOPs - Speedup: x{flops_list[idx] / flops_list[idx + 1]:.3f}")
        budget_accumulator += epochs * flops_list[idx + 1]

    logger.info(f"Remaining budget after last iteration: {remaining_budget - epochs_list[-2] * current_flops:.3e}") # Last epochs value corresponds to fine-tuning, so we want the previous one in the list (-2)
    logger.info(f"Budget for one-shot pruning: {total_budget:.3e}")
    logger.info(f"Budget for iterative pruning: {budget_accumulator:.3e}")
    logger.info(f"Budget Error: {(budget_accumulator - total_budget) / total_budget * 100:.2f} %")

    logger.info(f"One-shot speedup: x{initial_flops / one_shot_final_flops:.2f}")
    logger.info(f"Iterative speedup: x{initial_flops / flops_list[-1]:.2f}")
    logger.info(f"Iterative final prune ratio: {(initial_params - params_list[-1]) / initial_params * 100.0:.2f}")
    logger.info(f"Iterative final accuracy: {regained_accuracy_list[-1] * 100.0:.2f}%")
    logger.info(f"Iterative accuracy drop: {(regained_accuracy_list[-1] - regained_accuracy_list[0]) * 100.0:.2f}%")
    logger.info(f"--- Total runtime: {runtime} (hours:min:sec) ---")

    # Save results in json
    results = {
        'seed': seed,
        'prune_rate': target_prune_ratio * 100.0,
        'prune_iterations': num_iterations,
        'pruned_params': params_list[-1],
        'pruned_flops': flops_list[-1],
        'best_acc': best_acc * 100.0,
        'acc_drop': (regained_accuracy_list[-1] - regained_accuracy_list[0]) * 100.0,
        'speed_up': initial_flops / flops_list[-1],
        'real_prune_ratio': (initial_params - params_list[-1]) / initial_params * 100.0,
        'runtime': runtime
    }

    save_results(output_dir=output_dir,
                 results=results)

    list_results = {
        'epochs_list': epochs_list,
        'flops_list': flops_list,
        'params_list': params_list,
        'raw_accuracy_list': raw_accuracy_list,
        'regained_accuracy_list': regained_accuracy_list,
    }

    save_results(output_dir=output_dir,
                 results=list_results,
                 file_name='list_results')
    
    # Reset logging module after each run of this function. This prevents unwanted logging
    # behaviour when this function is called consecutive times during a single script run
    logging.shutdown()
    reload(logging)

    return results

def one_shot_prune(seed: int = None,
                   prune_ratio: float = 0.5,
                   train_epochs: int = 100):
    
    if seed is not None:
        torch.manual_seed(seed)

    logger, output_dir = setup_logger(
        output_dir="my_runs/" + time.strftime("%Y-%m-%d-%H:%M") + "/one-shot/" + str(prune_ratio) 
    )

    model, train_loader, test_loader, example_inputs, dataset_num_classes = setup_model_and_dataset(logger=logger)

    # Measure original model flops and parameter counts (before pruning)
    model.eval()
    initial_flops, initial_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    initial_acc, initial_val_loss = eval(model, test_loader)
    
    pruner = get_pruner(model=model,
                        example_inputs=example_inputs,
                        dataset_num_classes=dataset_num_classes)
    
    # Save start time (for runtime calculation)
    start_time = time.time()
    
    pruned_params, pruned_flops = prune(model=model,
                                        example_inputs=example_inputs,
                                        pruner=pruner,
                                        test_loader=test_loader,
                                        logger=logger,
                                        prune_rate=prune_ratio)
    
    best_acc, last_acc = retrain(model=model,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 logger=logger,
                                 output_dir=output_dir,
                                 total_epochs=train_epochs)
    
    # Calculate Runtime
    runtime = str(timedelta( seconds=(time.time() - start_time) ))

    # Save results in json
    results = {
        'seed': seed,
        'prune_rate': prune_ratio * 100.0,
        'prune_iterations': 1,
        'pruned_params': pruned_params,
        'pruned_flops': pruned_flops,
        'best_acc': best_acc * 100.0,
        'acc_drop': (best_acc - initial_acc) * 100.0,
        'speed_up': initial_flops / pruned_flops,
        'real_prune_ratio': (initial_params - pruned_params) / initial_params * 100.0,
        'runtime': runtime
    }

    save_results(output_dir=output_dir,
                 results=results)
    
    # Reset logging module after each run of this function. This prevents unwanted logging
    # behaviour when this function is called consecutive times during a single script run
    logging.shutdown()
    reload(logging)

    return results

if __name__ == "__main__":

    prune_rates = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    

    one_shot_final_flops = 6258198.0
    
    # dynamic_epoch_allocation_pruning(
    #     seed=0,
    #     target_prune_ratio=0.95,
    #     one_shot_final_flops=one_shot_final_flops,
    #     num_iterations=5
    # )

    for rate in prune_rates:
        for seed in range(5):
            one_shot_prune(seed=seed,
                           prune_ratio=rate)
