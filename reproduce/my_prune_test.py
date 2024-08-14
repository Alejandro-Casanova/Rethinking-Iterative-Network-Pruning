import argparse
from logging import Logger
import math
import os
import random
from scipy.optimize import fsolve

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp

import engine.utils as utils
import registry

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
    output_dir = os.path.join(output_dir, dataset, mode, logger_name)
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
          device: str = "cuda"):
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
    logger.info(model)
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
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay, #if pruner is None else 0,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
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

def retrain(model,
            train_loader,
            test_loader,
            logger: Logger,
            output_dir: str,
            device: str = "cuda",
            total_epochs: int = 100,
            lr: float = 0.01,
            lr_decay_milestones: str = "60,80"
            ):
    
    logger.info("Re-training...")
    train_model(
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

def dynamic_allocation(
    initial_epochs = 100,
    initial_flops = 1e9,  # example initial FLOPs of unpruned model
    target_pruning_ratio = 0.5,
    one_shot_final_flops = 1e9 * 0.6, # Placeholder, must be replaced with real value after trying one-shot pruning!
    num_iterations = 5,
):
    
    # Total computational budget
    total_budget = initial_epochs * one_shot_final_flops # This is the cost of one-shot pruning

    # Initialize variables
    remaining_budget = total_budget
    current_flops = initial_flops

    # Store epochs for each iteration
    epochs_list = []
    flops_list = [current_flops]

    pruning_ratio = 1 - math.pow((1 - target_pruning_ratio), 1 / num_iterations)

    for i in range(num_iterations):
        # Simulate pruning to get new FLOPs (this is a placeholder, replace with actual pruning function)
        aux = random.uniform(pruning_ratio * 0.6, pruning_ratio)
        #print(f"{pruning_ratio} - {aux}")
        new_flops = current_flops - current_flops * aux

        flops_list.append(new_flops)
        
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

        # Append to list
        epochs_list.append(epochs)
        
        # Update current FLOPs
        current_flops = new_flops

    budget_accumulator = 0

    # Print results
    for idx, epochs in enumerate(epochs_list):
        print(f"Iteration {idx + 1}: {epochs:.2f} epochs with {flops_list[idx + 1]:.3e} FLOPs - Speedup: x{flops_list[idx] / flops_list[idx + 1]:.3f}")
        budget_accumulator += epochs * flops_list[idx + 1]

    print(f"Remaining budget after last iteration: {remaining_budget - epochs_list[-1] * current_flops:.3e}")
    print(f"Budget for one-shot pruning: {total_budget:.3e}")
    print(f"Budget for iterative pruning: {budget_accumulator:.3e}")
    print(f"Budget Error: {(budget_accumulator - total_budget) / total_budget * 100:.2f} %")

    print(f"One-shot speedup: x{one_shot_final_flops / initial_flops:.2f}")
    print(f"Iterative speedup: x{flops_list[-1] / initial_flops:.2f}")
    
def main(seed: int = None,
         prune_rate: float = 0.5):
    
    if seed is not None:
        torch.manual_seed(seed)

    logger, output_dir = setup_logger()

    model, train_loader, test_loader, example_inputs, dataset_num_classes = setup_model_and_dataset(logger=logger)

    for x in range(2):
        pruner = get_pruner(model=model,
                            example_inputs=example_inputs,
                            dataset_num_classes=dataset_num_classes)
        
        prune(model=model,
            example_inputs=example_inputs,
            pruner=pruner,
            test_loader=test_loader,
            logger=logger,
            prune_rate=prune_rate)
        
        retrain(model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                logger=logger,
                output_dir=output_dir,
                total_epochs=10)

if __name__ == "__main__":
    main(seed=1)

    #dynamic_allocation(initial_epochs=200, num_iterations=1)
