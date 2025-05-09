REM Experiment 1:
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento1 -ltx -op -f one-shot,dynamic-iterative -plt acc_drop
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento1 -ltx -op -f one-shot,dynamic-iterative -plt speed_up
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento1 -ltx -op -f one-shot,dynamic-iterative -plt acc_drop,speed_up
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento1 -ltx -op -f one-shot,dynamic-iterative -plt runtime
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento1 -ltx -op -f one-shot,dynamic-iterative -plt runtime,speed_up

REM Experiment 2:
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento2 -ltx -op -f one-shot,dynamic-iterative-flops -fs -plt speed_up
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento2 -ltx -op -f one-shot,dynamic-iterative-flops -fs -plt acc_drop
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento2 -ltx -op -f one-shot,dynamic-iterative-flops -fs -plt acc_drop,speed_up
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento2 -ltx -op -f one-shot,dynamic-iterative-flops -fs -plt acc_drop,real_prune_ratio
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento2 -ltx -op -f one-shot,dynamic-iterative-flops -fs -plt real_prune_ratio
python .\reproduce\plot_results.py -o .\reproduce\my_results\experimento2 -ltx -op -f one-shot,dynamic-iterative-flops -fs -plt runtime