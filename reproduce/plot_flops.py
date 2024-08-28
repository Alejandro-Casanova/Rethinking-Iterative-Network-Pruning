import torch
import torch.nn as nn
from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis, flop_count_table
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load model
    model = resnet18(pretrained=True)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Measure FLOPs
    flops = FlopCountAnalysis(model, dummy_input)

    # Get FLOPs per module
    flop_dict = flops.by_module()
    module_names = list(flop_dict.keys())
    flops_per_module = list(flop_dict.values())

    # Plot pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(flops_per_module, labels=module_names, autopct='%1.1f%%', startangle=140)
    plt.title('FLOPs Distribution per Module')
    plt.axis('equal')
    plt.show()

    # Optional: print detailed FLOP count
    print(flop_count_table(flops))