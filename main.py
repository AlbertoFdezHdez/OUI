import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

from functions import *
from models import *
from datasets import *

def load_model_and_data(index):
    if index == 2:
        name = "CIFAR100_DenseNet-BC-100"
        wd_list = np.logspace(-5, -1, 9)[1:-1]

        batch_size = 64
        lr = 0.01

        train_data, val_data = cifar100(batch_size)    
        model = densenet_bc_100(num_classes=100)
        model = CaptureModel(model)
        print(f"Number of ReLU layers: {len(model.layers_to_capture)}")

        model.num_epochs = 200
        model.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=model.num_epochs)
        model.loss = torch.nn.CrossEntropyLoss()

    elif index == 5:
        name = "ImageNet1K_ResNet34"
        wd_list = np.logspace(-6, -2, 9)[1:-1]

        batch_size = 256
        warmup_epochs = 5
        lr = 0.1

        train_data, val_data = load_imagenet1k(batch_size=64, size=224)
        model = resnet34(num_classes=1000)
        model = CaptureModel(model)
        print(f"Number of ReLU layers: {len(model.layers_to_capture)}")

        model.num_epochs = 90
        model.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model.loss = torch.nn.CrossEntropyLoss()
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(model.optimizer, start_factor=1e-2, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=(model.num_epochs - warmup_epochs))
        model.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(model.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    elif index == 6:
        name = "TinyImageNet_EfficientNetB0"
        wd_list = np.logspace(-5, -2, 7)

        batch_size = 64
        lr = 0.01
        warmup_epochs = 15


        train_data, val_data = load_tinyimagenet(batch_size)
        model = EfficientNetB0(num_classes=200)
        model = CaptureModel(model)
        print(model)
        print(f"Number of SiLU layers: {len(model.layers_to_capture)}")

        model.num_epochs = 150
        model.loss = nn.CrossEntropyLoss()
        model.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(model.optimizer, start_factor=1e-1, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=(model.num_epochs - warmup_epochs))
        model.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(model.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    else:
        raise ValueError(f"Índice {index} no soportado.")
    return model, train_data, val_data, name, wd_list

def main(index, num):
    model, train_data, val_data, name, wd_list = load_model_and_data(index)
    base_filename = get_unique_basename(f"Resultados_Indice{index}_{name}")

    if num < 0 or num >= len(wd_list):
        raise ValueError(f"El número {num} está fuera del rango válido (0 a {len(wd_list) - 1}).")
    
    torch.cuda.set_device(0)  
    temp_filename = f"{base_filename}_num={num}.txt"  # Incluir el número en el archivo de salida
    
    with open(temp_filename, "a") as file:
        set_seed(42) 
        train_f(model, train_data, val_data, wd_list[num], file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, help="Índice del experimento.")
    parser.add_argument("--num", type=int, required=True, help="Número del experimento (índice de wd_list).")
    args = parser.parse_args()

    main(args.index, args.num)