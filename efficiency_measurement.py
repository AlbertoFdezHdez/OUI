import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

from functions import *
from models import *
from datasets import *

from main import load_model_and_data

import time

def train_f_time(model, train_data, wd):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device

    num_epochs = model.num_epochs

    val_acc = 0
    model.optimizer.param_groups[0]['weight_decay'] = wd
    lr = model.optimizer.param_groups[0]['lr']

    # Solo entrenamos una época
    epoch = 0

    model.train()
    running_loss = 0.0
    running_oui = 0.0
    total = 0
    correct = 0.0
    oui_time_accumulator = 0.0 

    progress_bar = tqdm(train_data, total=len(train_data),
                        desc=f"GPU = 0, WD = {wd:.6f}, epoch = {epoch+1}/{num_epochs}, LR = {lr:.6f}, last_val_acc = {val_acc:.2f}",
                        miniters=1, ncols=180, leave=False)

    total_start = time.time()  

    for batch_idx, (inputs, labels) in enumerate(progress_bar, start=1):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        model.optimizer.step()

        running_loss += loss.detach()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum()

        if batch_idx % 10 == 0:
            oui_start = time.time()  
            batch_oui = on_batch_OUI(model)
            oui_end = time.time()  

            running_oui += batch_oui
            oui_time_accumulator += (oui_end - oui_start)  

            train_loss = running_loss.item() / batch_idx
            oui = running_oui.item() / (batch_idx // 10)
            train_acc = 100. * correct.item() / total

            progress_bar.set_postfix(train_acc=train_acc, OUI=oui)

    total_end = time.time() 
    total_time = total_end - total_start

    progress_bar.close()

    print(f"\nTiempo total de la época: {total_time:.4f} segundos")
    print(f"Tiempo acumulado en OUI: {oui_time_accumulator:.4f} segundos")
    print(f"Porcentaje del tiempo total dedicado a OUI: {100. * oui_time_accumulator / total_time:.2f}%")



def main(index):
    model, train_data, val_data, name, wd_list = load_model_and_data(index)

    torch.cuda.set_device(0)  
    set_seed(42)
    train_f_time(model, train_data, wd=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, help="Índice del experimento.")
    args = parser.parse_args()

    main(args.index)