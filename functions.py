import math
from tqdm import tqdm
import torch
import torch.nn as nn
import itertools
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rounded(x): # Muestra por pantalla el elemento x redondeado a 4 cifras sifnificativas, en formato 1.234e-5
    if math.isinf(x): 
        return 'inf'
    if math.isnan(x):
        return x
    p = 4
    x = float(x)
    if x == 0.:
        return "0." + "0"*(p-1)
    out = []
    if x < 0:
        out.append("-")
        x = -x
    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)
    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)
    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1
    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1
    m = "%.*g" % (p, n)
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)
    return "".join(out)

def concatenate_and_cleanup(base_filename, temp_files):
    base_filename = f"{base_filename}.txt"
    with open(base_filename, "w") as final_file:
        for temp_file in sorted(temp_files):
            with open(temp_file, "r") as tf:
                final_file.write(tf.read())
            os.remove(temp_file)  # Borrar el archivo temporal

def get_unique_basename(base_name):
    counter = 0
    file_name = f"{base_name}"
    while os.path.exists(file_name+'_num=7.txt') or os.path.exists(file_name+'.txt'):
        counter += 1
        file_name = f"{base_name}({counter})"
    return base_name

def read(nombre, vector, archivo):  # Añadimos el parámetro archivo
    """Lee un vector y lo guarda en el archivo en el formato nombre = [ entrada1, entrada2, ... ]"""
    archivo.write(nombre + " = [")
    for i in range(len(vector)):
        if i < len(vector) - 1:
            archivo.write(f"{rounded(vector[i])}, ")  # Escribe en el archivo, no imprime
        else:
            archivo.write(f"{rounded(vector[i])}")  # No coma al final
    archivo.write("]\n")  # Añadimos una nueva línea al final

class CaptureLayer(torch.nn.Module):
    def __init__(self, layer):
        super(CaptureLayer, self).__init__()
        self.layer = layer

    def forward(self, input):
        output = self.layer(input)
        self.captured_output = output > 0
        return output

def convert_model(model):
    layers_to_capture = []
    layers_names = []
    for key, module in model._modules.items():
        if type(module) is CaptureLayer:
            pass
        elif type(module) is nn.ReLU:
            new_relu = CaptureLayer(nn.ReLU(inplace=module.inplace))
            layers_to_capture += [new_relu]
            setattr(model, key, new_relu)
            layers_names += [key]
        elif type(module) is nn.SiLU:
            new_silu = CaptureLayer(nn.SiLU(inplace=module.inplace))
            layers_to_capture += [new_silu]
            setattr(model, key, new_silu)
            layers_names += [key]
        elif type(module) is nn.GELU:
            new_gelu = CaptureLayer(nn.GELU())
            layers_to_capture += [new_gelu]
            setattr(model, key, new_gelu)
            layers_names += [key]
        else:
            _, new_layers_to_capture, new_layers_names = convert_model(module)
            if len(new_layers_to_capture) > 0:
               layers_to_capture += [*new_layers_to_capture]
               for l in new_layers_names:
                   layers_names += [f"{key}.{l}"]
    return model, layers_to_capture, layers_names

def CaptureModel(model):
    model, layers_to_capture, layers_names = convert_model(model)
    model.layers_to_capture = layers_to_capture
    model.layers_names = layers_names
    return model

def on_batch_OUI(model, k = 8*7//2):
    if not hasattr(model, "oui_comb"):
        num_rows = model.layers_to_capture[0].captured_output.shape[0]
        comb = list(itertools.combinations(range(num_rows), 2))
        if k is not None: 
            comb = random.sample(comb, np.min([len(comb),k]))
        else:
            k = len(comb)
        model.oui_comb = torch.tensor(comb, dtype=torch.long, device=model.device)
        model.k = k
    
    oui_list = torch.empty((model.k, len(model.layers_to_capture)), device=model.device)
    limit_list = torch.empty(len(model.layers_to_capture), device = model.device)
    
    for l, layer in enumerate(model.layers_to_capture):
        sub_matrix = layer.captured_output.reshape(layer.captured_output.shape[0], -1)
        limit = sub_matrix.shape[1]//2
        hamming_distances = torch.sum(sub_matrix[model.oui_comb[:, 0]] != sub_matrix[model.oui_comb[:, 1]], dim=1)
        oui_list[:, l] = torch.clamp(hamming_distances, max=limit) 
        limit_list[l] = limit
    oui = ( oui_list / limit_list ).mean()

    # sigma = torch.std( oui_list / limit_list , dim=0, unbiased=True)
    # print(len(sigma))
    # n = (1.96 / 0.05 * sigma )**2
    # print(f'Valor de n mínimo = {torch.min(n)}, máximo = {torch.max(n)}, medio = {torch.mean(n)}.')
    # exit()
    return oui

class AutomaticWD_OUI():
    def __init__(self, model, weight_decay_initial=1e-4, num_epochs=100, no_wd_epochs=None, initial_epochs = None, ideal_oui = 0.55):
        self.model = model
        self.optimizer = model.optimizer
        self.weight_decay_initial = weight_decay_initial
        self.num_epochs = num_epochs
        self.no_wd_epochs = no_wd_epochs if no_wd_epochs is not None else int(0.1 * num_epochs)
        self.initial_epochs = initial_epochs if initial_epochs is not None else int(0.1 * num_epochs)
        self.ideal_oui = ideal_oui

        self.factor = 10 ** (10  / self.num_epochs)
        print('Factor sobre el cual dividir el OUI:', self.factor)
        self.decrease_factor = self.factor

        self.lower_limit_oui = ideal_oui

        self.initial_oui = None

        self.start = True
        self.last_training_loss = float('inf')

        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = self.weight_decay_initial

    def step(self, epoch, oui, training_loss):
        if self.start: # En la fase inicial
            if epoch > self.initial_epochs: 
                self.start = False # Si nos pasamos de las épocas iniciales, salimos de esta fase
            else:
                if self.initial_oui == None: self.initial_oui = oui
                
                if oui > self.initial_oui + 0.2 or oui < self.initial_oui - 0.1 : 
                    self.start = False # Si el OUI ha aumentado más de 0.2 o disminuido más de 0.1 desde el principio, salimos de la fase inicial para iniciar correcciones
                else:             
                    print('>>>> Épocas iniciales: distancia entre valores de OUI de', oui-self.initial_oui)
            
        if oui < 0.55 and epoch < self.initial_epochs :
            print(">>>> OUI < 0.55: Reiniciando el entrenamiento.")
            for layer in self.model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] /= 10
                print(f">>>> Weight decay reinicializado a {param_group['weight_decay']}.")
            
            self.start = True
            self.initial_oui = None
            self.last_training_loss = float('inf')
            self.lower_limit_oui = self.ideal_oui
            self.decrease_factor = self.factor 

        if not self.start: # En la fase no inicial
            if epoch >= self.num_epochs - self.no_wd_epochs: # Si llegamos al final de las épocas, iremos bajando el WD cada época hasta dejarlo 10^4 veces más pequeño
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] /= self.factor ** 4
                    print('>>>> Épocas finales: Weight decay disminuyendo a', param_group['weight_decay'])

            elif oui < self.lower_limit_oui: # Si el OUI es bajo
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] /= self.decrease_factor
                    print('>>>> Weight decay disminuyendo a', param_group['weight_decay'])
                self.lower_limit_oui -= 0.025 # El OUI no sube aunque corrijamos el WD, con lo cual hay que bajar el umbral para mantenerse ahí
                self.decrease_factor *= self.factor # Cada vez que haya que subir el umbral mínimo del OUI seremos cada vez más duros bajando el WD
                print(f">>>> Nuevo valor límite máximo de OUI de {self.lower_limit_oui} y factor de decrecimiento del WD aumentado a {self.decrease_factor}.")
            elif oui > self.ideal_oui + 0.1 and training_loss < self.last_training_loss: # Si el OUI es alto y no ha empeorado la precisión en entrenamiento
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] *= self.factor
                    print('>>>> Weight decay aumentando a', param_group['weight_decay'])
                    self.last_training_loss = training_loss

def train_f(model, train_data, val_data, wd, file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device

    num_epochs = model.num_epochs

    loss_vector = []
    oui_vector = []
    val_loss_vector = []
    lr_vector = []
    val_acc_vector = []

    val_acc = 0
    model.optimizer.param_groups[0]['weight_decay'] = wd
    lr = model.optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_oui = 0.0
        total = 0
        correct = 0.0
 
        progress_bar = tqdm(train_data, total=len(train_data),
        desc =f"GPU = 0, WD = {wd:.6f}, epoch = {epoch+1}/{num_epochs}, LR = {lr:.6f}, last_val_acc = {val_acc:.2f}",
        miniters=1, ncols = 180, leave=False)
        
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
                batch_oui = on_batch_OUI(model) 
                running_oui += batch_oui

                train_loss = running_loss.item() / batch_idx 
                oui = running_oui.item() / (batch_idx // 10) 
                train_acc = 100. * correct.item() / total

                progress_bar.set_postfix(train_acc=train_acc, OUI=oui)    

        progress_bar.close()
        
        model.eval()
        val_loss = 0.0
        correct_top1 = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_data:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = model.loss(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                correct_top1 += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_data)
        val_acc = 100. * correct_top1 / total  

        lr = model.optimizer.param_groups[0]['lr']

        if isinstance(model.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            model.lr_scheduler.step(val_loss)
        else:
            model.lr_scheduler.step()

        loss_vector.append(train_loss)
        val_loss_vector.append(val_loss)
        lr_vector.append(lr)
        oui_vector.append(oui)
        val_acc_vector.append(val_acc)

    file.write(f"# WD={wd} training is complete. max_val_acc={np.max(val_acc_vector)}%. \n")

    for nombre, v in [("loss", loss_vector), ("val_loss", val_loss_vector), ("OUI", oui_vector), ("val_acc", val_acc_vector), ("LR", lr_vector)]:
        read(nombre, v, file)
    
    file.write(r"label = r'\textbf{WD = \$ ? \cdot 10^{-?} \$, MVA = ?? \%}'")
    file.write(f"\n")
    file.write(f"plot_all_metrics(loss, val_loss, OUI, label, filename='ind?_wd=?')\n")
    file.write(f"\n")