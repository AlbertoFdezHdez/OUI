import torchvision
import torch
import os
import torchvision.transforms as transforms

def cifar100(batch_size):
    transform_train = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def download_tinyimagenet():
    import kaggle
    # Download latest version
    path_to_download = "./data/"
    dataset = "xiataokang/tinyimagenettorch"
    kaggle.api.dataset_download_files(dataset, path=path_to_download, unzip=True)

def load_tinyimagenet(batch_size=64):
    # Download
    if not os.path.isdir("./data/tiny-imagenet-200"):
        download_tinyimagenet()

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    trainset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=test_transform)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def download_imagenet1k():
    raise Exception("Download Imagenet1k dataset from: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php")

def load_imagenet1k(batch_size=64, size=224):
    '''
    Imagenet 1k dataset images have different sizes, if no resize is in place, the dataloader needs this collate_fn to be passed
    as an argument. More testing needed in this case.
    Run in node r4n02: replace flat -N1 with -w r4n02
    '''
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    if not os.path.isdir("/scratch1/jmiravet/imagenet1k/"):
        download_imagenet1k()

    # Transformaciones para el conjunto de entrenamiento
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),  # Recorte aleatorio y redimensionado
        transforms.RandomHorizontalFlip(),   # Volteo horizontal aleatorio
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Variaciones de color
        transforms.ToTensor(),               # Conversión a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
    ])

    # Transformaciones para el conjunto de validación/prueba
    test_transform = transforms.Compose([
        transforms.Resize(256),              # Redimensionado para mantener la relación de aspecto
        transforms.CenterCrop(size),         # Recorte central
        transforms.ToTensor(),               # Conversión a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
    ])

    # Conjuntos de datos
    trainset = torchvision.datasets.ImageNet(root="/scratch1/jmiravet/imagenet1k/", split="train", transform=train_transform)
    testset = torchvision.datasets.ImageNet(root="/scratch1/jmiravet/imagenet1k/", split="val", transform=test_transform)

    # Cargadores de datos
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader