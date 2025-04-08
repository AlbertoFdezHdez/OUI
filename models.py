import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def save_model(self, epoch, path='model.pth', verbose=True):
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)
        if verbose:
            print(f"Model saved to {path}")

    def load_model(self, path='model.pth', verbose=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        
        if verbose:
            print(f"Model loaded from {path}, starting from epoch {epoch}")
        return epoch
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)  # Primera capa ReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)  # Segunda capa ReLU
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)  # Aplicar la primera ReLU

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)  # Aplicar la segunda ReLU

        return out

class ResNet(BasicModel):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # Esta ReLU se mantiene para la capa inicial
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=100):
    return ResNet(BasicBlock, [3, 4, 23, 3], num_classes)

class ResNetTinyImagenet(BasicModel):
    def __init__(self, block, layers, num_classes=200):  # CAMBIO: num_classes=200 para TinyImageNet
        super(ResNet, self).__init__()
        self.in_channels = 64
        # CAMBIO: Modificar la primera capa convolucional
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Cambio de kernel_size=7, stride=2 a kernel_size=3, stride=1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # CAMBIO: Eliminar la capa de maxpool
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # CAMBIO: Eliminar la aplicación de maxpool
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18_TinyImageNet(num_classes=200):  # CAMBIO: num_classes=200 para TinyImageNet
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)






class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet152(ResNet):
    def __init__(self, num_classes=100):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return torch.cat([x, out], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        
        return out

class DenseNet(BasicModel):
    def __init__(self, num_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        # Dense blocks
        self.dense1 = self._make_dense_block(num_channels, num_blocks[0], growth_rate)
        num_channels += num_blocks[0] * growth_rate
        self.trans1 = self._make_transition(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)

        self.dense2 = self._make_dense_block(num_channels, num_blocks[1], growth_rate)
        num_channels += num_blocks[1] * growth_rate
        self.trans2 = self._make_transition(num_channels, int(num_channels * reduction))
        num_channels = int(num_channels * reduction)

        self.dense3 = self._make_dense_block(num_channels, num_blocks[2], growth_rate)
        num_channels += num_blocks[2] * growth_rate
        
        # Final batch norm and classifier
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, in_channels, nblock, growth_rate):
        layers = []
        for i in range(nblock):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return Transition(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        
        # First dense block
        out = self.dense1(out)
        out = self.trans1(out)
        
        # Second dense block
        out = self.dense2(out)
        out = self.trans2(out)
        
        # Third dense block
        out = self.dense3(out)
        
        # Final layers
        out = self.bn(out)
        out = self.relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

# Definir DenseNet-BC-100 con 3 bloques de tamaño 16, 16 y 16
def densenet_bc_100(num_classes=10):
    return DenseNet(num_blocks=[16, 16, 16], growth_rate=12, reduction=0.5, num_classes=num_classes)

class BasicBlockP(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlockP, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None

        # Shortcut (residual connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        
        out += self.shortcut(x)
        return out


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, num_classes, dropout_rate=0.0):
        super(PyramidNet, self).__init__()
        self.in_planes = 16
        self.alpha = alpha
        self.addrate = alpha / (3 * ((depth - 2) // 6))
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layers
        self.layer1 = self._make_layer((depth - 2) // 6, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer((depth - 2) // 6, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer((depth - 2) // 6, stride=2, dropout_rate=dropout_rate)
        
        # Final layers
        self.bn_final = nn.BatchNorm2d(self.in_planes)
        self.relu_final = nn.ReLU(inplace=True)
        self.linear = nn.Linear(self.in_planes, num_classes)

    def _make_layer(self, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            out_planes = int(round(self.in_planes + self.addrate))
            layers.append(BasicBlockP(self.in_planes, out_planes, stride, dropout_rate))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        # Layer 1
        out = self.layer1(out)
        
        # Layer 2
        out = self.layer2(out)
        
        # Layer 3
        out = self.layer3(out)
        
        # Final batch norm, ReLU, and pooling
        out = self.bn_final(out)
        out = self.relu_final(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        
        # Fully connected layer
        out = self.linear(out)
        
        return out


class WideResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(WideResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        # Calcular número de capas por bloque y crear capas ReLU únicas
        n = (depth - 4) // 6
        k = widen_factor

        # Primeras capas
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16 * k, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(32 * k, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(64 * k, n, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(64 * k)
        self.relu1 = nn.ReLU(inplace=True)

        # Fully connected layer
        self.linear = nn.Linear(64 * k, num_classes)

    def _make_layer(self, out_planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(WideResNetBlock(self.in_planes, out_planes, stride, dropout_rate))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def WideResNet28_10(num_classes=100, dropout_rate=0.3):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropout_rate=dropout_rate)

def WideResNet28_20(num_classes=200, dropout_rate=0.3):
    return WideResNet(depth=28, widen_factor=20, num_classes=num_classes, dropout_rate=dropout_rate)

def WideResNet50_20(num_classes=200, dropout_rate=0.3):
    return WideResNet(depth=50, widen_factor=20, num_classes=num_classes, dropout_rate=dropout_rate)

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Embedding de los parches usando una convolución
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        
        # Token de clasificación y embeddings posicionales
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Bloques del Transformer
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Cabeza de clasificación
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.size(0)
        # Extraer parches: [B, embed_dim, H_patch, W_patch]
        x = self.patch_embed(x)
        # Reorganizar: aplanar parches
        x = x.flatten(2)       # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Concatenar el token de clasificación
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Sumar el embedding posicional y aplicar dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pasar por cada bloque del Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Usar el token de clasificación para la predicción final
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout)
    
    def forward(self, x):
        # x: [B, num_tokens, embed_dim]
        # Normalización y atención multi-cabeza (la función attn requiere [seq_len, B, embed_dim])
        x_norm = self.norm1(x)
        x_norm = x_norm.transpose(0, 1)  # [num_tokens, B, embed_dim]
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        attn_output = attn_output.transpose(0, 1)  # [B, num_tokens, embed_dim]
        x = x + self.drop1(attn_output)
        # Bloque MLP
        x = x + self.mlp(self.norm2(x))
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(MLP, self).__init__()
        hidden_dim = embed_dim * 4  # Dimensión intermedia típica
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # Función de activación definida como capa (en lugar de usar F.gelu en forward)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)  # Output: (6, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)  # Output: (16, 14, 14)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=200):
        super(EfficientNetB0, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)