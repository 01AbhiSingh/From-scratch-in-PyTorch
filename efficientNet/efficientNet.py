import torch
import torch.nn as nn
import math

def swish(x):
    #Activation function
    return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    #squeeze and excitation block
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

class MBConvBlock(nn.Module):
    """mobile inverted besidual bottleneck block"""
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, 
                 se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expansion phase
        expanded_channels = int(in_channels * expand_ratio)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 
                      kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )

        # Squeeze-and-Excitation
        self.se_block = SEBlock(expanded_channels, reduction_ratio=int(1/se_ratio)) if se_ratio else nn.Identity()

        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def drop_connect(self, x, drop_connect_rate, training):
        if not training:
            return x
        keep_prob = 1 - drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x * binary_tensor / keep_prob

    def forward(self, x):
        identity = x

        # Expansion and Depthwise Convolution
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)

        x = self.se_block(x)

        # Pointwise Convolution
        x = self.pointwise_conv(x)

        # Residual connection
        if self.use_residual:
            if self.training and self.drop_connect_rate:
                x = self.drop_connect(x, self.drop_connect_rate, self.training)
            x += identity

        return x

class EfficientNet(nn.Module):
    #efficientNet base model
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000):
        super().__init__()
        
        # base configuration for efficientNet-B0
        base_blocks = [
            # [expand_ratio, channels, repeats, stride, kernel_size]
            [1, 32, 1, 2, 3],
            [6, 16, 2, 2, 3],
            [6, 24, 2, 2, 5],
            [6, 40, 3, 2, 3],
            [6, 80, 3, 1, 5],
            [6, 112, 4, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]

        # scaling
        channels = [round(x * width_mult) for x in [32, 16, 24, 40, 80, 112, 192, 320]]
        repeats = [math.ceil(x * depth_mult) for x in [1, 2, 2, 3, 3, 4, 4, 1]]

        # initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        # build blocks
        self.blocks = nn.ModuleList()
        input_channels = channels[0]

        for i, (expand_ratio, out_channels, num_repeat, stride, k_size) in enumerate(zip(base_blocks, channels[1:], repeats, [2, 2, 2, 2, 1, 1, 2, 1], [3, 3, 5, 3, 5, 5, 5, 3])):
            for j in range(num_repeat):
                stride = stride if j == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        input_channels, out_channels, 
                        expand_ratio=expand_ratio, 
                        stride=stride, 
                        kernel_size=k_size
                    )
                )
                input_channels = out_channels

        # final layers
        final_channels = 1280
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channels, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_efficientnet(model_variant='b0', num_classes=1000):
    width_mult_dict = {
        'b0': 1.0, 'b1': 1.0, 'b2': 1.1, 'b3': 1.2, 
        'b4': 1.4, 'b5': 1.6, 'b6': 1.8, 'b7': 2.0
    }
    depth_mult_dict = {
        'b0': 1.0, 'b1': 1.1, 'b2': 1.2, 'b3': 1.3, 
        'b4': 1.4, 'b5': 1.6, 'b6': 1.8, 'b7': 2.0
    }
    
    variant = model_variant.lower()
    width_mult = width_mult_dict.get(variant, 1.0)
    depth_mult = depth_mult_dict.get(variant, 1.0)
    
    return EfficientNet(width_mult=width_mult, depth_mult=depth_mult, num_classes=num_classes)
