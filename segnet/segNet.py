import torch 
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Encoder block with convolutions and max pooling
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(EncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
    def forward(self, x):
        """
        Forward pass storing max pooling indices
        Returns:
            Tuple of (output tensor, pooling indices)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x, indices = self.pool(x)
        return x, indices

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        
        
class SegNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=21):
        """
        Initialize SegNet architecture
        Args:
            input_channels: Number of input image channels (default: 3 for RGB)
            num_classes: Number of output classes (default: 21 for PASCAL VOC)
        """
        super(SegNet, self).__init__()
        
        # Important architectural detail: Base number of features
        base_features = 64
        
        # Encoder blocks
        # Each block increases the number of features while reducing spatial dimensions
        self.enc1 = EncoderBlock(input_channels, base_features)  # 64 features
        self.enc2 = EncoderBlock(base_features, base_features*2)  # 128 features
        self.enc3 = EncoderBlock(base_features*2, base_features*4)  # 256 features
        self.enc4 = EncoderBlock(base_features*4, base_features*8)  # 512 features
        self.enc5 = EncoderBlock(base_features*8, base_features*8)  # 512 features
        
        # Decoder blocks
        # Each block decreases the number of features while increasing spatial dimensions
        self.dec5 = DecoderBlock(base_features*8, base_features*8)
        self.dec4 = DecoderBlock(base_features*8, base_features*4)
        self.dec3 = DecoderBlock(base_features*4, base_features*2)
        self.dec2 = DecoderBlock(base_features*2, base_features)
        self.dec1 = DecoderBlock(base_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        # Encoder Path with pooling indices
        x, ind1 = self.enc1(x)
        x, ind2 = self.enc2(x)
        x, ind3 = self.enc3(x)
        x, ind4 = self.enc4(x)
        x, ind5 = self.enc5(x)
        
        # Decoder Path using stored pooling indices
        x = self.dec5(x, ind5)
        x = self.dec4(x, ind4)
        x = self.dec3(x, ind3)
        x = self.dec2(x, ind2)
        x = self.dec1(x, ind1)
        
        return x
    
def test_segnet():
    """
    Test function to verify SegNet implementation
    """
    # Create a sample input tensor
    batch_size, channels, height, width = 1, 3, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    
    # Initialize SegNet
    model = SegNet(input_channels=3, num_classes=21)
    
    # Forward pass
    output = model(x)
    
    # Verify output shape
    expected_shape = (batch_size, 21, height, width)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("SegNet test passed successfully!")
    return output.shape

if __name__ == "__main__":
    test_segnet()
