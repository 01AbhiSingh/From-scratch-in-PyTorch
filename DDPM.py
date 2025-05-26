import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import random 

# --- Utility for setting seed ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NoiseScheduler(nn.Module):
    def __init__(self, start_beta=0.0001, end_beta=0.02, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps

        self.betas = torch.linspace(start_beta, end_beta, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original_images, noise, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        noisy_images = sqrt_alphas_cumprod_t * original_images + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_images

    def sample_prev_timestep(self, xt, noise_prediction, t):
        x0_prediction = (xt - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise_prediction) / \
                        self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_prediction = torch.clamp(x0_prediction, -1.0, 1.0)

        mean = (xt - self.betas[t].view(-1, 1, 1, 1) * noise_prediction / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)) / \
               self.alphas[t].sqrt().view(-1, 1, 1, 1)

        variance = self.posterior_variance[t].view(-1, 1, 1, 1)

        if t[0] == 0:
            return mean, x0_prediction
        else:
            z = torch.randn_like(xt)
            xt_minus_1 = mean + torch.sqrt(variance) * z
            return xt_minus_1, x0_prediction


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.linear1 = nn.Linear(time_emb_dim, time_emb_dim * 4)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_emb_dim * 4, time_emb_dim * 4)

    def forward(self, t):
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim * 4, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_proj_out = self.time_proj(t_emb)
        h = h + time_proj_out[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        h = self.norm(x)
        h = h.view(batch_size, channels, H * W).transpose(1, 2)
        h, _ = self.attention(h, h, h)
        h = h.transpose(1, 2).view(batch_size, channels, H, W)
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_resnet_blocks=1, num_heads=8, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_in_channels = in_channels
        for i in range(num_resnet_blocks):
            self.blocks.append(ResnetBlock(current_in_channels, out_channels, time_emb_dim))
            self.blocks.append(SelfAttentionBlock(out_channels, num_heads))
            current_in_channels = out_channels 

        if downsample:
            
            self.downsample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample_layer = nn.Identity()

    def forward(self, x, t_emb):
       
        skip_connection_output = None 
        
        for block in self.blocks:
            if isinstance(block, ResnetBlock):
                x = block(x, t_emb)
            else: # SelfAttentionBlock
                x = block(x)
            skip_connection_output = x 
        
        x = self.downsample_layer(x)
        return x, skip_connection_output
        

class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, num_resnet_blocks=1, num_heads=8):
        super().__init__()
        self.resnet1 = ResnetBlock(channels, channels, time_emb_dim)
        
        self.attn_resnet_blocks = nn.ModuleList()
        for _ in range(num_resnet_blocks):
            self.attn_resnet_blocks.append(SelfAttentionBlock(channels, num_heads))
            self.attn_resnet_blocks.append(ResnetBlock(channels, channels, time_emb_dim))

    def forward(self, x, t_emb):
        x = self.resnet1(x, t_emb)
        for block in self.attn_resnet_blocks:
            if isinstance(block, ResnetBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_resnet_blocks=1, num_heads=8, upsample=True):
        super().__init__()
       
        self.upsample_layer = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        
        self.blocks = nn.ModuleList()
       
        self.blocks.append(ResnetBlock(in_channels + out_channels, out_channels, time_emb_dim)) 
        self.blocks.append(SelfAttentionBlock(out_channels, num_heads))

        for i in range(num_resnet_blocks - 1): 
            self.blocks.append(ResnetBlock(out_channels, out_channels, time_emb_dim))
            self.blocks.append(SelfAttentionBlock(out_channels, num_heads))

    def forward(self, x, skip_connection, t_emb):
        x = self.upsample_layer(x)
        
        if x.shape[-2:] != skip_connection.shape[-2:]:
            target_h, target_w = skip_connection.shape[-2:]
            x_h, x_w = x.shape[-2:]
            
            diff_h = x_h - target_h
            diff_w = x_w - target_w
            
            if diff_h > 0 or diff_w > 0:

                x = x[:, :, diff_h//2 : x_h - diff_h//2, diff_w//2 : x_w - diff_w//2]
            elif diff_h < 0 or diff_w < 0:

                padding_h = (abs(diff_h)//2, abs(diff_h) - abs(diff_h)//2)
                padding_w = (abs(diff_w)//2, abs(diff_w) - abs(diff_w)//2)
                x = F.pad(x, (padding_w[0], padding_w[1], padding_h[0], padding_h[1]))


        x = torch.cat([x, skip_connection], dim=1) # Concatenate along channel dimension

        for block in self.blocks:
            if isinstance(block, ResnetBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256,
                 down_channels=[64, 128, 256], mid_channels=[256],
                 up_channels=[256, 128, 64], num_resnet_blocks=1, num_heads=8):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)

        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)):
            in_c = down_channels[i-1] if i > 0 else down_channels[0]
            out_c = down_channels[i]
            downsample = True if i < len(down_channels) - 1 else False
            self.down_blocks.append(DownBlock(in_c, out_c, time_emb_dim, num_resnet_blocks, num_heads, downsample))

        self.mid_block = MidBlock(down_channels[-1], time_emb_dim, num_resnet_blocks, num_heads)

        self.up_blocks = nn.ModuleList()
        for i in range(len(up_channels)):
            in_c_upsample = down_channels[-1] if i == 0 else up_channels[i-1]
            
            out_c = up_channels[i]
            
            upsample = True if i < len(up_channels) -1 else False 

            self.up_blocks.append(UpBlock(in_c_upsample, out_c, time_emb_dim, num_resnet_blocks, num_heads, upsample))


        self.norm_out = nn.GroupNorm(32, up_channels[-1])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x = self.conv_in(x)

        skip_connections = []
        for down_block in self.down_blocks:
            x, skip_conn_output = down_block(x, t_emb)
            skip_connections.append(skip_conn_output) 

        x = self.mid_block(x, t_emb)

        for i, up_block in enumerate(self.up_blocks):
            skip_idx = len(skip_connections) - 1 - i
            skip_conn = skip_connections[skip_idx]
            x = up_block(x, skip_conn, t_emb)

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        return image

def train_ddpm(model, scheduler, dataloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for i, images in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]

            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()

            noise = torch.randn_like(images, device=device)

            noisy_images = scheduler.add_noise(images, noise, t)

            predicted_noise = model(noisy_images, t)

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

def sample_ddpm(model, scheduler, num_samples, img_size, channels, device):
    model.eval()
    with torch.no_grad():
        xt = torch.randn(num_samples, channels, img_size, img_size, device=device)

        for i in reversed(range(scheduler.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            noise_prediction = model(xt, t)
            
            xt, x0_prediction = scheduler.sample_prev_timestep(xt, noise_prediction, t)

            if (i + 1) % 100 == 0:
                print(f"Sampling timestep: {i+1}")

        generated_images = (xt.clamp(-1., 1.) + 1) * 0.5 
        return generated_images


if __name__ == "__main__":
    set_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMG_SIZE = 28
    CHANNELS = 1
    TIME_EMB_DIM = 256
    DOWN_CHANNELS = [64, 128, 256, 512]
    MID_CHANNELS = [512] # This should usually match the last down_channel value
    UP_CHANNELS = [512, 256, 128, 64] # Reversed order of down_channels for upsampling
    NUM_RESNET_BLOCKS = 1 
    NUM_HEADS = 8
    TIMESTEPS = 1000
    START_BETA = 0.0001
    END_BETA = 0.02

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    
    dummy_img_dir = "images/mnist"
    os.makedirs(dummy_img_dir, exist_ok=True)
    for k in range(10):
        dummy_img = Image.fromarray(np.random.randint(0, 256, (28, 28), dtype=np.uint8))
        dummy_img.save(os.path.join(dummy_img_dir, f"dummy_img_{k}.png"))

    scheduler = NoiseScheduler(start_beta=START_BETA, end_beta=END_BETA, timesteps=TIMESTEPS).to(device)
    model = Unet(in_channels=CHANNELS, out_channels=CHANNELS, time_emb_dim=TIME_EMB_DIM,
                 down_channels=DOWN_CHANNELS, mid_channels=MID_CHANNELS, up_channels=UP_CHANNELS,
                 num_resnet_blocks=NUM_RESNET_BLOCKS, num_heads=NUM_HEADS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = CustomImageDataset(img_dir=dummy_img_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("Starting DDPM training...")
    train_ddpm(model, scheduler, dataloader, optimizer, NUM_EPOCHS, device)
    print("Training complete.")

    torch.save(model.state_dict(), "ddpm_model.pth")
    print("Model saved to ddpm_model.pth")

    print("Starting DDPM sampling...")
    NUM_SAMPLES = 4
    generated_images = sample_ddpm(model, scheduler, NUM_SAMPLES, IMG_SIZE, CHANNELS, device)
    
    for k in range(NUM_SAMPLES):
        img_tensor = generated_images[k].cpu().squeeze(0)
        img_pil = transforms.ToPILImage()(img_tensor)
        img_pil.save(f"generated_image_{k}.png")
    print(f"Generated {NUM_SAMPLES} images.")
