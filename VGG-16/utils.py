import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def calc_mean_RGB(train_data_path):
    total_r, total_g, total_b, pixel_count = 0, 0, 0, 0

    for class_folder in os.listdir(train_data_path):
        class_path = os.path.join(train_data_path, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_path, filename)
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                        if len(img_array.shape) == 3:  # Ensure it's a color image
                            total_r += np.sum(img_array[:,:,0])
                            total_g += np.sum(img_array[:,:,1])
                            total_b += np.sum(img_array[:,:,2])
                            pixel_count += img_array.shape[0] * img_array.shape[1]

    mean_r = total_r / pixel_count
    mean_g = total_g / pixel_count
    mean_b = total_b / pixel_count

    return mean_r, mean_g, mean_b

class CustomTransform:
    def __init__(self, mean_r, mean_g, mean_b):
        self.mean_r = mean_r
        self.mean_g = mean_g
        self.mean_b = mean_b

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img[:,:,0] -= self.mean_r
        img[:,:,1] -= self.mean_g
        img[:,:,2] -= self.mean_b
        return transforms.ToTensor()(img)