import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OldPhotoDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir

        # Load all jpg images
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        #  Resize to 128x128 for better detail learning
        self.clean_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),   # Height x Width
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    #IMPROVED DAMAGE FUNCTION (REALISTIC OLD PHOTO EFFECTS)
    def add_synthetic_damage(self, image):
        img = np.array(image)

        # 1. Film grain noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

        # 2. Blur (simulating old camera focus issues)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 3. Brightness fading (old photos fade over time)
        alpha = np.random.uniform(0.6, 0.9)  # darker
        img = cv2.convertScaleAbs(img, alpha=alpha)

        # 4. Contrast variation
        beta = np.random.randint(-30, 30)
        img = cv2.convertScaleAbs(img, beta=beta)

        # 5. Color fading (convert to grayscale and back to RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 6. Random scratches
        for _ in range(np.random.randint(3, 7)):
            x1, y1 = np.random.randint(0, 128), np.random.randint(0, 128)
            x2, y2 = np.random.randint(0, 128), np.random.randint(0, 128)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        return img

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        # Load clean image
        clean_img = cv2.imread(img_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

        # Create damaged version
        damaged_img = self.add_synthetic_damage(clean_img)

        # Convert to tensors
        clean_tensor = self.clean_transform(clean_img)
        damaged_tensor = self.clean_transform(damaged_img)

        return damaged_tensor, clean_tensor