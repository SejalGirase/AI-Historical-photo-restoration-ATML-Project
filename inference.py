import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import UNetGenerator
from dataset_builder import OldPhotoDataset

def run_inference(image_path, model_path):
    device = torch.device("cpu")
    
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = OldPhotoDataset(image_dir="Data/images")
    
    clean_img = cv2.imread(image_path)
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    damaged_img = dataset.add_synthetic_damage(clean_img)
    
    input_tensor = dataset.clean_transform(damaged_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_img = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output_img = (output_img * 0.5) + 0.5
    output_img = np.clip(output_img, 0, 1)

    input_display = (input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5) + 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_img_resized := cv2.resize(clean_img, (64, 128)))
    axes[0].set_title("Ground Truth")
    axes[1].imshow(input_display)
    axes[1].set_title("Damaged Input")
    axes[2].imshow(output_img)
    axes[2].set_title("AI Restored")
    plt.show()

if __name__ == "__main__":
    # Change 'saved_models' to 'saved_gan_models' 
    # and use the new filename 'generator_epoch_15.pth'
    run_inference("Data/images/000010.jpg", "saved_gan_models_final/generator_epoch_25.pth")