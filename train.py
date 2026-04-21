import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from dataset_builder import OldPhotoDataset
from models import UNetGenerator, Discriminator


def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware set to: {device} (GAN Mode Active)")

    # ✅ Load VGG for perceptual loss
    vgg = vgg16(pretrained=True).features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # ✅ Dataset + Dataloader
    dataset = OldPhotoDataset(image_dir="Data/images")
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ✅ Models
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    # ✅ Loss functions
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    # ✅ Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 🔥 Train fewer epochs (enough for good results)
    epochs = 30

    save_dir = "saved_gan_models_final"
    os.makedirs(save_dir, exist_ok=True)

    print("Initiating Stable GAN Training...")

    # ================= TRAINING LOOP =================
    for epoch in range(epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        for batch_idx, (damaged_imgs, clean_imgs) in enumerate(dataloader):

            damaged_imgs = damaged_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            batch_size = clean_imgs.size(0)

            # ✅ LABEL SMOOTHING (IMPORTANT)
            real_labels = torch.full((batch_size, 1), 0.9).to(device)
            fake_labels = torch.full((batch_size, 1), 0.1).to(device)

            # ================= TRAIN DISCRIMINATOR =================
            optimizer_D.zero_grad()

            outputs_real = discriminator(clean_imgs)
            loss_d_real = criterion_bce(outputs_real, real_labels)

            fake_imgs = generator(damaged_imgs)

            outputs_fake = discriminator(fake_imgs.detach())
            loss_d_fake = criterion_bce(outputs_fake, fake_labels)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()

            # ✅ SLOW DOWN DISCRIMINATOR
            if batch_idx % 3 == 0:
                optimizer_D.step()

            # ================= TRAIN GENERATOR =================
            optimizer_G.zero_grad()

            outputs_fake_for_g = discriminator(fake_imgs)

            loss_g_adv = criterion_bce(outputs_fake_for_g, real_labels)
            loss_g_mse = criterion_mse(fake_imgs, clean_imgs)

            # ✅ OPTIMIZED PERCEPTUAL LOSS
            with torch.no_grad():
                real_features = vgg(clean_imgs)

            fake_features = vgg(fake_imgs)
            loss_p = F.l1_loss(fake_features, real_features)

            # ✅ FINAL STABLE LOSS
            loss_g = (loss_g_mse * 20) + (loss_g_adv * 0.1) + (loss_p * 0.1)

            loss_g.backward()
            optimizer_G.step()

            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()

            # ✅ REDUCED PRINTING (FASTER)
            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] "
                    f"| D_Loss: {loss_d.item():.4f} | G_Loss: {loss_g.item():.4f} "
                    f"| MSE: {loss_g_mse.item():.4f} | ADV: {loss_g_adv.item():.4f} | P: {loss_p.item():.4f}"
                )

        print(
            f"=== Epoch {epoch+1} Completed | Avg G_Loss: {(epoch_loss_g/len(dataloader)):.4f} "
            f"| Avg D_Loss: {(epoch_loss_d/len(dataloader)):.4f} ==="
        )

        # ✅ Save model
        torch.save(generator.state_dict(), f"{save_dir}/generator_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train_gan()