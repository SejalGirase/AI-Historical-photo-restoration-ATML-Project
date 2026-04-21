# AI-Historical-photo-restoration-ATML-Project

# AI Photo Restoration Models

This project uses two separate AI models to restore damaged photos. Here is how to access and set them up:

### 1. The Main U-Net Model (`generator_epoch_25.pth`)
This is our custom-trained AI built to fix scratches, tears, and fading. 
* **Status:** Already included! You do not need to download this separately. 
### 2. The GFPGAN Model (HD Facial Restoration)
This pre-trained model focuses strictly on restoring facial geometry in Ultra-HD. Because these files are too massive for GitHub's standard storage limits, they are hosted separately.
* **How to install it:**
  1. Click on the **Releases** tab on the right side of this GitHub page.
  2. Download the `.zip` file containing the GFPGAN weights.
  3. Extract the `.pth` files and place them exactly inside the `gfpgan/weights/` folder on your computer.
