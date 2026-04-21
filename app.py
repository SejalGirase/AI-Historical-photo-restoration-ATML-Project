import os
import cv2
import time
import csv
import sys
import subprocess
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from models import UNetGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['XAI_FOLDER'] = 'static/xai'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['XAI_FOLDER'], exist_ok=True)

CSV_FILE = 'analytics_log.csv'
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Processing_Time_Sec", "SSIM", "PSNR", "User_Rating", "Architecture"])

device = torch.device("cpu")
model = UNetGenerator().to(device)
model.load_state_dict(torch.load("saved_gan_models_final/generator_epoch_25.pth", map_location=device, weights_only=True))
model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.bottleneck.register_forward_hook(get_activation('bottleneck'))

clean_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def process_image(filepath, filename, mode):
    start_time = time.time()
    
    img = cv2.imread(filepath)
    original_height, original_width = img.shape[:2]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    input_tensor_base = clean_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        base_output_tensor = model(input_tensor_base)

    feature_map = activation['bottleneck'][0, 0].cpu().numpy()
    feature_map_normalized = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(feature_map_normalized, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_img, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    overlay_img = cv2.addWeighted(img, 0.5, heatmap_resized, 0.5, 0)
    
    heatmap_filename = "xai_" + filename
    cv2.imwrite(os.path.join(app.config['XAI_FOLDER'], heatmap_filename), overlay_img)

    if mode == 'gfpgan':
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'gfpgan_out')
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [sys.executable, "gfpgan/inference_gfpgan.py", "-i", filepath, "-o", output_dir, "-v", "1.3", "-s", "2"]
        print(f"🚀 Executing GFPGAN: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ GFPGAN Log:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print("🚨 GFPGAN CRASHED. Here is the exact error:")
            print(e.stderr)
        
        base_name = os.path.splitext(filename)[0]
        possible_path_1 = os.path.join(output_dir, 'restored_imgs', filename)
        possible_path_2 = os.path.join(output_dir, 'restored_imgs', f"{base_name}.png")
        
        if os.path.exists(possible_path_1):
            final_output_bgr = cv2.imread(possible_path_1)
        elif os.path.exists(possible_path_2):
            final_output_bgr = cv2.imread(possible_path_2)
        else:
            print("🚨 Falling back to Phase 2 (Sliding Window).")
            mode = 'unet_sliding'

    if mode == 'unet_sliding':
        patch_size = 128
        pad_h = (patch_size - original_height % patch_size) % patch_size
        pad_w = (patch_size - original_width % patch_size) % patch_size
        img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        out_padded = np.zeros_like(img_padded, dtype=np.float32)
        new_h, new_w, _ = img_padded.shape

        for y in range(0, new_h, patch_size):
            for x in range(0, new_w, patch_size):
                patch = img_padded[y:y+patch_size, x:x+patch_size]
                input_tensor = clean_transform(patch).unsqueeze(0).to(device)
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                out_patch = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                out_patch = (out_patch * 0.5) + 0.5
                out_patch = np.clip(out_patch, 0, 1) * 255.0
                out_padded[y:y+patch_size, x:x+patch_size] = out_patch

        ai_restored_resized = out_padded[:original_height, :original_width].astype(np.uint8)
        final_output_bgr = cv2.cvtColor(ai_restored_resized, cv2.COLOR_RGB2BGR)

    elif mode == 'unet_standard':
        output_img = base_output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_img = (output_img * 0.5) + 0.5  
        output_img = np.clip(output_img, 0, 1)
        output_img_8bit = (output_img * 255).astype(np.uint8)
        ai_restored_resized = cv2.resize(output_img_8bit, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
        final_output_bgr = cv2.cvtColor(ai_restored_resized, cv2.COLOR_RGB2BGR)

    restored_filename = "restored_" + filename
    restored_filepath = os.path.join(app.config['UPLOAD_FOLDER'], restored_filename)
    cv2.imwrite(restored_filepath, final_output_bgr)
    
    orig_128 = cv2.resize(img_gray, (128, 128))
    restored_128 = cv2.cvtColor(final_output_bgr, cv2.COLOR_BGR2GRAY)
    restored_128 = cv2.resize(restored_128, (128, 128))
    
    ssim_score = round(ssim(orig_128, restored_128, data_range=255), 4)
    psnr_score = round(psnr(orig_128, restored_128, data_range=255), 2)
    time_taken = round(time.time() - start_time, 2)

    return restored_filename, heatmap_filename, time_taken, f"{original_width}x{original_height}", ssim_score, psnr_score, mode

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        mode = request.form.get('model_mode', 'unet_standard')

        if file.filename == '':
            return "No file selected", 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            restored_img, xai_img, time_taken, dims, ssim_val, psnr_val, actual_mode = process_image(filepath, filename, mode)
            
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, time_taken, ssim_val, psnr_val, "Pending", actual_mode])

            return render_template('index.html', 
                                   original_img=filename, 
                                   restored_img=restored_img,
                                   xai_img=xai_img,
                                   time_taken=time_taken,
                                   dims=dims,
                                   ssim=ssim_val,
                                   psnr=psnr_val,
                                   filename=filename,
                                   selected_mode=actual_mode)

    return render_template('index.html', original_img=None, selected_mode='unet_standard')

@app.route('/rate', methods=['POST'])
def rate_restoration():
    data = request.json
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([data['filename'], "N/A", "N/A", "N/A", data['rating'], "N/A"])
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)