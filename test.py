import matplotlib.pyplot as plt
from dataset_builder import OldPhotoDataset

def test_pipeline():
    dataset = OldPhotoDataset(image_dir="Data/images")
    
    if len(dataset) == 0:
        print("No images found in Data/images")
        return
        
    damaged_tensor, clean_tensor = dataset[0]
    
    def to_img(tensor):
        img = tensor.numpy().transpose(1, 2, 0)
        img = (img * 0.5) + 0.5
        return img
        
    damaged_img = to_img(damaged_tensor)
    clean_img = to_img(clean_tensor)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(clean_img)
    axes[0].set_title("Original")
    axes[1].imshow(damaged_img)
    axes[1].set_title("Damaged")
    plt.show()

if __name__ == "__main__":
    test_pipeline()