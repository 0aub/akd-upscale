import os
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm

def resize_images(input_dir, output_dir, size):
    """
    Resizes all images in the input directory to the specified size
    and saves them in the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save resized images.
        size (int): The target size for resizing (e.g., 512 for 512x512).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the resize transform
    resize_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()  # Convert to tensor for any additional processing
    ])

    # Get the list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Initialize tqdm progress bar
    with tqdm(total=len(image_files), desc=f"Resizing Images to {size}x{size}", unit="file", ncols=80, colour="green") as pbar:
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            try:
                # Open the image
                with Image.open(input_path).convert("RGB") as img:
                    # Apply the resize transformation
                    resized_img = resize_transform(img)
                    resized_img = transforms.ToPILImage()(resized_img)  # Convert back to PIL

                    # Save the resized image
                    output_path = os.path.join(output_dir, filename)
                    resized_img.save(output_path)
                    pbar.set_postfix({"Last File": filename})
                pbar.update(1)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images to a specified size and save them.")
    parser.add_argument("--input_dir", type=str, default="./data/all_images", help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, default="data/all_lr_images", help="Path to the output directory for resized images.")
    parser.add_argument("--size", type=int, default=256, help="Target size for resizing (default: 512).")
    args = parser.parse_args()

    resize_images(args.input_dir, args.output_dir, args.size)
