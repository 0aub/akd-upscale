import os
import argparse
import subprocess
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def resize_images(input_dir, output_dir, size):
    """
    Resizes all images in the input directory (including subdirectories) to the specified size
    and saves them in the output directory, preserving the directory structure.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # Count total image files for progress tracking
    total_files = sum(
        1 for root, _, files in os.walk(input_dir)
        for file in files if file.lower().endswith(valid_extensions)
    )

    resize_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()  # Convert to tensor for processing if needed
    ])

    with tqdm(total=total_files, desc=f"Resizing Images to {size}x{size}", unit="file", ncols=80, colour="green") as pbar:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not filename.lower().endswith(valid_extensions):
                    continue

                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    with Image.open(input_path).convert("RGB") as img:
                        resized_img = resize_transform(img)
                        resized_img = transforms.ToPILImage()(resized_img)
                        resized_img.save(output_path)
                        pbar.set_postfix({"Last File": relative_path})
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
                finally:
                    pbar.update(1)

def resize_video(input_video_path, output_video_path, size):
    """
    Resizes a single video to the specified (size x size), preserving audio,
    by invoking FFmpeg directly via subprocess.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to the output resized video file.
        size (int): Target dimension for both width and height (square).
    """
    # Construct FFmpeg scale filter. This forces the video to be square (size x size).
    # If you want to preserve aspect ratio, you can do more advanced logic, e.g.:
    #   scale='-1:{}'.format(size) or scale='{}:-1'.format(size)
    scale_filter = f"scale={size}:{size}"

    # Build the FFmpeg command. Here:
    #  -i            : Input file
    #  -vf           : Video filter for resizing
    #  -c:v libx264  : Example video codec for output (H.264)
    #  -c:a copy     : Copy the audio track without re-encoding
    # Adjust or add flags as needed (e.g., -crf, -preset, etc.)
    command = [
        "ffmpeg",
        "-i", input_video_path,
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-c:a", "copy",
        "-y",  # Overwrite output if exists
        output_video_path
    ]

    # Call FFmpeg via subprocess
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Video successfully resized to {size}x{size} and saved at: {output_video_path}")
    except subprocess.CalledProcessError as e:
        # Print FFmpeg's error output for debugging
        print(f"Error resizing video:\n{e.stderr.decode('utf-8', errors='replace')}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images or a single video to a specified size and save them, preserving structure (for images) and audio (for video).")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input directory (images) or single video file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory (for images) or single video file (for video).")
    parser.add_argument("--size", type=int, default=512, help="Target size for resizing (default: 512).")
    args = parser.parse_args()

    # Check if input_path is a directory or a file
    if os.path.isdir(args.input_path):
        # We assume it's a directory of images -> proceed with image resizing
        resize_images(args.input_path, args.output_path, args.size)
    else:
        # We assume it's a video -> handle video resizing (with FFmpeg)
        valid_video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv')
        if args.input_path.lower().endswith(valid_video_extensions):
            # If user didn't specify a valid file extension for output, let's default to .mp4
            if not any(args.output_path.lower().endswith(ext) for ext in valid_video_extensions):
                args.output_path += ".mp4"
            resize_video(args.input_path, args.output_path, args.size)
        else:
            raise ValueError(f"Input file does not appear to be a valid video: {args.input_path}")
