import os
from PIL import Image

# Path to your dataset
dataset_path = "./data/dataset/cats_dogs/train"

def check_images(folder_path):
    corrupted_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the file can be opened
            except (IOError, OSError):
                print(f"Corrupted file found: {file_path}")
                corrupted_files.append(file_path)
    return corrupted_files

# Check train and test folders
corrupted_train = check_images("./data/dataset/cats_dogs/train")
corrupted_test = check_images("./data/dataset/cats_dogs/test")

# Remove corrupted files (optional)
for file_path in corrupted_train + corrupted_test:
    os.remove(file_path)
    print(f"Removed corrupted file: {file_path}")
