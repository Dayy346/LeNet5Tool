import os
import zipfile
import shutil
from sklearn.model_selection import train_test_split

# Function to process and split the data
def process_zip_file(zip_file_path, output_zip_file):
    # Step 1: Extract the zip file
    temp_dir = "temp_extracted"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Locate the root directory containing the class folders
    root_dir = temp_dir
    while len(os.listdir(root_dir)) == 1 and os.path.isdir(os.path.join(root_dir, os.listdir(root_dir)[0])):
        root_dir = os.path.join(root_dir, os.listdir(root_dir)[0])

    # Step 2: Create train and test folders
    train_dir = os.path.join(temp_dir, "train")
    test_dir = os.path.join(temp_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Step 3: Split each class folder into train and test
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):  # Ensure it's a directory
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if len(images) == 0:
                print(f"Skipping empty folder: {class_folder}")
                continue

            print(f"Processing class: {class_folder}, Total images: {len(images)}")

            # Split into train and test
            train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

            # Create class-specific train and test directories
            class_train_dir = os.path.join(train_dir, class_folder)
            class_test_dir = os.path.join(test_dir, class_folder)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            # Move images to train folder
            for img in train_images:
                shutil.move(img, os.path.join(class_train_dir, os.path.basename(img)))

            # Move images to test folder
            for img in test_images:
                shutil.move(img, os.path.join(class_test_dir, os.path.basename(img)))

    # Step 4: Zip the processed dataset
    with zipfile.ZipFile(output_zip_file, 'w') as zip_ref:
        for folder, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(folder, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zip_ref.write(file_path, arcname)

    # Cleanup temporary directory
    shutil.rmtree(temp_dir)
    print(f"Processed zip file saved to {output_zip_file}")

# Usage
if __name__ == "__main__":
    original_zip_path = "C:/Users/dayya/Downloads/archive.zip"  # Replace with the path to your zip file
    output_zip_path = "flowersdata.zip"  # Replace with desired output zip file name

    process_zip_file(original_zip_path, output_zip_path)
