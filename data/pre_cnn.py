import os
import numpy as np
from PIL import Image
from pre_process import TomatoLeafPreprocessor  # ไฟล์ pre_process.py

def load_and_preprocess_images(root_dir):
    X, y = [], []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"❌ Skipping non-directory: {class_name}")
            continue

        print(f"📂 Loading class: {class_name}")
        file_count = 0

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, file_name)

                try:
                    image = Image.open(img_path).convert("RGB")
                    image = image.resize((224, 224))
                    img_array = np.array(image).astype(np.float32) / 255.0
                    X.append(img_array)
                    y.append(class_name)
                    file_count += 1
                except Exception as e:
                    print(f"❌ Error reading file: {img_path} | {e}")

        print(f"✅ Loaded {file_count} files from class: {class_name}")

    if not X or not y:
        print(f"❌ No data found in {root_dir}. Please check the directory structure.")
    return np.array(X), np.array(y)

# สร้าง preprocessor
pre = TomatoLeafPreprocessor(input_size=(224, 224))

# โหลดชุด train
train_root = './data/tomatoleaf/tomato/train'
X_train, y_train = load_and_preprocess_images(train_root)

# โหลดชุด val
val_root = './data/tomatoleaf/tomato/val'
X_val, y_val = load_and_preprocess_images(val_root)

print(f"✅ Train: {X_train.shape}, {y_train.shape}")
print(f"✅ Val:   {X_val.shape}, {y_val.shape}")
os.makedirs('./preprocessed_cnn/', exist_ok=True)

# บันทึกข้อมูลหลังจาก preprocess
np.save('./preprocessed_cnn/X_train.npy', X_train)
np.save('./preprocessed_cnn/y_train.npy', y_train)
np.save('./preprocessed_cnn//X_val.npy', X_val)
np.save('./preprocessed_cnn/y_val.npy', y_val)

print("✅ Data saved to ./preprocessed_cnn/")
