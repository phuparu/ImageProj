from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ฟังก์ชันสำหรับโหลดและ normalize รูปภาพจากโฟลเดอร์
""" def load_and_preprocess_images(root_dir):
    X, y = [], []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path): continue

        print(f"📂 Loading class: {class_name}")
        file_count = 0

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, file_name)

                try:
                    image = Image.open(img_path).convert("RGB")       # แปลงให้เป็น RGB
                    image = image.resize((224, 224))                  # resize ด้วย PIL
                    img_array = np.array(image).astype(np.float32) / 255.0  # normalize
                    X.append(img_array)
                    y.append(class_name)
                    file_count += 1
                except Exception as e:
                    print(f"❌ อ่านไฟล์ไม่ได้: {img_path} | {e}")

        print(f"✅ พบรูป {file_count} ไฟล์ใน {class_name}")

    if not X or not y:
        print(f"❌ No data found in {root_dir}. Please check the directory structure.")
    return np.array(X), np.array(y) """


# โหลดข้อมูลจาก preprocessed_cnn
X_train = np.load('./preprocessed_cnn/X_train.npy')
y_train = np.load('./preprocessed_cnn/y_train.npy')
X_val = np.load('./preprocessed_cnn/X_val.npy')
y_val = np.load('./preprocessed_cnn/y_val.npy')

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)


print(f"✅ Train: {X_train.shape}, {y_train.shape}")
print(f"✅ Val:   {X_val.shape}, {y_val.shape}")

# แปลง label เป็น one-hot
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_val_enc = to_categorical(le.transform(y_val))

# สร้างโมเดล CNN 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')  # output เท่ากับจำนวน class
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# เทรนโมเดล
print("🚀 เริ่มเทรนโมเดล...")
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=30,
    batch_size=32
)

# แสดงกราฟความแม่นยำ
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# บันทึกโมเดลไว้ใช้ต่อ
model.save("tomato_leaf_cnn_model.keras")
print("✅ บันทึกโมเดลสำเร็จ: tomato_leaf_cnn_model.keras")
