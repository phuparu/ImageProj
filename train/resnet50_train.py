import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ✅ โหลด ResNet50 
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# ✅ freeze layers ของ ResNet (ไม่ให้ปรับ weight ช่วงแรก)
for layer in base_model.layers:
    layer.trainable = False
    

# โหลดข้อมูลจาก preprocessed_cnn
X_train = np.load('./preprocessed_cnn/X_train.npy')
y_train = np.load('./preprocessed_cnn/y_train.npy')
X_val = np.load('./preprocessed_cnn/X_val.npy')
y_val = np.load('./preprocessed_cnn/y_val.npy')

# ✅ แปลง label เป็น one-hot
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_val_enc = to_categorical(le.transform(y_val))

# ✅ ต่อหัวใหม่ (Top Model)
x = base_model.output
x = GlobalAveragePooling2D()(x)       # หรือใช้ Flatten() ก็ได้
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(le.classes_), activation='softmax')(x)

# ✅ สร้าง model ใหม่
model = Model(inputs=base_model.input, outputs=predictions)

# ✅ Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ เทรน model
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=30,
    batch_size=32
)

# ✅ แสดง accuracy และ loss
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save("resnet_tomato_model.h5")
print("✅ บันทึกโมเดลสำเร็จ: resnet_tomato_model.h5")