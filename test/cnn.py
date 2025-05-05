import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ✅ โหลดข้อมูลจากไฟล์ .npy
X_val = np.load('./preprocessed_cnn/X_val.npy')
y_val = np.load('./preprocessed_cnn/y_val.npy')

# ✅ โหลดโมเดลจากไฟล์ .h5
model = load_model('tomato_leaf_cnn_model.keras')

# ✅ สร้าง LabelEncoder และแปลง label เป็น one-hot
le = LabelEncoder()
y_val_enc = to_categorical(le.fit_transform(y_val))
y_val_labels = le.transform(y_val)

# ✅ Evaluate บนชุด validation
loss, accuracy = model.evaluate(X_val, y_val_enc, verbose=1)
print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%\n")

# ✅ Predict
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)

# ✅ Classification Report
print("📋 Classification Report:")
report = classification_report(
    y_val_labels,           # true labels
    y_pred_labels,          # predicted labels
    target_names=le.classes_,  # ชื่อ class
    digits=2                # ทศนิยม 2 ตำแหน่ง
)
print(report)
with open("classification_report_1.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✅ บันทึกผลไว้ใน classification_report_1.txt แล้ว")

# ✅ Confusion Matrix
cm = confusion_matrix(y_val_labels, y_pred_labels)
plt.figure(figsize=(14, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ✅ ดูภาพที่ทำนายผิด
""" wrong_idxs = np.where(y_pred_labels != y_val_labels)[0]
print(f"❌ ทำนายผิดทั้งหมด {len(wrong_idxs)} ภาพ")

for i in wrong_idxs[:5]:
    plt.imshow(X_val[i])
    plt.title(f"Predict: {le.classes_[y_pred_labels[i]]} | True: {le.classes_[y_val_labels[i]]}")
    plt.axis('off')
    plt.show() """

# ✅ ฟังก์ชันสำหรับพยากรณ์ภาพใหม่
def predict_new_image(image_path, model, label_encoder):
    try:
        # โหลดและ preprocess ภาพ
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # พยากรณ์ผลลัพธ์
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = label_encoder.classes_[predicted_class]

        # แสดงผลลัพธ์
        print(f"✅ Predicted Class: {predicted_class_name}")
        plt.imshow(Image.open(image_path))
        plt.title(f"Predicted: {predicted_class_name}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")

# ตัวอย่างการพยากรณ์ภาพใหม่
image_path = "C:/Users/User/Desktop/ImageProcessing/Project/ImageProj/data/tomatoleaf/tomato/val/Tomato___Tomato_mosaic_virus/0a7cc59f-b2b0-4201-9c4a-d91eca5c03a3___PSU_CG 2230.JPG"
predict_new_image(image_path, model, le)
