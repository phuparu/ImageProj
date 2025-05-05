import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ✅ โหลดข้อมูล val
X_val = np.load('./preprocessed_cnn/X_val.npy')
y_val = np.load('./preprocessed_cnn/y_val.npy')

# ✅ โหลด ResNet โมเดล
model = load_model('resnet_tomato_model.h5')

# ✅ LabelEncoder จาก y_val (หรือจะ fit จาก y_train ก็ได้ถ้าตรงกว่า)
le = LabelEncoder()
y_val_labels = le.fit_transform(y_val)
y_val_enc = to_categorical(y_val_labels)

# ✅ Evaluate
loss, acc = model.evaluate(X_val, y_val_enc, verbose=1)
print(f"\n✅ ResNet Validation Accuracy: {acc*100:.2f}%\n")

# ✅ Predict
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)

# ✅ Classification Report
print("📋 Classification Report (ResNet):")
report = classification_report(y_val_labels, y_pred_labels, target_names=le.classes_, digits=2)
print(report)

with open("resnet_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✅ บันทึกไว้ที่ resnet_classification_report.txt แล้ว")

# ✅ Confusion Matrix
cm = confusion_matrix(y_val_labels, y_pred_labels)
plt.figure(figsize=(14, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("ResNet Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ✅ พยากรณ์ภาพใหม่ด้วย ResNet
def predict_new_image_resnet(image_path, model, label_encoder):
    try:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_name = label_encoder.classes_[pred_class]
        confidence = np.max(pred)

        print(f"✅ Predicted Class: {pred_name} (Confidence: {confidence*100:.2f}%)")
        plt.imshow(image)
        plt.title(f"Predicted: {pred_name} ({confidence*100:.1f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")

# ✅ ตัวอย่างพยากรณ์ภาพใหม่ (ResNet)
image_path = "C:/Users/User/Desktop/ImageProcessing/Project/ImageProj/data/tomatoleaf/tomato/val/Tomato___Tomato_mosaic_virus/0a7cc59f-b2b0-4201-9c4a-d91eca5c03a3___PSU_CG 2230.JPG"
predict_new_image_resnet(image_path, model, le)
