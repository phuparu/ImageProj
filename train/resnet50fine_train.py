import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ โหลดข้อมูล
X_train = np.load('./preprocessed_cnn/X_train.npy')
y_train = np.load('./preprocessed_cnn/y_train.npy')
X_val = np.load('./preprocessed_cnn/X_val.npy')
y_val = np.load('./preprocessed_cnn/y_val.npy')

# ✅ Label encoding และ One-hot
le = LabelEncoder()
y_train_labels = le.fit_transform(y_train)
y_val_labels = le.transform(y_val)

y_train_enc = to_categorical(y_train_labels)
y_val_enc = to_categorical(y_val_labels)

# ✅ Class Weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights_dict = dict(enumerate(class_weights))

# ✅ Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# ✅ ResNet + Fine-tune
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ เทรน
history = model.fit(
    datagen.flow(X_train, y_train_enc, batch_size=16),
    validation_data=(X_val, y_val_enc),
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# ✅ บันทึกโมเดล
model.save("resnet50_finetune_aug_weighted.h5")
print("✅ บันทึกโมเดลแล้ว")

# ✅ 🔧 Evaluate & Confusion Matrix
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)

# Accuracy
val_acc = np.mean(y_pred_labels == y_val_labels)
print(f"\n🎯 Final Validation Accuracy: {val_acc*100:.2f}%")

# Classification Report
report = classification_report(y_val_labels, y_pred_labels, target_names=le.classes_, digits=2)
print("\n📋 Classification Report:\n", report)

# Confusion Matrix
cm = confusion_matrix(y_val_labels, y_pred_labels)
plt.figure(figsize=(14, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("ResNet50 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ✅ 📈 กราฟ Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
