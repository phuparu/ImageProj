import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# ฟังก์ชันสำหรับแสดงภาพ
def visualize_augmentations(original_image, augmented_images, titles):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(augmented_images) + 1, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    
    for i, (aug_img, title) in enumerate(zip(augmented_images, titles), start=2):
        plt.subplot(1, len(augmented_images) + 1, i)
        plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# โหลดภาพตัวอย่าง
image_path = "data/tomatoleaf/tomato/train/Tomato___Late_blight/975cefdc-29dc-4355-b9bb-5abd4a910654___GHLB2 Leaf 8657.JPG"
original_image = cv2.imread(image_path)

if original_image is None:
    print("ไม่สามารถโหลดภาพได้ กรุณาตรวจสอบ path")
else:
    # สร้าง pipeline สำหรับ augmentation
    augmentations = [
        A.Rotate(limit=45, p=1.0),  # หมุนภาพ
        A.HorizontalFlip(p=1.0),   # กลับด้านแนวนอน
        A.VerticalFlip(p=1.0),     # กลับด้านแนวตั้ง
        A.RandomBrightnessContrast(p=1.0),  # ปรับความสว่าง/ความคมชัด
        A.Blur(blur_limit=(3, 7), p=1.0),  # เบลอภาพ (ใช้ Blur แทน GaussianBlur)
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)  # เพิ่ม noise
    ]

    augmented_images = []
    titles = []

    # ใช้ augmentation แต่ละตัวกับภาพ
    for aug in augmentations:
        augmented = aug(image=original_image)
        augmented_images.append(augmented["image"])
        titles.append(aug.__class__.__name__)

    # แสดงผลลัพธ์
    visualize_augmentations(original_image, augmented_images, titles)