import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def apply_morphological_operations(self, image):
        """
        ใช้ morphological operations เพื่อปรับปรุงภาพ
        
        Args:
            image: ภาพ input (BGR format หรือ grayscale)
            
        Returns:
            ภาพที่ผ่านการทำ morphological operations
        """
        if len(image.shape) == 3:
            # แปลงเป็นภาพเทาถ้าเป็นภาพสี
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # สร้าง kernel
        kernel = np.ones((5, 5), np.uint8)
        
        # ใช้ opening (erosion แล้วตามด้วย dilation)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # ใช้ closing (dilation แล้วตามด้วย erosion)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        return closing

# ฟังก์ชันหลักสำหรับโหลดรูปและแสดงผล
def main():
    # โหลดรูปภาพ
    image_path = "data/tomatoleaf/tomato/train/Tomato___Late_blight/975cefdc-29dc-4355-b9bb-5abd4a910654___GHLB2 Leaf 8657.JPG"
    image = cv2.imread(image_path)
    
    if image is None:
        print("ไม่สามารถโหลดรูปภาพได้ กรุณาตรวจสอบ path")
        return
    
    # สร้างออบเจ็กต์ ImageProcessor
    processor = ImageProcessor()
    
    # ใช้ฟังก์ชัน apply_morphological_operations
    result = processor.apply_morphological_operations(image)
    
    # แสดงผลลัพธ์ด้วย matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(result, cmap="gray")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()