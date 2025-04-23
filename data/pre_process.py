import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm

class TomatoLeafPreprocessor:
    def __init__(self, input_size=(224, 224)):
        """
        เริ่มต้นการทำงานของ TomatoLeafPreprocessor
        
        Args:
            input_size: ขนาดภาพที่ต้องการ (width, height)
        """
        self.input_size = input_size
        
    def resize_image(self, image):
        """
        ปรับขนาดภาพให้เท่ากับ input_size
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            ภาพที่ถูกปรับขนาดแล้ว
        """
        return cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image):
        """
        Normalize ภาพให้อยู่ในช่วง 0-1
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            ภาพที่ถูก normalize แล้ว
        """
        return image.astype(np.float32) / 255.0
    
    def apply_clahe(self, image):
        """
        ใช้ CLAHE (Contrast Limited Adaptive Histogram Equalization) เพื่อเพิ่มความชัดเจนของรายละเอียด
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            ภาพที่ผ่านการปรับปรุงความคมชัดด้วย CLAHE
        """
        # แปลงเป็น LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # แยกช่องสัญญาณ L
        l, a, b = cv2.split(lab)
        
        # สร้าง CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # ใช้ CLAHE กับช่องสัญญาณ L
        cl = clahe.apply(l)
        
        # รวมช่องสัญญาณกลับ
        limg = cv2.merge((cl, a, b))
        
        # แปลงกลับเป็น BGR
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    def segment_leaf(self, image):
        """
        แยกส่วนใบออกจากพื้นหลัง
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            mask: mask ของใบไม้
            segmented: ภาพที่แยกส่วนใบแล้ว
        """
        # แปลงเป็น HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # กำหนดช่วงสีเขียวในสเปซ HSV
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # สร้าง mask สำหรับสีเขียว
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # ใช้ morphological operations เพื่อปรับปรุง mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # ใช้ mask กับภาพต้นฉบับ
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return mask, segmented
    
    def apply_canny_edge_detection(self, image):
        """
        ใช้ Canny edge detection เพื่อหาขอบในภาพ
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            edges: ภาพขอบที่ตรวจจับได้
        """
        # แปลงเป็นภาพเทา
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ลด noise ด้วย Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # ใช้ Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def apply_sobel_edge_detection(self, image):
        """
        ใช้ Sobel edge detection เพื่อหาขอบในภาพ
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            edges: ภาพขอบที่ตรวจจับได้
        """
        # แปลงเป็นภาพเทา
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ลด noise ด้วย Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # คำนวณ gradient ในแนวแกน x และ y
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # แปลงเป็นค่าสัมบูรณ์
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # รวม gradient
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        return edges
    
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
    
    def create_augmentation_pipeline(self):
        """
        สร้าง pipeline สำหรับ data augmentation
        
        Returns:
            transform: albumentations transform pipeline
        """
        transform = A.Compose([
            A.RandomCrop(width=self.input_size[0], height=self.input_size[1], p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GridDistortion(p=0.2),
        ])
        
        return transform
    
    def crop_to_content(self, image):
        """
        ตัดภาพให้เหลือเฉพาะส่วนที่มีเนื้อหา (ตัดขอบดำออก)
        
        Args:
            image: ภาพ input (BGR format)
            
        Returns:
            ภาพที่ถูกตัดแล้ว
        """
        # แปลงเป็นภาพเทา
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # ใช้ threshold เพื่อแยกวัตถุออกจากพื้นหลัง
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # หาขอบเขตของวัตถุ
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # หากมี contour พบ
            # หากรวม contours ทั้งหมด
            cnt = contours[0]
            for c in contours[1:]:
                cnt = np.concatenate((cnt, c))
                
            # หา bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # ตัดภาพตาม bounding box
            cropped = image[y:y+h, x:x+w]
            
            return cropped
        else:
            # หากไม่พบ contour (อาจเป็นภาพทั้งหมดสีดำ)
            return image
    
    def preprocess_image(self, image, apply_augmentation=False):
        """
        ทำ preprocessing กับภาพ ด้วยขั้นตอนทั้งหมด
        
        Args:
            image: ภาพ input (BGR format)
            apply_augmentation: ระบุว่าจะทำ data augmentation หรือไม่
            
        Returns:
            results: dictionary ที่มีภาพผลลัพธ์จากขั้นตอนต่างๆ
        """
        results = {}
        
        # เก็บภาพต้นฉบับ
        results['original'] = image.copy()
        
        # ทำ CLAHE
        clahe_result = self.apply_clahe(image)
        results['clahe'] = clahe_result
        
        # แยกส่วนใบไม้
        mask, segmented = self.segment_leaf(clahe_result)
        results['mask'] = mask
        results['segmented'] = segmented
        
        # ตัดภาพให้เหลือเฉพาะส่วนที่มีเนื้อหา
        cropped = self.crop_to_content(segmented)
        results['cropped'] = cropped
        
        # ปรับขนาดภาพ
        resized = self.resize_image(cropped)
        results['resized'] = resized
        
        # Normalize ภาพ
        normalized = self.normalize_image(resized)
        results['normalized'] = normalized
        
        # ใช้ Canny edge detection
        canny_edges = self.apply_canny_edge_detection(resized)
        results['canny_edges'] = canny_edges
        
        # ใช้ Sobel edge detection
        sobel_edges = self.apply_sobel_edge_detection(resized)
        results['sobel_edges'] = sobel_edges
        
        # ใช้ morphological operations
        morphology_result = self.apply_morphological_operations(resized)
        results['morphology'] = morphology_result
        
        # ทำ Data Augmentation ถ้าจำเป็น
        if apply_augmentation:
            transform = self.create_augmentation_pipeline()
            augmented = transform(image=resized)['image']
            results['augmented'] = augmented
        
        return results
    
    def process_dataset(self, input_dir, output_dir, apply_augmentation=False, augmentation_factor=1):
        """
        ทำ preprocessing กับชุดข้อมูลทั้งหมด
        
        Args:
            input_dir: โฟลเดอร์ที่มีภาพต้นฉบับ
            output_dir: โฟลเดอร์ที่จะเก็บภาพที่ผ่านการ preprocessing
            apply_augmentation: ระบุว่าจะทำ data augmentation หรือไม่
            augmentation_factor: จำนวนภาพที่จะสร้างเพิ่มจากภาพต้นฉบับแต่ละภาพ
        """
        # สร้างโฟลเดอร์ผลลัพธ์
        os.makedirs(output_dir, exist_ok=True)
        
        # สร้างโฟลเดอร์ย่อยสำหรับประเภทการ preprocessing ต่างๆ
        preprocess_types = ['original', 'clahe', 'segmented', 'resized', 
                           'normalized', 'canny_edges', 'sobel_edges', 'morphology']
        
        for preprocess_type in preprocess_types:
            os.makedirs(os.path.join(output_dir, preprocess_type), exist_ok=True)
            
        if apply_augmentation:
            os.makedirs(os.path.join(output_dir, 'augmented'), exist_ok=True)
        
        # วนลูปผ่านไฟล์ในโฟลเดอร์ต้นฉบับ
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file_name in tqdm(image_files, desc="Processing images"):
            # อ่านภาพ
            image_path = os.path.join(input_dir, file_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            # ทำ preprocessing
            results = self.preprocess_image(image, apply_augmentation=apply_augmentation)
            
            # บันทึกผลลัพธ์
            for preprocess_type, processed_image in results.items():
                if preprocess_type != 'augmented':
                    # บันทึกภาพที่ผ่านการ preprocess แต่ละประเภท
                    output_path = os.path.join(output_dir, preprocess_type, file_name)
                    
                    # แปลงจาก float กลับเป็น uint8 ถ้าจำเป็น
                    if preprocess_type == 'normalized':
                        # แปลงกลับเป็น 0-255
                        output_image = (processed_image * 255).astype(np.uint8)
                    else:
                        output_image = processed_image
                        
                    cv2.imwrite(output_path, output_image)
            
            # บันทึกภาพที่ผ่านการ augment
            if apply_augmentation and 'augmented' in results:
                # บันทึกภาพที่มีการ augment แล้ว
                for i in range(augmentation_factor):
                    # ทำ augmentation ใหม่ทุกครั้ง
                    transform = self.create_augmentation_pipeline()
                    augmented = transform(image=results['resized'])['image']
                    
                    # สร้างชื่อไฟล์ใหม่
                    base_name, ext = os.path.splitext(file_name)
                    augmented_file_name = f"{base_name}_aug_{i}{ext}"
                    
                    # บันทึกภาพ
                    output_path = os.path.join(output_dir, 'augmented', augmented_file_name)
                    cv2.imwrite(output_path, augmented)

# ฟังก์ชันสำหรับการทำ preprocessing ชุดข้อมูลทั้งหมด
def preprocess_tomato_dataset(data_dir, output_dir, input_size=(224, 224), apply_augmentation=True):
    """
    ทำ preprocessing กับชุดข้อมูลมะเขือเทศทั้งหมด
    
    Args:
        data_dir: โฟลเดอร์หลักของชุดข้อมูล (ที่มีโฟลเดอร์ย่อย train, val)
        output_dir: โฟลเดอร์ที่จะเก็บผลลัพธ์
        input_size: ขนาดภาพที่ต้องการ (width, height)
        apply_augmentation: ระบุว่าจะทำ data augmentation หรือไม่
    """
    # สร้าง TomatoLeafPreprocessor
    preprocessor = TomatoLeafPreprocessor(input_size=input_size)
    
    # ตรวจสอบการมีอยู่ของโฟลเดอร์ train และ val
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Could not find train or val directory in {data_dir}")
        return
    
    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
    train_output_dir = os.path.join(output_dir, 'train')
    val_output_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    # ดึงรายการโฟลเดอร์โรคมะเขือเทศทั้งหมด
    disease_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('Tomato_')]
    
    # ทำ preprocessing กับแต่ละโฟลเดอร์โรค
    for disease_folder in disease_folders:
        print(f"Processing {disease_folder}...")
        
        # สร้างโฟลเดอร์ผลลัพธ์สำหรับแต่ละโรค
        train_disease_output_dir = os.path.join(train_output_dir, disease_folder)
        val_disease_output_dir = os.path.join(val_output_dir, disease_folder)
        
        os.makedirs(train_disease_output_dir, exist_ok=True)
        os.makedirs(val_disease_output_dir, exist_ok=True)
        
        # ทำ preprocessing กับข้อมูล train
        train_disease_dir = os.path.join(train_dir, disease_folder)
        preprocessor.process_dataset(
            train_disease_dir, 
            train_disease_output_dir, 
            apply_augmentation=apply_augmentation,
            augmentation_factor=3 if disease_folder == 'Tomato__healthy' else 1  # เพิ่มข้อมูลใบที่ปกติมากขึ้น
        )
        
        # ทำ preprocessing กับข้อมูล validation
        val_disease_dir = os.path.join(val_dir, disease_folder)
        if os.path.exists(val_disease_dir):  # ตรวจสอบว่ามีโฟลเดอร์นี้ใน val หรือไม่
            preprocessor.process_dataset(
                val_disease_dir, 
                val_disease_output_dir, 
                apply_augmentation=False  # ไม่ทำ augmentation กับข้อมูล validation
            )

# ฟังก์ชันสำหรับแสดงตัวอย่างผลลัพธ์ของการ preprocessing
def visualize_preprocessing_results(image_path, output_dir=None, input_size=(224, 224)):
    """
    แสดงตัวอย่างผลลัพธ์ของการ preprocessing
    
    Args:
        image_path: path ของภาพที่ต้องการแสดงผลลัพธ์
        output_dir: โฟลเดอร์ที่จะเก็บผลลัพธ์ (optional)
        input_size: ขนาดภาพที่ต้องการ (width, height)
    """
    # อ่านภาพ
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # สร้าง TomatoLeafPreprocessor
    preprocessor = TomatoLeafPreprocessor(input_size=input_size)
    
    # ทำ preprocessing
    results = preprocessor.preprocess_image(image, apply_augmentation=True)
    
    # แสดงผลลัพธ์
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # กำหนดภาพที่จะแสดง
    display_images = [
        ('Original', results['original'], 'BGR'),
        ('CLAHE Enhanced', results['clahe'], 'BGR'),
        ('Segmented Leaf', results['segmented'], 'BGR'),
        ('Cropped', results['cropped'], 'BGR'),
        ('Resized', results['resized'], 'BGR'),
        ('Normalized', results['normalized'], 'BGR'),
        ('Canny Edges', results['canny_edges'], 'GRAY'),
        ('Sobel Edges', results['sobel_edges'], 'GRAY'),
        ('Morphology', results['morphology'], 'GRAY')
    ]
    
    # แสดงภาพแต่ละชนิด
    for i, (title, img, color_type) in enumerate(display_images):
        row, col = divmod(i, 3)
        
        if color_type == 'BGR':
            # แปลงจาก BGR เป็น RGB สำหรับการแสดงผลใน matplotlib
            img_display = cv2.cvtColor(img if img.dtype == np.uint8 else (img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            img_display = img
            
        axes[row, col].imshow(img_display, cmap='gray' if color_type == 'GRAY' else None)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # บันทึกภาพถ้ามีการระบุ output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'preprocessing_visualization.png'))
    
    plt.show()
    
    # แสดงตัวอย่าง augmentation
    if 'augmented' in results:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # แสดงภาพต้นฉบับ
        original_rgb = cv2.cvtColor(results['resized'], cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # แสดงตัวอย่าง augmentation 5 ภาพ
        for i in range(5):
            row, col = divmod(i + 1, 3)
            
            # ทำ augmentation ใหม่
            transform = preprocessor.create_augmentation_pipeline()
            augmented = transform(image=results['resized'])['image']
            
            # แปลงจาก BGR เป็น RGB
            augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            
            # แสดงภาพ
            axes[row, col].imshow(augmented_rgb)
            axes[row, col].set_title(f'Augmented {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # บันทึกภาพถ้ามีการระบุ output_dir
        if output_dir:
            fig.savefig(os.path.join(output_dir, 'augmentation_visualization.png'))
            
        plt.show()

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # กำหนด path
    data_dir = "./data/tomatoleaf/tomato"  # โฟลเดอร์ของชุดข้อมูล
    output_dir = "./preprocessed_data"  # โฟลเดอร์ที่จะเก็บผลลัพธ์
    
    # ทำ preprocessing กับชุดข้อมูลทั้งหมด
    preprocess_tomato_dataset(data_dir, output_dir, apply_augmentation=True)
    
    # แสดงตัวอย่างผลลัพธ์ (ต้องระบุ path ของภาพที่ต้องการแสดงผล)
    sample_image_path = "./data/tomatoleaf/train/Tomato__Leaf_Mold/7f3398ef-a359-4ec4-b295-998c2851dea3___Crnl_L.Mold 6585.JPG"
    visualize_preprocessing_results(sample_image_path, output_dir=output_dir)