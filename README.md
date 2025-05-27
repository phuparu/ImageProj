# ImageProj

โปรเจคเป็นระบบประมวลผลภาพสำหรับจำแนกโรคใบมะเขือเทศด้วย Deep Learning (CNN) พร้อมกระบวนการเตรียมข้อมูลและตัวอย่างการทำ Data Augmentation

## โครงสร้างโปรเจกต์

ImageProj/
│ ├── data/
│ │ ├── pre_cnn.py # สคริปต์สำหรับ preprocess ข้อมูลภาพ
│ │ ├── pre_process.py # คลาสและฟังก์ชันสำหรับ preprocessing/augmentation
│ │ ├── data.ipynb # สคริปต์สำหรับดาวน์โหลด Dataset
│ ├── test/
│ │ ├── preprocess_cnn.py # ตัวอย่างการทำ Data Augmentation และแสดงผล
│ │ ├── test.py # ตัวอย่างการประมวลผลภาพด้วย OpenCV
│ │ ├── cnn.py # ประเมินผลโมเดล
│ │ ├── resnet50.py # ประเมินผลโมเดล
│ ├── preprocessed_cnn/ # โฟลเดอร์เก็บไฟล์ .npy หลัง preprocess
| ├── train/ # โฟลเดอร์ train โมเดล
│ │ ├── cnn_train.py
│ │ ├── resnet50_train.py
│ │ ├── resnet50fine_train.py
│ ├── README.md
└── .gitignore

## ขั้นตอนการทำงาน

### 1. Preprocessing ข้อมูลภาพ

- แปลงภาพเป็น RGB
- ปรับขนาดภาพเป็น 224x224 พิกเซล
- Normalize ค่าพิกเซลให้อยู่ในช่วง 0-1
- บันทึกข้อมูลและ label เป็นไฟล์ `.npy`

### 2. Data Augmentation (ตัวอย่างใน `test/preprocess_cnn.py`)

- การหมุนภาพ (Rotation)
- การกลับด้าน (Flip)
- การปรับความสว่าง/ความคมชัด (Brightness/Contrast)
- การเพิ่ม noise หรือการเบลอ (Gaussian Noise/Blur)

### 3. การนำข้อมูลไปใช้กับโมเดล CNN

- โหลดไฟล์ `.npy` ที่ได้จากการ preprocess
- ใช้เป็น input สำหรับเทรนและประเมินโมเดล

## วิธีใช้งาน

1. รันไฟล์ data.ipynb ในโฟลเดอร์ data เพื่อดาวน์โหลด Dataset
2. เตรียมชุดข้อมูลภาพแยกตามโฟลเดอร์คลาสใน `data/tomatoleaf/tomato/train` และ `val`
3. รัน `data/pre_cnn.py` เพื่อ preprocess ข้อมูลและบันทึกไฟล์ `.npy`
4. (ไม่บังคับ) รัน `test/preprocess_cnn.py` เพื่อดูตัวอย่างการทำ Data Augmentation
5. นำไฟล์ `.npy` ไปใช้กับโมเดล CNN ของคุณ
6. หากมีไฟล์ tomato_leaf_cnn_model.keras อยู่แล้วสามารถนำโมเดลไปประยุกต์ใช้ได้

## ข้อควรระวัง

- ตรวจสอบ path ของไฟล์ภาพให้ถูกต้อง
- ควรใช้ Python 3.8+ และไลบรารีที่รองรับกับ NumPy เวอร์ชันที่ใช้งาน

## ไลบรารีที่ใช้

- numpy
- pillow (PIL)
- opencv-python
- albumentations
- matplotlib
- tqdm
- scikit-learn

---

**ผู้พัฒนา:**  
นายภูมิภัทร สันติถาวรยิ่ง
