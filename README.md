# PlateRect — Vehicle License Plate Rectification

เว็บ Streamlit สำหรับเปลี่ยนภาพรถหรือภาพนิ่งจาก CCTV ให้เป็นภาพป้ายทะเบียนที่มองตรงและพร้อมนำไปใช้กับ OCR

**ทดลองใช้งาน:** [Streamlit Community Cloud](https://vehicle-plate-rectification-rupcdtmqkf9ncjqiqgptdw.streamlit.app/)

## ขั้นตอนการทำงาน

1. อัปโหลดภาพรถ 1 ภาพ หรือถ่ายภาพจากกล้อง
2. ตรวจหาป้ายจากเส้นขอบและพื้นที่สี จัดอันดับด้วยแถวตัวอักษรและหลักฐานขอบแผ่นป้าย
3. ตัดภาพป้ายก่อนปรับ เพื่อใช้เปรียบเทียบ
4. สร้างภาพ frontal reference เบื้องต้นจากมุมป้าย 4 จุด
5. ตรวจหา SIFT หรือ ORB keypoints และสร้าง descriptors ในป้ายต้นฉบับกับ reference
6. จับคู่ด้วย KNN แล้วคัด outlier ขั้นแรกด้วย Lowe's ratio test
7. คำนวณ Homography จากคู่จุดด้วย RANSAC และตรวจสอบ inlier ratio/ความบิดของผลลัพธ์
8. ใช้ Perspective Transform เพื่อปรับป้ายให้มองตรง
9. แปลง Grayscale เพิ่ม Contrast ด้วย CLAHE และลด Noise
10. แสดงภาพก่อน–หลัง ภาพ match/inlier และดาวน์โหลด PNG

หาก feature matches ไม่เพียงพอ ระบบจะ fallback ไปใช้ RANSAC จากจุดตามขอบ และถ้ายังไม่เสถียรจึงใช้มุมป้าย 4 จุด ทุกกรณีจะแจ้งวิธีที่ใช้จริงบนหน้าเว็บอย่างชัดเจน OCR ไม่ใช่ส่วนหลักของโปรเจกต์ แต่ภาพสุดท้ายถูกเตรียมให้พร้อมสำหรับต่อกับ OCR ภายนอก

ทั้ง feature RANSAC และ edge RANSAC ต้องผ่านการตรวจมุมหลังแปลง ทิศทางและพื้นที่ภาพก่อนใช้งาน จุดขอบที่ตรวจไม่พบจะไม่ถูกสร้างขึ้นมาแทน ระบบแจ้งเตือนเมื่อป้ายเล็กหรือใช้ fallback และคะแนนที่แสดงเป็น **คะแนนจัดอันดับ ไม่ใช่เปอร์เซ็นต์ความแม่นยำ**

ดู [ผลประเมินก่อน–หลังและข้อจำกัด](EVALUATION.md): ชุดภาพไทยที่กันไว้ 37 ภาพเลือกกรอบอันดับแรกผ่านเกณฑ์ IoU ≥ 0.50 เพิ่มจาก 12 เป็น 35 ภาพ ผลนี้วัดการหาตำแหน่งป้าย ไม่ใช่ความถูกต้องของมุมหรือ OCR และยังไม่ครอบคลุม CCTV ทุกสถานการณ์

## รันบนเครื่อง

ต้องใช้ Python 3.10 ขึ้นไป

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

จากนั้นเปิด `http://localhost:8501`

## ทดสอบ

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

ชุดทดสอบครอบคลุม SIFT, ORB, KNN ratio test, RANSAC, fallback และภาพรถ/ป้ายเอียงสังเคราะห์โดยไม่ต้องพึ่งไฟล์ภายนอก

รวมการตรวจ homography ที่สะท้อน/บิดรูป, มุมที่ไม่ถูกต้อง, ขอบที่ว่างเปล่า และการอัปโหลดผ่าน Streamlit AppTest

## Deploy บน Streamlit Community Cloud

แอปเวอร์ชันปัจจุบัน deploy จาก branch `main` และเปิดใช้งานได้จากลิงก์ด้านบน การ push commit ใหม่ขึ้น branch นี้จะทำให้ Streamlit deploy เวอร์ชันล่าสุดให้อัตโนมัติ

1. Push repository นี้ขึ้น GitHub
2. เข้า Streamlit Community Cloud และเลือก **Create app**
3. เลือก repository และ branch ที่ต้องการ
4. กำหนด Main file path เป็น `app.py`
5. กด Deploy — ระบบจะติดตั้งแพ็กเกจจาก `requirements.txt` อัตโนมัติ

## โครงสร้างหลัก

- `app.py` — หน้าเว็บ Streamlit
- `plate_rectification.py` — detection, SIFT/ORB, descriptor matching, RANSAC homography, perspective transform และ preprocessing
- `tests/test_plate_rectification.py` — unit/integration tests ของ pipeline
- `.streamlit/config.toml` — theme และขนาดไฟล์อัปโหลด
