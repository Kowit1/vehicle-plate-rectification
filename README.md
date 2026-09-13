# PlateRect — Vehicle License Plate Rectification

เว็บ Streamlit สำหรับเปลี่ยนภาพรถหรือภาพนิ่งจาก CCTV ให้เป็นภาพป้ายทะเบียนที่มองตรงและพร้อมนำไปใช้กับ OCR

## ขั้นตอนการทำงาน

1. อัปโหลดภาพรถ 1 ภาพ หรือถ่ายภาพจากกล้อง
2. ตรวจหาบริเวณป้ายทะเบียนจากเส้นขอบและลักษณะสี่เหลี่ยม
3. ตัดภาพป้ายก่อนปรับ เพื่อใช้เปรียบเทียบ
4. เก็บจุดตามขอบป้ายหลายจุดและคำนวณ Homography ด้วย RANSAC
5. ใช้ Perspective Transform เพื่อปรับป้ายให้มองตรง
6. แปลง Grayscale เพิ่ม Contrast ด้วย CLAHE และลด Noise
7. แสดงภาพก่อน–หลัง พร้อมดาวน์โหลด PNG

หากจุดตามขอบไม่เพียงพอ ระบบจะ fallback ไปใช้มุมป้าย 4 จุด และแจ้งวิธีที่ใช้บนหน้าเว็บอย่างชัดเจน OCR ไม่ใช่ส่วนหลักของโปรเจกต์ แต่ภาพสุดท้ายถูกเตรียมให้พร้อมสำหรับต่อกับ OCR ภายนอก

## รันบนเครื่อง

ต้องใช้ Python 3.10 ขึ้นไป

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

จากนั้นเปิด `http://localhost:8501`

## ทดสอบ

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

ชุดทดสอบสร้างภาพรถและป้ายเอียงสังเคราะห์ขึ้นในหน่วยความจำ จึงไม่ต้องใช้ไฟล์ภาพจริงในการทดสอบ pipeline

## Deploy บน Streamlit Community Cloud

1. Push repository นี้ขึ้น GitHub
2. เข้า Streamlit Community Cloud และเลือก **Create app**
3. เลือก repository และ branch ที่ต้องการ
4. กำหนด Main file path เป็น `app.py`
5. กด Deploy — ระบบจะติดตั้งแพ็กเกจจาก `requirements.txt` อัตโนมัติ

## โครงสร้างหลัก

- `app.py` — หน้าเว็บ Streamlit
- `plate_rectification.py` — detection, RANSAC homography, perspective transform และ preprocessing
- `tests/test_plate_rectification.py` — unit/integration tests ของ pipeline
- `.streamlit/config.toml` — theme และขนาดไฟล์อัปโหลด
