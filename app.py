"""Streamlit interface for the vehicle license-plate rectification pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from plate_rectification import (
    crop_plate,
    decode_image,
    detect_plate_candidates,
    draw_detection,
    encode_png,
    order_points,
    preprocess_for_ocr,
    rectify_plate,
)


st.set_page_config(page_title="PlateRect", page_icon="🚘", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stFileUploaderDropzone"] {border: 1.5px dashed #6b7cff; border-radius: 16px; padding: 1.2rem;}
    [data-testid="stMetric"] {background: #f6f7fb; border: 1px solid #e8eaf2; border-radius: 14px; padding: .7rem 1rem;}
    .step {display:inline-block; color:#4255d4; background:#eef0ff; border-radius:999px; padding:.3rem .7rem; margin:0 .3rem .4rem 0; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PlateRect")
st.caption("ทำภาพป้ายทะเบียนที่เอียงจากภาพรถให้ตรงและพร้อมสำหรับ OCR")
st.markdown(
    '<span class="step">1 อัปโหลดภาพรถ</span>'
    '<span class="step">2 ตรวจหาป้าย</span>'
    '<span class="step">3 ปรับ Perspective</span>'
    '<span class="step">4 เตรียมภาพ OCR</span>',
    unsafe_allow_html=True,
)


def show_bgr(image: np.ndarray, caption: str | None = None) -> None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


source_mode = st.radio("แหล่งภาพ", ["อัปโหลดไฟล์", "ถ่ายจากกล้อง"], horizontal=True, label_visibility="collapsed")
if source_mode == "อัปโหลดไฟล์":
    uploaded = st.file_uploader(
        "ลากภาพรถมาวาง หรือกดเพื่อเลือกไฟล์",
        type=["jpg", "jpeg", "png", "webp"],
        help="แนะนำภาพที่เห็นป้ายชัดพอสมควร ขนาดไฟล์ไม่เกิน 20 MB",
    )
else:
    uploaded = st.camera_input("ถ่ายภาพรถให้เห็นป้ายทะเบียน")

if uploaded is None:
    st.info("เริ่มจากอัปโหลดภาพรถ 1 ภาพ ระบบจะตรวจหาและตัดป้ายให้อัตโนมัติ")
    st.stop()

try:
    vehicle_image = decode_image(uploaded.getvalue())
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.spinner("กำลังตรวจหาป้ายทะเบียน…"):
    candidates, diagnostics = detect_plate_candidates(vehicle_image)

if not candidates:
    st.error("ยังตรวจไม่พบกรอบที่น่าจะเป็นป้ายทะเบียนในภาพนี้")
    st.markdown("ลองใช้ภาพที่ป้ายใหญ่และชัดขึ้น ลดแสงสะท้อน หรือครอปรถให้ใกล้ขึ้นแล้วอัปโหลดใหม่")
    show_bgr(vehicle_image, "ภาพที่ได้รับ")
    with st.expander("ดูภาพขอบที่ระบบตรวจพบ"):
        st.image(diagnostics["edges"], use_container_width=True, clamp=True)
    st.stop()

selector_col, metric_col, size_col = st.columns([2.2, 1, 1])
with selector_col:
    selected_index = st.selectbox(
        "กรอบป้ายที่ตรวจพบ",
        range(len(candidates)),
        format_func=lambda index: f"ตัวเลือก {index + 1} — ความมั่นใจ {candidates[index].score * 100:.0f}%",
    )
candidate = candidates[selected_index]
with metric_col:
    st.metric("ความมั่นใจ", f"{candidate.score * 100:.0f}%")
with size_col:
    st.metric("ขนาดภาพ", f"{vehicle_image.shape[1]}×{vehicle_image.shape[0]}")

use_manual = False
manual_points = candidate.points.copy()
with st.expander("กรอบไม่ตรง? ปรับพิกัดมุมด้วยตนเอง"):
    st.caption("พิกัดเรียงเป็น บนซ้าย → บนขวา → ล่างขวา → ล่างซ้าย")
    point_columns = st.columns(4)
    point_names = ["บนซ้าย", "บนขวา", "ล่างขวา", "ล่างซ้าย"]
    for index, column in enumerate(point_columns):
        with column:
            st.markdown(f"**{point_names[index]}**")
            manual_points[index, 0] = st.number_input(
                "X", 0, vehicle_image.shape[1] - 1, int(candidate.points[index, 0]), key=f"x-{selected_index}-{index}"
            )
            manual_points[index, 1] = st.number_input(
                "Y", 0, vehicle_image.shape[0] - 1, int(candidate.points[index, 1]), key=f"y-{selected_index}-{index}"
            )
    use_manual = st.checkbox("ใช้พิกัดที่แก้ไข", key=f"manual-{selected_index}")

try:
    plate_points = order_points(manual_points if use_manual else candidate.points)
    before_crop = crop_plate(vehicle_image, plate_points)
    result = rectify_plate(vehicle_image, plate_points)
    ocr_ready = preprocess_for_ocr(result.image)
except (ValueError, cv2.error) as error:
    st.error(f"ปรับภาพไม่สำเร็จ: {error}")
    st.stop()

st.subheader("ผลลัพธ์")
overview, details = st.tabs(["ก่อน–หลัง", "ภาพรถและรายละเอียด"])
with overview:
    before_col, straight_col, ready_col = st.columns(3)
    with before_col:
        st.markdown("**1. ป้ายที่ตรวจพบ**")
        show_bgr(before_crop, "ก่อนปรับมุม")
    with straight_col:
        st.markdown("**2. ป้ายที่ปรับตรงแล้ว**")
        show_bgr(result.image, "Perspective Transform")
    with ready_col:
        st.markdown("**3. ภาพพร้อม OCR**")
        st.image(ocr_ready, caption="Grayscale + Contrast + Denoise", use_container_width=True, clamp=True)

    if result.method == "RANSAC edge homography":
        st.success(f"ปรับป้ายด้วย Homography + RANSAC สำเร็จ — จุดที่ผ่าน {result.inliers}/{result.correspondences}")
    else:
        st.warning("จุดขอบสำหรับ RANSAC ไม่เสถียร จึงใช้การแปลงจากมุมป้าย 4 จุดแทน")

    download_one, download_two = st.columns(2)
    with download_one:
        st.download_button("ดาวน์โหลดป้ายที่ปรับตรง", encode_png(result.image), "plate-rectified.png", "image/png", use_container_width=True)
    with download_two:
        st.download_button("ดาวน์โหลดภาพพร้อม OCR", encode_png(ocr_ready), "plate-ocr-ready.png", "image/png", use_container_width=True, type="primary")

with details:
    show_bgr(draw_detection(vehicle_image, plate_points), "กรอบที่นำไปประมวลผล")
    detail_one, detail_two, detail_three = st.columns(3)
    detail_one.metric("วิธีปรับภาพ", "RANSAC" if result.method.startswith("RANSAC") else "4 corners")
    detail_two.metric("จุด Inlier", f"{result.inliers}/{result.correspondences}")
    detail_three.metric("ขนาดป้ายผลลัพธ์", f"{result.image.shape[1]}×{result.image.shape[0]}")
    with st.expander("ข้อมูลทางเทคนิค"):
        st.markdown("**Homography matrix**")
        st.code(np.array2string(result.homography, precision=5, suppress_small=True))
        mask_one, mask_two = st.columns(2)
        mask_one.image(diagnostics["edges"], caption="Canny edges", use_container_width=True, clamp=True)
        mask_two.image(diagnostics["candidate_mask"], caption="Connected edge mask", use_container_width=True, clamp=True)

st.caption("OCR เป็นขั้นตอนเสริม ภาพสุดท้ายถูกเตรียมไว้เพื่อนำไปใช้กับ OCR ภายนอกได้ทันที")
