"""Streamlit interface for the vehicle license-plate rectification pipeline."""

from __future__ import annotations

import cv2
import hashlib
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
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=caption, width="stretch")


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

with st.expander("ตั้งค่าการจับคู่จุดตาม rubric"):
    st.caption("ค่าเริ่มต้นเหมาะกับภาพทั่วไป ปรับเมื่ออยากสาธิตผลของ detector และ RANSAC")
    setting_one, setting_two, setting_three = st.columns(3)
    with setting_one:
        feature_detector = st.selectbox(
            "Feature detector",
            ["SIFT", "ORB"],
            help="SIFT ทนต่อการเปลี่ยนสเกลและมุมมองได้ดี ส่วน ORB ทำงานเร็วกว่า",
        )
    with setting_two:
        ratio_threshold = st.slider(
            "Lowe's ratio threshold",
            min_value=0.55,
            max_value=0.90,
            value=0.75,
            step=0.05,
            help="ค่ายิ่งต่ำยิ่งคัดคู่ descriptor เข้มงวด",
        )
    with setting_three:
        ransac_threshold = st.slider(
            "RANSAC reprojection (px)",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
        )

with st.spinner("กำลังตรวจหาป้ายทะเบียน…"):
    candidates, diagnostics = detect_plate_candidates(vehicle_image)

if not candidates:
    st.error("ยังตรวจไม่พบกรอบที่น่าจะเป็นป้ายทะเบียนในภาพนี้")
    st.markdown("ลองใช้ภาพที่ป้ายใหญ่และชัดขึ้น ลดแสงสะท้อน หรือครอปรถให้ใกล้ขึ้นแล้วอัปโหลดใหม่")
    show_bgr(vehicle_image, "ภาพที่ได้รับ")
    with st.expander("ดูภาพขอบที่ระบบตรวจพบ"):
        st.image(diagnostics["edges"], width="stretch", clamp=True)
    st.stop()

selector_col, metric_col, size_col = st.columns([2.2, 1, 1])
with selector_col:
    selected_index = st.selectbox(
        "กรอบป้ายที่ตรวจพบ",
        range(len(candidates)),
        format_func=lambda index: f"ตัวเลือก {index + 1} — คะแนนจัดอันดับ {candidates[index].score * 100:.0f}/100",
    )
candidate = candidates[selected_index]
with metric_col:
    st.metric("คะแนนจัดอันดับ", f"{candidate.score * 100:.0f}/100",
              help="คะแนนเปรียบเทียบตัวเลือกในภาพเดียวกัน ไม่ใช่เปอร์เซ็นต์ความแม่นยำ")
with size_col:
    st.metric("ขนาดภาพ", f"{vehicle_image.shape[1]}×{vehicle_image.shape[0]}")

if candidate.character_evidence < 0.25:
    st.warning("หลักฐานตัวอักษรในกรอบนี้ยังน้อย กรุณาตรวจกรอบบนภาพรถ หรือเลือกตัวเลือกอื่นก่อนนำผลไปใช้")

use_manual = False
manual_points = candidate.points.copy()
image_key = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
with st.expander("กรอบไม่ตรง? ปรับพิกัดมุมด้วยตนเอง"):
    st.caption("พิกัดเรียงเป็น บนซ้าย → บนขวา → ล่างขวา → ล่างซ้าย")
    point_columns = st.columns(4)
    point_names = ["บนซ้าย", "บนขวา", "ล่างขวา", "ล่างซ้าย"]
    for index, column in enumerate(point_columns):
        with column:
            st.markdown(f"**{point_names[index]}**")
            manual_points[index, 0] = st.number_input(
                "X", 0, vehicle_image.shape[1] - 1, int(candidate.points[index, 0]), key=f"x-{image_key}-{selected_index}-{index}"
            )
            manual_points[index, 1] = st.number_input(
                "Y", 0, vehicle_image.shape[0] - 1, int(candidate.points[index, 1]), key=f"y-{image_key}-{selected_index}-{index}"
            )
    use_manual = st.checkbox("ใช้พิกัดที่แก้ไข", key=f"manual-{image_key}-{selected_index}")

try:
    plate_points = order_points(manual_points if use_manual else candidate.points)
    before_crop = crop_plate(vehicle_image, plate_points)
    result = rectify_plate(
        vehicle_image,
        plate_points,
        feature_detector=feature_detector,
        ratio_threshold=ratio_threshold,
        ransac_threshold=ransac_threshold,
    )
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
        st.markdown("**2. ผลปรับมุมป้าย**")
        show_bgr(result.image, "Perspective Transform")
    with ready_col:
        st.markdown("**3. ภาพเตรียมสำหรับ OCR**")
        st.image(ocr_ready, caption="Grayscale + Contrast + Denoise", width="stretch", clamp=True)

    if "+ KNN ratio + RANSAC" in result.method:
        st.success(
            f"ปรับป้ายด้วย {result.feature_detector} + KNN + Lowe's ratio test + RANSAC สำเร็จ "
            f"— inliers {result.feature_inliers}/{result.good_matches}"
        )
    elif result.method == "RANSAC edge fallback":
        st.warning(
            f"Feature matching ยังไม่เสถียร ({result.feature_failure_reason}) "
            f"จึงใช้ RANSAC จากขอบป้ายแทน — inliers {result.inliers}/{result.correspondences}"
        )
    else:
        st.warning(
            f"Feature matching ยังไม่เสถียร ({result.feature_failure_reason}) "
            f"— {result.geometry_warning}"
        )

    for warning in result.quality_warnings:
        st.warning(warning)

    download_one, download_two = st.columns(2)
    with download_one:
        st.download_button("ดาวน์โหลดป้ายที่ปรับตรง", encode_png(result.image), "plate-rectified.png", "image/png", width="stretch")
    with download_two:
        st.download_button("ดาวน์โหลดภาพพร้อม OCR", encode_png(ocr_ready), "plate-ocr-ready.png", "image/png", width="stretch", type="primary")

with details:
    show_bgr(draw_detection(vehicle_image, plate_points), "กรอบที่นำไปประมวลผล")
    detail_one, detail_two, detail_three, detail_four = st.columns(4)
    detail_one.metric("Feature detector", result.feature_detector)
    detail_two.metric("Keypoints", f"{result.source_keypoints} → {result.target_keypoints}")
    detail_three.metric("Good matches", result.good_matches)
    detail_four.metric("RANSAC inliers", f"{result.feature_inliers}/{result.good_matches}")
    st.caption(
        f"วิธีที่ใช้จริง: {result.method} · Lowe's ratio = {result.ratio_threshold:.2f} · "
        f"Feature inlier ratio = {result.feature_inlier_ratio * 100:.1f}% · "
        f"ผลลัพธ์ {result.image.shape[1]}×{result.image.shape[0]} px"
    )
    if result.match_visualization is not None:
        st.markdown("**คู่จุดที่ผ่าน Lowe's ratio test และ RANSAC**")
        show_bgr(result.match_visualization, "เส้นสีเขียวคือ inlier matches ที่ใช้คำนวณ Homography")
    elif result.feature_failure_reason:
        st.info(f"ไม่มีภาพคู่จุด: {result.feature_failure_reason}")
    with st.expander("ข้อมูลทางเทคนิค"):
        st.markdown("**Homography matrix**")
        st.code(np.array2string(result.homography, precision=5, suppress_small=True))
        mask_one, mask_two, mask_three = st.columns(3)
        mask_one.image(diagnostics["edges"], caption="Canny edges", width="stretch", clamp=True)
        mask_two.image(diagnostics["candidate_mask"], caption="Connected edge mask", width="stretch", clamp=True)
        mask_three.image(diagnostics["text_mask"], caption="Character-group mask", width="stretch", clamp=True)

st.caption("ตรวจว่าครอบครบทั้งป้ายและตัวอักษรไม่บิดก่อนนำไปใช้กับ OCR ความชัดยังขึ้นกับภาพต้นฉบับ")
