"""Computer-vision pipeline for detecting and rectifying a license plate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


MAX_IMAGE_SIDE = 2200


@dataclass
class PlateCandidate:
    points: np.ndarray
    score: float
    bbox: tuple[int, int, int, int]
    rectangularity: float
    edge_density: float


@dataclass
class RectificationResult:
    image: np.ndarray
    homography: np.ndarray
    method: str
    inliers: int
    correspondences: int


def decode_image(file_bytes: bytes, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    """Decode an uploaded image and constrain it to a practical working size."""
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("ไฟล์นี้ไม่ใช่ภาพที่ OpenCV อ่านได้")

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def order_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Return quadrilateral points ordered TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        raise ValueError("พิกัดมุมป้ายไม่ถูกต้อง")

    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    cyclic = pts[np.argsort(angles)]
    start = int(np.argmin(cyclic.sum(axis=1)))
    cyclic = np.roll(cyclic, -start, axis=0)
    first_edge = cyclic[1] - cyclic[0]
    second_edge = cyclic[2] - cyclic[1]
    signed_turn = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    if signed_turn < 0:
        cyclic = cyclic[[0, 3, 2, 1]]

    area = abs(cv2.contourArea(cyclic))
    if area < 25:
        raise ValueError("พื้นที่ป้ายเล็กเกินไปหรือมุมทับกัน")
    return cyclic.astype(np.float32)


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _candidate_from_contour(
    contour: np.ndarray,
    edge_map: np.ndarray,
    gray: np.ndarray,
) -> PlateCandidate | None:
    height, width = gray.shape
    image_area = float(height * width)
    contour_area = float(cv2.contourArea(contour))
    if contour_area < image_area * 0.0012 or contour_area > image_area * 0.24:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        points = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) < 8:
            return None
        points = cv2.boxPoints(rect).astype(np.float32)

    try:
        points = order_points(points)
    except ValueError:
        return None

    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    plate_width = max(top, bottom)
    plate_height = max(left, right)
    if plate_height < 1:
        return None
    aspect = plate_width / plate_height
    if not 1.45 <= aspect <= 7.5:
        return None

    polygon_area = abs(cv2.contourArea(points))
    if polygon_area <= 0:
        return None
    rectangularity = min(1.0, contour_area / polygon_area)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    x, y = max(0, x), max(0, y)
    w, h = min(width - x, w), min(height - y, h)
    if w < 30 or h < 10:
        return None
    # A plate in a vehicle/CCTV frame should not span nearly the whole image.
    # This rejects bumpers, grilles and windshields that otherwise look rectangular.
    if w / width > 0.72 or h / height > 0.34:
        return None

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
    pixels = max(1, cv2.countNonZero(mask))
    edge_density = cv2.countNonZero(cv2.bitwise_and(edge_map, edge_map, mask=mask)) / pixels
    inside_values = gray[mask > 0]
    contrast = min(1.0, float(inside_values.std()) / 70.0) if inside_values.size else 0.0
    expected_aspect = 3.0
    aspect_score = float(np.exp(-abs(np.log(aspect / expected_aspect))))
    coverage = polygon_area / image_area
    coverage_score = min(1.0, coverage / 0.025)
    centre_x = float(points[:, 0].mean()) / width
    centre_y = float(points[:, 1].mean()) / height
    horizontal_score = float(np.exp(-abs(centre_x - 0.50) / 0.30))
    vertical_score = float(np.exp(-abs(centre_y - 0.72) / 0.28))
    position_score = float(np.sqrt(horizontal_score * vertical_score))
    width_fraction = w / width
    size_score = float(np.exp(-abs(np.log(max(width_fraction, 0.01) / 0.30))))
    score = (
        0.26 * aspect_score
        + 0.15 * rectangularity
        + 0.10 * min(1.0, edge_density / 0.20)
        + 0.10 * contrast
        + 0.15 * coverage_score
        + 0.19 * position_score
        + 0.05 * size_score
    )
    return PlateCandidate(points, float(score), (x, y, w, h), rectangularity, edge_density)


def detect_plate_candidates(image: np.ndarray, limit: int = 8) -> tuple[list[PlateCandidate], dict[str, np.ndarray]]:
    """Find plausible plate quadrilaterals using complementary edge masks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.bilateralFilter(clahe, 9, 60, 60)
    edges = cv2.Canny(blurred, 55, 170)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    closed_edges = cv2.morphologyEx(
        closed_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    sobel = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    _, sobel_mask = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sobel_mask = cv2.morphologyEx(sobel_mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)

    candidates: list[PlateCandidate] = []
    for candidate_mask in (closed_edges, sobel_mask):
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
            candidate = _candidate_from_contour(contour, edges, gray)
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    unique: list[PlateCandidate] = []
    for candidate in candidates:
        if all(_iou(candidate.bbox, accepted.bbox) < 0.55 for accepted in unique):
            unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique, {"edges": edges, "candidate_mask": closed_edges}


def draw_detection(image: np.ndarray, points: np.ndarray, label: str = "PLATE") -> np.ndarray:
    output = image.copy()
    quad = order_points(points).astype(np.int32)
    cv2.polylines(output, [quad], True, (44, 216, 151), max(2, image.shape[1] // 500), cv2.LINE_AA)
    x, y = int(quad[:, 0].min()), int(quad[:, 1].min())
    cv2.putText(output, label, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (44, 216, 151), 2, cv2.LINE_AA)
    return output


def crop_plate(image: np.ndarray, points: np.ndarray, padding: float = 0.05) -> np.ndarray:
    quad = order_points(points)
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    pad_x, pad_y = int(w * padding), int(h * padding)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
    return image[y0:y1, x0:x1].copy()


def _edge_correspondences(
    gray: np.ndarray,
    quad: np.ndarray,
    target_width: int,
    target_height: int,
    samples_per_edge: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 150)
    destination_corners = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []
    search_radius = max(3, int(round(min(target_width, target_height) * 0.055)))

    for edge_index in range(4):
        start = quad[edge_index]
        end = quad[(edge_index + 1) % 4]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 2:
            continue
        normal = np.array([-vector[1], vector[0]], dtype=np.float32) / length
        dst_start = destination_corners[edge_index]
        dst_end = destination_corners[(edge_index + 1) % 4]
        for t in np.linspace(0.0, 1.0, samples_per_edge):
            expected = start + vector * t
            best = expected
            best_strength = -1
            for offset in range(-search_radius, search_radius + 1):
                probe = expected + normal * offset
                px, py = int(round(probe[0])), int(round(probe[1]))
                if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                    strength = int(edges[py, px])
                    if strength > best_strength:
                        best_strength = strength
                        best = probe
            source_points.append(best)
            destination_points.append(dst_start + (dst_end - dst_start) * t)
    return np.asarray(source_points, np.float32), np.asarray(destination_points, np.float32)


def rectify_plate(image: np.ndarray, points: np.ndarray) -> RectificationResult:
    """Estimate a robust edge homography and warp the plate into a front-facing view."""
    quad = order_points(points)
    widths = [np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])]
    heights = [np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])]
    target_width = int(np.clip(round(max(widths)), 120, 1200))
    target_height = int(np.clip(round(max(heights)), 40, 500))
    if target_width / target_height < 1.4:
        target_width = int(round(target_height * 2.5))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source, destination = _edge_correspondences(gray, quad, target_width, target_height)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 3.5)
    inliers = int(mask.sum()) if mask is not None else 0
    minimum_inliers = max(8, int(len(source) * 0.45))

    if homography is not None and np.isfinite(homography).all() and inliers >= minimum_inliers:
        method = "RANSAC edge homography"
    else:
        destination_quad = np.array(
            [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(quad, destination_quad)
        inliers = 4
        source = quad
        method = "4-corner fallback"

    rectified = cv2.warpPerspective(
        image,
        homography,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return RectificationResult(rectified, homography, method, inliers, len(source))


def preprocess_for_ocr(rectified: np.ndarray) -> np.ndarray:
    """Convert to grayscale, increase local contrast, and reduce noise."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    scale = min(3.0, max(1.0, 720.0 / max(1, gray.shape[1])))
    if scale > 1.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(gray, 7, 42, 42)
    enhanced = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8)).apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("ไม่สามารถสร้างไฟล์ PNG ได้")
    return encoded.tobytes()
