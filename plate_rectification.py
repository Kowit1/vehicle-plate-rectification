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
    feature_detector: str
    source_keypoints: int
    target_keypoints: int
    good_matches: int
    feature_inliers: int
    feature_inlier_ratio: float
    ratio_threshold: float
    match_visualization: np.ndarray | None
    feature_failure_reason: str | None


@dataclass
class _FeatureMatchResult:
    homography: np.ndarray | None
    source_keypoints: int
    target_keypoints: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    visualization: np.ndarray | None
    failure_reason: str | None


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


def _expand_text_quad(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Expand a glyph-group quad to include the plate margins and province line."""
    quad = points.astype(np.float32).copy()
    top_vector = quad[1] - quad[0]
    bottom_vector = quad[2] - quad[3]
    left_vector = quad[3] - quad[0]
    right_vector = quad[2] - quad[1]

    quad[0] -= 0.03 * left_vector + 0.02 * top_vector
    quad[1] -= 0.03 * right_vector - 0.02 * top_vector
    quad[3] += 0.12 * left_vector - 0.02 * bottom_vector
    quad[2] += 0.12 * right_vector + 0.02 * bottom_vector

    height, width = image_shape
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def _line_intersection(first: np.ndarray, second: np.ndarray, third: np.ndarray, fourth: np.ndarray) -> np.ndarray | None:
    first_direction = second - first
    second_direction = fourth - third
    denominator = first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0]
    if abs(float(denominator)) < 1e-5:
        return None
    offset = third - first
    scale = (offset[0] * second_direction[1] - offset[1] * second_direction[0]) / denominator
    return first + scale * first_direction


def _text_quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Recover a plate quad while preserving a strongly sloped lower edge."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quad_approx = None
    for epsilon in (0.02, 0.03, 0.04, 0.05, 0.06, 0.075):
        simplified = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(simplified) == 4 and cv2.isContourConvex(simplified):
            quad_approx = simplified
            break
    if quad_approx is None:
        return None

    quad = order_points(quad_approx.reshape(4, 2))
    fine_hull = cv2.approxPolyDP(hull, 0.018 * perimeter, True).reshape(-1, 2).astype(np.float32)
    bottom_left, bottom_right = quad[3], quad[2]
    bottom_direction = bottom_right - bottom_left
    bottom_length = float(np.linalg.norm(bottom_direction))
    if bottom_length < 2:
        return quad

    signed_distances = np.array(
        [
            (bottom_direction[0] * (point[1] - bottom_left[1]) - bottom_direction[1] * (point[0] - bottom_left[0]))
            / bottom_length
            for point in fine_hull
        ]
    )
    deepest_index = int(np.argmax(signed_distances))
    deepest = fine_hull[deepest_index]
    plate_height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))

    # A small protrusion is often a dealer frame, logo, or decorative strip.
    # Only alter the geometric corner when the lower-edge deviation is large
    # enough to indicate genuine perspective, otherwise rectification can make
    # an already straight plate look more skewed.
    if signed_distances[deepest_index] > max(10.0, plate_height * 0.26):
        projection = float(np.dot(deepest - bottom_left, bottom_direction) / (bottom_length**2))
        if projection >= 0.5:
            refined = _line_intersection(bottom_left, deepest, quad[1], bottom_right)
            if refined is not None and np.linalg.norm(refined - bottom_right) < plate_height * 0.85:
                quad[2] = refined
        else:
            refined = _line_intersection(deepest, bottom_right, quad[0], bottom_left)
            if refined is not None and np.linalg.norm(refined - bottom_left) < plate_height * 0.85:
                quad[3] = refined
    return quad


def _character_layout_score(gray: np.ndarray, points: np.ndarray) -> float:
    """Estimate whether a quad contains several aligned, character-sized strokes."""
    destination = np.array([[0, 0], [359, 0], [359, 119], [0, 119]], dtype=np.float32)
    normalized = cv2.warpPerspective(
        gray,
        cv2.getPerspectiveTransform(order_points(points), destination),
        (360, 120),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    normalized = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(normalized)
    blackhat = cv2.morphologyEx(
        normalized,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7)),
    )
    _, strokes = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strokes = cv2.morphologyEx(
        strokes,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(strokes)
    character_count = 0
    for x, _, width, height, area in stats[1:]:
        aspect = width / max(1, height)
        if (
            30 <= height <= 106
            and 7 <= width <= 86
            and area >= 60
            and 0.12 <= aspect <= 1.30
            and x >= 4
            and x + width <= 356
        ):
            character_count += 1
    return min(1.0, character_count / 6.0)


def _candidate_from_contour(
    contour: np.ndarray,
    edge_map: np.ndarray,
    gray: np.ndarray,
    text_group: bool = False,
    hsv: np.ndarray | None = None,
) -> PlateCandidate | None:
    height, width = gray.shape
    image_area = float(height * width)
    contour_area = float(cv2.contourArea(contour))
    maximum_area = 0.55 if text_group else 0.24
    if contour_area < image_area * 0.0012 or contour_area > image_area * maximum_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    text_quad = _text_quad_from_contour(contour) if text_group else None
    if text_quad is not None:
        points = text_quad
    elif len(approx) == 4 and cv2.isContourConvex(approx):
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

    if text_group:
        points = _expand_text_quad(points, gray.shape)

    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    plate_width = max(top, bottom)
    plate_height = max(left, right)
    if plate_height < 1:
        return None
    aspect = plate_width / plate_height
    minimum_aspect = 0.70 if text_group else 1.45
    if not minimum_aspect <= aspect <= 7.5:
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
    maximum_width = 0.95 if text_group else 0.72
    maximum_height = 1.00 if text_group else 0.34
    if w / width > maximum_width or h / height > maximum_height:
        return None

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
    pixels = max(1, cv2.countNonZero(mask))
    edge_density = cv2.countNonZero(cv2.bitwise_and(edge_map, edge_map, mask=mask)) / pixels
    inside_values = gray[mask > 0]
    contrast = min(1.0, float(inside_values.std()) / 70.0) if inside_values.size else 0.0
    chroma_fraction = 0.0
    dark_fraction = 0.0
    if hsv is not None:
        saturation = hsv[:, :, 1][mask > 0]
        value = hsv[:, :, 2][mask > 0]
        if saturation.size:
            chroma_fraction = float(np.mean((saturation > 70) & (value > 55)))
            dark_fraction = float(np.mean(value < 80))
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
    character_score = _character_layout_score(gray, points)
    score = (
        0.26 * aspect_score
        + 0.15 * rectangularity
        + 0.10 * min(1.0, edge_density / 0.20)
        + 0.10 * contrast
        + 0.15 * coverage_score
        + 0.19 * position_score
        + 0.05 * size_score
        + 0.12 * character_score
    )
    if text_group:
        # A connected cluster of dark glyph-like strokes is stronger evidence
        # than a plain rectangular vehicle part, especially in close-up views.
        score += 0.15
    # Thai plates are commonly white, yellow, red, or orange. A strong chromatic
    # surface is useful positive evidence, while a mostly dark region is more
    # likely to be a grille. These terms are deliberately bounded so neutral
    # white plates are neither rewarded nor penalized.
    score += 0.13 * float(np.clip((chroma_fraction - 0.30) / 0.60, 0.0, 1.0))
    score -= 0.16 * float(np.clip((dark_fraction - 0.55) / 0.35, 0.0, 1.0))
    return PlateCandidate(points, float(np.clip(score, 0.0, 1.0)), (x, y, w, h), rectangularity, edge_density)


def _refine_strongly_skewed_colored_candidate(
    candidate: PlateCandidate,
    adaptive_candidates: list[PlateCandidate],
    hsv: np.ndarray,
) -> PlateCandidate:
    """Replace an unstable colored-text quad with a better overlapping frame quad."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(candidate.points).astype(np.int32), 255)
    saturation = hsv[:, :, 1][mask > 0]
    value = hsv[:, :, 2][mask > 0]
    if not saturation.size:
        return candidate
    chroma_fraction = float(np.mean((saturation > 70) & (value > 55)))

    if chroma_fraction < 0.62:
        return candidate

    candidate_area = candidate.bbox[2] * candidate.bbox[3]
    candidate_centre = candidate.points.mean(axis=0)
    alternatives: list[tuple[float, PlateCandidate]] = []
    for alternative in adaptive_candidates:
        overlap = _iou(candidate.bbox, alternative.bbox)
        area_ratio = (alternative.bbox[2] * alternative.bbox[3]) / max(1, candidate_area)
        centre_distance = float(np.linalg.norm(alternative.points.mean(axis=0) - candidate_centre))
        distance_limit = max(candidate.bbox[2], candidate.bbox[3]) * 0.25
        rectangularity_gain = alternative.rectangularity - candidate.rectangularity
        if (
            overlap >= 0.35
            and 0.65 <= area_ratio <= 1.60
            and centre_distance <= distance_limit
            and rectangularity_gain >= 0.08
        ):
            quality = 0.55 * alternative.rectangularity + 0.30 * overlap - 0.15 * abs(1.0 - area_ratio)
            alternatives.append((quality, alternative))

    if not alternatives:
        return candidate
    refined = max(alternatives, key=lambda item: item[0])[1]
    return PlateCandidate(
        points=refined.points,
        score=min(1.0, max(candidate.score, refined.score) + 0.02),
        bbox=refined.bbox,
        rectangularity=refined.rectangularity,
        edge_density=refined.edge_density,
    )


def detect_plate_candidates(image: np.ndarray, limit: int = 8) -> tuple[list[PlateCandidate], dict[str, np.ndarray]]:
    """Find plausible plate quadrilaterals using complementary edge masks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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

    # Group dark characters on a lighter plate. This is deliberately separate
    # from the outer-edge detector so it also works on colored plates and on
    # close-up plates with strong perspective distortion.
    blackhat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)),
    )
    text_gradient = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    text_gradient = np.absolute(text_gradient)
    gradient_min, gradient_max = float(text_gradient.min()), float(text_gradient.max())
    text_gradient = ((text_gradient - gradient_min) / (gradient_max - gradient_min + 1e-6) * 255).astype(np.uint8)
    text_gradient = cv2.GaussianBlur(text_gradient, (5, 5), 0)
    text_gradient = cv2.morphologyEx(
        text_gradient,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)),
        iterations=2,
    )
    _, text_mask = cv2.threshold(text_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        iterations=1,
    )

    candidates: list[PlateCandidate] = []
    masks = ((closed_edges, False), (sobel_mask, False), (text_mask, True))
    for candidate_mask, text_group in masks:
        retrieval = cv2.RETR_EXTERNAL if text_group else cv2.RETR_LIST
        contours, _ = cv2.findContours(candidate_mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
            candidate = _candidate_from_contour(contour, edges, gray, text_group=text_group, hsv=hsv)
            if candidate is not None:
                candidates.append(candidate)

    adaptive_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    adaptive_mask = cv2.morphologyEx(
        adaptive_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    enhanced_adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    enhanced_adaptive_mask = cv2.morphologyEx(
        enhanced_adaptive_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    adaptive_candidates: list[PlateCandidate] = []
    for refinement_mask in (adaptive_mask, enhanced_adaptive_mask):
        adaptive_contours, _ = cv2.findContours(refinement_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(adaptive_contours, key=cv2.contourArea, reverse=True)[:180]:
            text_candidate = _candidate_from_contour(contour, edges, gray, text_group=True, hsv=hsv)
            frame_candidate = _candidate_from_contour(contour, edges, gray, text_group=False, hsv=hsv)
            if text_candidate is not None:
                adaptive_candidates.append(text_candidate)
            if frame_candidate is not None:
                adaptive_candidates.append(frame_candidate)

    candidates = [
        _refine_strongly_skewed_colored_candidate(candidate, adaptive_candidates, hsv)
        for candidate in candidates
    ]

    candidates.sort(key=lambda item: item.score, reverse=True)
    unique: list[PlateCandidate] = []
    for candidate in candidates:
        if all(_iou(candidate.bbox, accepted.bbox) < 0.55 for accepted in unique):
            unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique, {
        "edges": edges,
        "candidate_mask": closed_edges,
        "text_mask": text_mask,
        "adaptive_mask": adaptive_mask,
        "enhanced_adaptive_mask": enhanced_adaptive_mask,
    }


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


def _feature_match_homography(
    image: np.ndarray,
    quad: np.ndarray,
    provisional: np.ndarray,
    destination_quad: np.ndarray,
    detector_name: str,
    ratio_threshold: float,
    ransac_threshold: float,
) -> _FeatureMatchResult:
    """Refine the plate homography using local features, KNN matching and RANSAC."""
    detector_name = detector_name.upper()
    x, y, width, height = cv2.boundingRect(quad.astype(np.int32))
    padding = max(4, int(round(max(width, height) * 0.035)))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(image.shape[1], x + width + padding), min(image.shape[0], y + height + padding)
    source_roi = image[y0:y1, x0:x1].copy()
    if source_roi.size == 0 or provisional.size == 0:
        return _FeatureMatchResult(None, 0, 0, 0, 0, 0.0, None, "พื้นที่ป้ายไม่ถูกต้อง")

    local_quad = quad - np.array([x0, y0], dtype=np.float32)
    source_mask = np.zeros(source_roi.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, np.round(local_quad).astype(np.int32), 255)
    target_mask = np.full(provisional.shape[:2], 255, dtype=np.uint8)

    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(provisional, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    source_gray = clahe.apply(source_gray)
    target_gray = clahe.apply(target_gray)

    if detector_name == "SIFT":
        detector = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.018, edgeThreshold=12)
        norm = cv2.NORM_L2
    elif detector_name == "ORB":
        detector = cv2.ORB_create(
            nfeatures=2400,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            fastThreshold=7,
        )
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("รองรับ feature detector เฉพาะ SIFT หรือ ORB")

    source_kp, source_desc = detector.detectAndCompute(source_gray, source_mask)
    target_kp, target_desc = detector.detectAndCompute(target_gray, target_mask)
    source_count, target_count = len(source_kp), len(target_kp)
    if source_desc is None or target_desc is None or source_count < 2 or target_count < 2:
        reason = "ตรวจพบ keypoints ไม่เพียงพอสำหรับ descriptor matching"
        return _FeatureMatchResult(None, source_count, target_count, 0, 0, 0.0, None, reason)

    matcher = cv2.BFMatcher(normType=norm, crossCheck=False)
    pairs = matcher.knnMatch(source_desc, target_desc, k=2)
    good_matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance]
    if len(good_matches) < 8:
        reason = f"คู่จุดที่ผ่าน Lowe's ratio test มีเพียง {len(good_matches)} คู่ (ต้องการอย่างน้อย 8)"
        visualization = cv2.drawMatches(
            source_roi,
            source_kp,
            provisional,
            target_kp,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        ) if good_matches else None
        return _FeatureMatchResult(
            None, source_count, target_count, len(good_matches), 0, 0.0, visualization, reason
        )

    source_points = np.float32([source_kp[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    source_points[:, 0, 0] += x0
    source_points[:, 0, 1] += y0
    target_points = np.float32([target_kp[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        ransac_threshold,
    )
    inlier_flags = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.zeros(len(good_matches), bool)
    inliers = int(inlier_flags.sum())
    inlier_ratio = inliers / len(good_matches)
    visualization = cv2.drawMatches(
        source_roi,
        source_kp,
        provisional,
        target_kp,
        good_matches,
        None,
        matchColor=(44, 216, 151),
        singlePointColor=(120, 120, 120),
        matchesMask=inlier_flags.astype(np.uint8).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    minimum_inliers = max(6, int(round(len(good_matches) * 0.35)))
    if homography is None or not np.isfinite(homography).all():
        reason = "RANSAC ไม่สามารถคำนวณ Homography จากคู่จุดได้"
        return _FeatureMatchResult(None, source_count, target_count, len(good_matches), inliers, inlier_ratio, visualization, reason)
    if inliers < minimum_inliers:
        reason = f"RANSAC inliers มีเพียง {inliers}/{len(good_matches)} จุด"
        return _FeatureMatchResult(None, source_count, target_count, len(good_matches), inliers, inlier_ratio, visualization, reason)

    projected = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), homography).reshape(4, 2)
    diagonal = float(np.linalg.norm(destination_quad[2] - destination_quad[0]))
    mean_corner_error = float(np.linalg.norm(projected - destination_quad, axis=1).mean())
    if not cv2.isContourConvex(np.round(projected).astype(np.int32)) or mean_corner_error > diagonal * 0.22:
        reason = "Homography จาก feature matches บิดรูปเกินเกณฑ์ความปลอดภัย"
        return _FeatureMatchResult(None, source_count, target_count, len(good_matches), inliers, inlier_ratio, visualization, reason)

    return _FeatureMatchResult(
        homography, source_count, target_count, len(good_matches), inliers, inlier_ratio, visualization, None
    )


def rectify_plate(
    image: np.ndarray,
    points: np.ndarray,
    feature_detector: str = "SIFT",
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 4.0,
) -> RectificationResult:
    """Rectify a plate using feature matching with robust disclosed fallbacks."""
    if not 0.4 <= ratio_threshold <= 0.95:
        raise ValueError("Lowe's ratio threshold ต้องอยู่ระหว่าง 0.40 ถึง 0.95")
    if not 0.5 <= ransac_threshold <= 20.0:
        raise ValueError("RANSAC threshold ต้องอยู่ระหว่าง 0.5 ถึง 20 pixels")

    quad = order_points(points)
    widths = [np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])]
    heights = [np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])]
    target_width = int(np.clip(round(max(widths)), 120, 1200))
    target_height = int(np.clip(round(max(heights)), 40, 500))
    if target_width / target_height < 1.4:
        target_width = int(round(target_height * 2.5))

    destination_quad = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    corner_homography = cv2.getPerspectiveTransform(quad, destination_quad)
    provisional = cv2.warpPerspective(
        image,
        corner_homography,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    feature_result = _feature_match_homography(
        image,
        quad,
        provisional,
        destination_quad,
        feature_detector,
        ratio_threshold,
        ransac_threshold,
    )

    if feature_result.homography is not None:
        homography = feature_result.homography
        inliers = feature_result.inliers
        correspondences = feature_result.good_matches
        method = f"{feature_detector.upper()} + KNN ratio + RANSAC"
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source, destination = _edge_correspondences(gray, quad, target_width, target_height)
        homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 3.5)
        inliers = int(mask.sum()) if mask is not None else 0
        correspondences = len(source)
        minimum_inliers = max(8, int(len(source) * 0.45))
        if homography is not None and np.isfinite(homography).all() and inliers >= minimum_inliers:
            method = "RANSAC edge fallback"
        else:
            homography = corner_homography
            inliers = 4
            correspondences = 4
            method = "4-corner fallback"

    rectified = cv2.warpPerspective(
        image,
        homography,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return RectificationResult(
        image=rectified,
        homography=homography,
        method=method,
        inliers=inliers,
        correspondences=correspondences,
        feature_detector=feature_detector.upper(),
        source_keypoints=feature_result.source_keypoints,
        target_keypoints=feature_result.target_keypoints,
        good_matches=feature_result.good_matches,
        feature_inliers=feature_result.inliers,
        feature_inlier_ratio=feature_result.inlier_ratio,
        ratio_threshold=ratio_threshold,
        match_visualization=feature_result.visualization,
        feature_failure_reason=feature_result.failure_reason,
    )


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
