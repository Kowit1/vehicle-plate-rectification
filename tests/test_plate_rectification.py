import unittest

import cv2
import numpy as np

from plate_rectification import (
    _text_quad_from_contour,
    crop_plate,
    decode_image,
    detect_plate_candidates,
    order_points,
    preprocess_for_ocr,
    rectify_plate,
)


def synthetic_vehicle() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((720, 1100, 3), (62, 70, 78), dtype=np.uint8)
    cv2.rectangle(image, (135, 190), (965, 610), (41, 48, 55), -1)
    cv2.ellipse(image, (275, 600), (90, 90), 0, 0, 360, (18, 20, 23), -1)
    cv2.ellipse(image, (825, 600), (90, 90), 0, 0, 360, (18, 20, 23), -1)

    plate = np.full((130, 390, 3), 238, dtype=np.uint8)
    cv2.rectangle(plate, (3, 3), (386, 126), (24, 24, 24), 5)
    cv2.putText(plate, "1ABC 234", (37, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (18, 18, 18), 4, cv2.LINE_AA)
    source = np.array([[0, 0], [389, 0], [389, 129], [0, 129]], np.float32)
    destination = np.array([[350, 420], [765, 392], [785, 530], [330, 548]], np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(plate, transform, (1100, 720))
    mask = cv2.warpPerspective(np.full((130, 390), 255, np.uint8), transform, (1100, 720))
    image[mask > 0] = warped[mask > 0]
    return image, destination


class PlatePipelineTests(unittest.TestCase):
    def test_point_ordering(self):
        shuffled = [[90, 70], [10, 10], [10, 70], [90, 10]]
        ordered = order_points(shuffled)
        np.testing.assert_allclose(ordered, [[10, 10], [90, 10], [90, 70], [10, 70]])

    def test_decode_rejects_non_image(self):
        with self.assertRaises(ValueError):
            decode_image(b"not an image")

    def test_blank_image_has_no_plate(self):
        image = np.full((500, 800, 3), 127, np.uint8)
        candidates, _ = detect_plate_candidates(image)
        self.assertEqual(candidates, [])

    def test_end_to_end_synthetic_plate(self):
        image, expected = synthetic_vehicle()
        candidates, diagnostics = detect_plate_candidates(image)
        self.assertTrue(candidates)
        detected = candidates[0]
        self.assertLess(float(np.linalg.norm(expected.mean(axis=0) - detected.points.mean(axis=0))), 80)
        self.assertGreater(detected.score, 0.45)
        self.assertLessEqual(detected.score, 1.0)
        self.assertEqual(diagnostics["edges"].shape, image.shape[:2])

        crop = crop_plate(image, detected.points)
        result = rectify_plate(image, detected.points)
        processed = preprocess_for_ocr(result.image)
        self.assertGreater(crop.size, 0)
        self.assertEqual(processed.ndim, 2)
        self.assertGreater(result.image.shape[1], result.image.shape[0])
        self.assertTrue(np.isfinite(result.homography).all())
        self.assertGreaterEqual(result.inliers, 4)

    def test_sift_knn_ratio_and_ransac_from_known_quad(self):
        image, quad = synthetic_vehicle()
        result = rectify_plate(image, quad)
        self.assertEqual(result.method, "SIFT + KNN ratio + RANSAC")
        self.assertGreater(result.source_keypoints, 20)
        self.assertGreater(result.target_keypoints, 20)
        self.assertGreater(result.good_matches, 8)
        self.assertGreater(result.feature_inliers, 6)
        self.assertGreater(result.feature_inlier_ratio, 0.35)
        self.assertIsNotNone(result.match_visualization)
        self.assertGreater(result.image.shape[1] / result.image.shape[0], 2.0)

    def test_orb_is_supported_with_ratio_test_and_ransac(self):
        image, quad = synthetic_vehicle()
        result = rectify_plate(image, quad, feature_detector="ORB", ratio_threshold=0.80)
        self.assertEqual(result.feature_detector, "ORB")
        self.assertGreater(result.source_keypoints, 20)
        self.assertGreater(result.good_matches, 8)
        self.assertGreater(result.feature_inliers, 6)
        self.assertIn(result.method, ["ORB + KNN ratio + RANSAC", "RANSAC edge fallback"])

    def test_featureless_plate_uses_disclosed_fallback(self):
        image = np.full((240, 500, 3), 210, np.uint8)
        quad = np.array([[80, 70], [420, 70], [420, 175], [80, 175]], np.float32)
        result = rectify_plate(image, quad)
        self.assertIn(result.method, ["RANSAC edge fallback", "4-corner fallback"])
        self.assertIsNotNone(result.feature_failure_reason)
        self.assertEqual(result.good_matches, 0)
        self.assertTrue(np.isfinite(result.homography).all())

    def test_large_grille_does_not_outrank_lower_plate(self):
        image = np.full((900, 900, 3), 185, dtype=np.uint8)
        # A wide high-contrast grille is a common false positive.
        grille = np.array([[55, 270], [845, 270], [805, 485], [95, 485]], np.int32)
        cv2.fillConvexPoly(image, grille, (28, 31, 34))
        cv2.polylines(image, [grille], True, (210, 214, 218), 12)
        for x in range(115, 810, 45):
            cv2.line(image, (x, 290), (x - 20, 465), (105, 110, 115), 4)

        plate = np.array([[295, 625], [620, 612], [632, 748], [286, 757]], np.int32)
        cv2.fillConvexPoly(image, plate, (238, 238, 232))
        cv2.polylines(image, [plate], True, (30, 70, 55), 8)
        cv2.putText(image, "ABC 123", (320, 710), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (25, 80, 55), 4, cv2.LINE_AA)

        candidates, _ = detect_plate_candidates(image)
        self.assertTrue(candidates)
        detected_centre = candidates[0].points.mean(axis=0)
        self.assertGreater(float(detected_centre[1]), 570)
        self.assertLess(float(np.linalg.norm(detected_centre - plate.mean(axis=0))), 75)

    def test_small_colored_plate_outranks_dark_grille(self):
        image = np.full((600, 900, 3), (70, 82, 95), dtype=np.uint8)
        grille = np.array([[210, 250], [720, 225], [750, 455], [185, 480]], np.int32)
        cv2.fillConvexPoly(image, grille, (24, 27, 31))
        cv2.polylines(image, [grille], True, (120, 35, 28), 12)
        for x in range(220, 720, 38):
            cv2.line(image, (x, 255), (x - 20, 455), (78, 84, 90), 5)

        plate = np.array([[405, 315], [548, 307], [552, 382], [402, 389]], np.int32)
        cv2.fillConvexPoly(image, plate, (32, 58, 222))
        cv2.polylines(image, [plate], True, (225, 225, 218), 6)
        cv2.putText(image, "1-7659", (414, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (16, 18, 20), 3, cv2.LINE_AA)
        cv2.putText(image, "BANGKOK", (425, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (18, 20, 22), 1, cv2.LINE_AA)

        candidates, _ = detect_plate_candidates(image)
        self.assertTrue(candidates)
        detected = candidates[0]
        self.assertLess(float(np.linalg.norm(detected.points.mean(axis=0) - plate.mean(axis=0))), 45)
        self.assertLess(detected.bbox[2], 230)

    def test_close_up_oblique_plate_is_detected(self):
        image = np.full((420, 640, 3), (150, 155, 160), dtype=np.uint8)
        plate = np.full((180, 430, 3), 235, dtype=np.uint8)
        cv2.rectangle(plate, (4, 4), (425, 175), (35, 40, 42), 7)
        cv2.putText(plate, "49-92", (52, 122), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (15, 20, 22), 7, cv2.LINE_AA)
        source = np.array([[0, 0], [429, 0], [429, 179], [0, 179]], np.float32)
        expected = np.array([[170, 8], [540, 80], [420, 412], [75, 220]], np.float32)
        transform = cv2.getPerspectiveTransform(source, expected)
        warped = cv2.warpPerspective(plate, transform, (640, 420))
        mask = cv2.warpPerspective(np.full((180, 430), 255, np.uint8), transform, (640, 420))
        image[mask > 0] = warped[mask > 0]

        candidates, _ = detect_plate_candidates(image)
        self.assertTrue(candidates)
        detected_centre = candidates[0].points.mean(axis=0)
        self.assertLess(float(np.linalg.norm(detected_centre - expected.mean(axis=0))), 90)

    def test_decorative_lower_tab_does_not_tilt_straight_plate(self):
        contour = np.array(
            [[[324, 148]], [[321, 236]], [[164, 255]], [[87, 238]], [[79, 152]]],
            dtype=np.int32,
        )
        quad = _text_quad_from_contour(contour)
        self.assertIsNotNone(quad)
        ordered = order_points(quad)
        self.assertLess(abs(float(ordered[2, 1] - ordered[3, 1])), 8.0)
        self.assertLess(float(ordered[3, 1]), 245.0)


if __name__ == "__main__":
    unittest.main()
