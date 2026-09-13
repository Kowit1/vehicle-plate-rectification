import unittest

import cv2
import numpy as np

from plate_rectification import crop_plate, decode_image, detect_plate_candidates, order_points, preprocess_for_ocr, rectify_plate


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
        self.assertEqual(diagnostics["edges"].shape, image.shape[:2])

        crop = crop_plate(image, detected.points)
        result = rectify_plate(image, detected.points)
        processed = preprocess_for_ocr(result.image)
        self.assertGreater(crop.size, 0)
        self.assertEqual(processed.ndim, 2)
        self.assertGreater(result.image.shape[1], result.image.shape[0])
        self.assertTrue(np.isfinite(result.homography).all())
        self.assertGreaterEqual(result.inliers, 4)

    def test_ransac_rectification_from_known_quad(self):
        image, quad = synthetic_vehicle()
        result = rectify_plate(image, quad)
        self.assertEqual(result.method, "RANSAC edge homography")
        self.assertGreater(result.inliers, 8)
        self.assertGreater(result.image.shape[1] / result.image.shape[0], 2.0)


if __name__ == "__main__":
    unittest.main()
