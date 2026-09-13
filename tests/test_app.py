"""Exercise the actual Streamlit script with an in-memory upload."""
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / 'app.py'


class AppTests(unittest.TestCase):
    def test_empty_upload_shows_start_prompt(self):
        app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(app.info)

    def test_upload_renders_ranking_and_results(self):
        image = np.full((500, 800, 3), 80, np.uint8)
        cv2.rectangle(image, (250, 300), (560, 410), (235, 235, 235), -1)
        cv2.rectangle(image, (250, 300), (560, 410), (15, 15, 15), 4)
        cv2.putText(image, 'ABC 123', (268, 378), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (15, 15, 15), 3)
        uploaded = io.BytesIO(cv2.imencode('.png', image)[1].tobytes())
        with patch('streamlit.file_uploader', return_value=uploaded):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(metric.label == 'คะแนนจัดอันดับ' for metric in app.metric))
        self.assertTrue(any(heading.value == 'ผลลัพธ์' for heading in app.subheader))
