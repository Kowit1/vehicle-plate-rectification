# External-image evaluation — 2026-09-13

Compared the Python pipeline before these changes (commit `7e55952`) with the improved pipeline. The application calls the same detection and rectification functions. All evaluation inputs retain vehicle context; they are not isolated license-plate crops.

## Localization results

A hit means that the first-ranked candidate's axis-aligned bounding box has IoU ≥ 0.50 with the annotated plate box. IoU is intersection area divided by union area. Each evaluated image has one annotated target plate. These are image-level localization hit rates, **not OCR accuracy, corner accuracy, or proof of a usable rectified plate**.

| Set | Images | Baseline top-1 hits | Improved top-1 hits | Improved any-of-top-8 hits |
|---|---:|---:|---:|---:|
| Thai development images, front/high-angle view | 28 | 4 (14.3%) | 23 (82.1%) | 26 |
| Supplementary Brazilian development images, front/rear | 12 | 10 (83.3%) | 12 (100%) | 12 |
| Thai held-out validation images | 37 | 12 (32.4%) | 35 (94.6%) | 36 |

Seven of the 12 supplementary images are rear views. Top-1 localization on this subset improved from 6/7 to 7/7; this is not an additional independent set. Any-of-top-8 uses the annotation to inspect alternatives after processing and is not the application's automatic success rate.

The seven user-provided Car examples were also checked visually before/after. The improved version localized the plate region in all seven; the red foreground car in the two-car photograph now includes substantially more of its plate. Strongly oblique and blurred examples still have imperfect rectification. Approximate manually drawn boxes on these seven examples are not mixed into the external-dataset rates.

## Protocol and provenance

- **Thai data:** [naruesorn / thai license plate](https://universe.roboflow.com/naruesorn/thai-license-plate-j6y9l), CC BY 4.0. Downloaded the files distributed by [xFieldxGod/Detect_Plate](https://github.com/xFieldxGod/Detect_Plate) at revision `e467840c2b1f86ebd2b61d0871fc5ac75f1fa944`.
- Development: every JPG directly in `plate/test/images` (28), excluding notebook checkpoint duplicates; matching YOLO labels in `plate/test/labels`. This source split became our **development** set once we inspected failures and adjusted the algorithm.
- Held-out evaluation: every JPG directly in `plate/valid/images` (37), with matching labels. This split was downloaded and inspected only after freezing the improved pipeline. SHA-256 of the pipeline at that point: `afad53e91ea1a27743ff4770812384ff6edc238185543390cabaae3e1317f4b8`. The pipeline was unchanged during this held-out comparison. No byte-identical images were found between these two splits; vehicle/near-frame independence has not been established.
- **Supplementary data:** [OpenALPR benchmarks](https://github.com/openalpr/benchmarks) at revision `9790ed20d475be1e940a7e2a3f492a97c2ebed9b`. Selected 12 evenly spaced indices across the alphabetically sorted JPG names in `endtoend/br`, before inspecting outputs, and used the supplied bounding boxes. Images: AYO9034, JQS5683, MTW5608, NYM7544, NZW2197, OKP7250, OUM7311, OZC8400, OZV6697, PJI5396, PJU2853, PYB6477.
- Evaluate the unmodified source image with `decode_image`, `detect_plate_candidates` (default limit 8), and `rectify_plate` (default SIFT settings). Scale source boxes with `decode_image` if resizing occurs. Compare candidate bounding boxes to labels in the same coordinate system. No manual candidate selection contributes to top-1 counts.
- Unit and Streamlit integration tests: **19 passed**, plus 6 parameterized subtests. These include detection on synthetic images, SIFT/ORB, unsafe homographies, missing edges, invalid corners, fallback behavior, and upload/result rendering.

Downloaded photographs and generated result images are kept outside the application repository. The developer's local evaluation bundle retains per-image hashes, source URLs, annotations, predicted boxes, method information, comparison images, and scripts. Dataset images are not required to run the app or automated tests.

## What changed

- Added hypotheses from lightly closed brightness/color masks to preserve physical plate boundaries, alongside existing edge/text candidates.
- Rank candidates using aligned glyph components inside margins, evidence of a closed plate surface, and plate shape. Background edges or a tight text-row crop receive less weight. Ranking scores are not displayed as confidence percentages.
- Reject duplicate/concave corners and out-of-image manual coordinates; require spatial coverage of feature inliers.
- Apply common homography checks to both feature and edge RANSAC. Reject reflections, a projective horizon crossing the plate, excessive corner drift, and large area changes. Edge matching now uses observed edges only and chooses the closest supported point.
- Fall back to the supplied four corners when an edge transform fails checks, disclose that fallback, and warn about insufficient source resolution. Upload-specific manual-edit state prevents coordinates from a previous image being reused accidentally.

## Remaining limitations

The held-out set is from the same collection/camera style as the development images. Source documentation says Thai images were stretched to 640×640; they are not untouched camera frames. Their aspect ratios and photographic conditions limit conclusions about other cameras. Source annotations have not been independently verified pixel by pixel.

Two held-out targets still fail top-1 IoU: image216 and image505. Development failures remain on dark/red plates and confusing vehicle/taxi details. Some boxes passing IoU still include extra background, truncate plate margins, or use imperfect corners. There are no ground-truth corner labels, quantified OCR scores, or systematic negative/multiple-plate/night/motion-blur evaluations here. The rear examples are Brazilian parked vehicles, not a Thai moving-CCTV benchmark.

The feature reference is still generated from the detected quadrilateral. A stable homography validates consistency with those corners, not whether those corners are correct. More varied independently labeled data and a separately evaluated plate/corner detector remain future improvements.
