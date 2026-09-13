# Vehicle Plate Rectification for LPR

A browser application that accepts **one vehicle or CCTV still image**, proposes a plate region, rectifies its perspective and produces a grayscale image for downstream OCR. No frontal reference is required in the primary workflow. An optional two-image ORB/RANSAC lab is retained for experiments.

Primary workflow: vehicle image → plate detection → original plate crop → four-corner homography → rectified color plate → grayscale, contrast enhancement and denoising → PNG download.

The single-image path uses `getPerspectiveTransform` from detected corners, **not RANSAC homography estimation**. The optional lab uses `findHomography(..., RANSAC)` from feature correspondences. This distinction must be preserved in any project report; compliance with the original assignment PDF has not been verified.

## Live pipeline

- Multi-pass automatic plate proposal using two Canny thresholds, horizontal gradients, adaptive thresholding, morphology and geometry/texture scoring
- Ranked plate candidates with previous/next controls when the first proposal is not correct
- Drag-and-drop upload, automatic processing after upload and draggable corner fine-tuning
- Manual four-corner fallback for difficult real-world images
- ORB keypoint and binary descriptor extraction
- KNN Hamming descriptor matching
- Lowe-style ratio test
- RANSAC outlier rejection
- 3 × 3 planar homography and perspective warping
- OCR-ready equalization, denoising and adaptive thresholding
- Built-in reproducible demo and failure handling
- Client-side processing: uploaded images never leave the browser

OpenCV.js runs locally in the browser without a Python server. Automated tests include real OpenCV detection/rectification on a synthetic angled vehicle image, blank-image rejection, and UI upload/initialization/failure-state tests. These tests do not establish accuracy on real CCTV footage.

## Run locally

Requirements: Node.js 20 or newer.

```bash
npm install
npm run dev
```

Open the local URL shown by Vite.

## Test and build

```bash
npm test
npm run build
```

The production files are generated in `dist/`.

## Deploy to Vercel

### Vercel Dashboard

1. Push this project to GitHub.
2. In Vercel, choose **Add New → Project**.
3. Import the `Kowit1/vehicle-plate-rectification` repository.
4. Vercel should detect **Vite** automatically.
5. Build Command: `npm run build`
6. Output Directory: `dist`
7. Click **Deploy**.

### Vercel CLI

```bash
npm install -g vercel
vercel
vercel --prod
```

No environment variables or server functions are required.

## Application modes

### Single-image vehicle rectification (primary workflow)

Upload or drop one vehicle image (up to 25 MB). The app automatically searches for a plate, marks the proposed corners, and displays the original cropped plate, rectified plate and grayscale OCR preparation side by side. Download either output as PNG. If the selected region is wrong, try another candidate or manually select and drag four corners. The change-image button stays available even when detection fails. Replacing an image or editing corners invalidates previous downloads.

### Feature Matching Rectification (optional lab)

Upload an angled query image and a frontal reference image of the **same physical plate**. ORB extracts local descriptors, KNN generates candidate matches, the ratio test removes ambiguous pairs, and RANSAC estimates the homography used to warp the angled query onto the frontal reference plane. The app exposes keypoint, match and inlier counts, the inlier visualization, the 3 x 3 homography matrix and a downloadable rectified result.

## Honest limitations

- Feature matching cannot use a blank generic plate template because different plate numbers do not share enough local features. It requires the same physical plate.
- Automatic contour detection may fail on tiny, dark, blurred, borderless or heavily occluded plates.
- This accepts still images; it does not connect to a live CCTV stream. Rectification cannot recover missing or unreadable characters. Output aspect is estimated from visible edge lengths, not a calibrated physical plate size.
- OCR-ready preprocessing is included, but Thai OCR itself is intentionally outside the project scope.
- Real evaluation should include different angles, distances, lighting, blur levels and explicit failure cases.

## Suggested 10-minute presentation

| Time | Content |
|---|---|
| 0:00–0:45 | Problem and objective |
| 0:45–2:10 | End-to-end pipeline |
| 2:10–4:00 | ORB, descriptors and ratio test |
| 4:00–5:30 | RANSAC and homography |
| 5:30–8:20 | Live demo with group images |
| 8:20–9:15 | Failure/edge-case demonstration |
| 9:15–10:00 | Limitations, future work and conclusion |

Keep all five members visibly involved; the assignment deducts presentation points when a member does not contribute or the video exceeds 10 minutes.
