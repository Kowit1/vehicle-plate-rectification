# Vehicle Plate Rectification for LPR

Tier 3 web application for **CP461: Introduction to Computer Vision**. It rectifies angled license plates and exposes the complete feature-matching pipeline in the browser.

## Live pipeline

- Automatic plate proposal using grayscale, histogram equalization, Canny edges, morphology and quadrilateral scoring
- Manual four-corner fallback for difficult real-world images
- ORB keypoint and binary descriptor extraction
- KNN Hamming descriptor matching
- Lowe-style ratio test
- RANSAC outlier rejection
- 3 × 3 planar homography and perspective warping
- OCR-ready equalization, denoising and adaptive thresholding
- Built-in reproducible demo and failure handling
- Client-side processing: uploaded images never leave the browser

The CP461 brief accepts **SIFT, SURF or ORB**. This Vercel-native version uses ORB because it is available in the official browser build of OpenCV.js and does not require a Python server.

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

### Automatic rectification

Upload one vehicle/CCTV image. The app proposes a plate quadrilateral and estimates a perspective transform from its four corners. If automatic detection fails, choose **Select 4 corners manually**, then click the visible corners in this order: top-left, top-right, bottom-right, bottom-left.

### Feature Matching Lab

Upload an angled query and a frontal reference image of the **same physical plate**. ORB finds local descriptors, KNN produces two candidate matches per descriptor, the ratio test rejects ambiguous matches, and RANSAC estimates the homography from geometrically consistent inliers.

## Honest limitations

- Feature matching cannot use a blank generic plate template because different plate numbers do not share enough local features. It requires the same physical plate.
- Automatic contour detection may fail on tiny, dark, blurred, borderless or heavily occluded plates.
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
