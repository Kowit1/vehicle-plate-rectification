import { orderPoints, plateAspect, targetSize } from './geometry.js'

function safeDelete(...items) {
  for (const item of items) {
    if (item && typeof item.delete === 'function') item.delete()
  }
}

function imageCoverage(points, width, height) {
  let area = 0
  for (let i = 0; i < points.length; i += 1) {
    const current = points[i]
    const next = points[(i + 1) % points.length]
    area += current.x * next.y - next.x * current.y
  }
  return Math.abs(area / 2) / (width * height)
}

function boundingBox(points) {
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  }
}

function quadArea(points) {
  let area = 0
  for (let i = 0; i < points.length; i += 1) {
    const next = points[(i + 1) % points.length]
    area += points[i].x * next.y - next.x * points[i].y
  }
  return Math.abs(area / 2)
}

export function scorePlateCandidate(points, width, height, edgeDensity = 0.12) {
  const aspect = plateAspect(points)
  const coverage = imageCoverage(points, width, height)
  const box = boundingBox(points)
  const rectangularity = quadArea(points) / Math.max(1, box.width * box.height)
  const centerX = box.x + box.width / 2
  const centerY = box.y + box.height / 2
  const horizontalScore = Math.max(0, 1 - Math.abs(centerX / width - 0.5) * 1.3)
  const verticalScore = Math.max(0, 1 - Math.abs(centerY / height - 0.66) * 1.1)
  const aspectScore = Math.exp(-((Math.log(aspect) - Math.log(2.8)) ** 2) / 0.42)
  const sizeScore = Math.min(coverage / 0.025, 1)
  const textureScore = Math.max(0, 1 - Math.abs(edgeDensity - 0.15) / 0.2)
  const score = (
    aspectScore * 0.34
    + sizeScore * 0.2
    + rectangularity * 0.14
    + textureScore * 0.18
    + horizontalScore * 0.08
    + verticalScore * 0.06
  )
  return { aspect, coverage, rectangularity, score }
}

function pointsFromApprox(approx) {
  const raw = approx.data32S
  return orderPoints([
    { x: raw[0], y: raw[1] },
    { x: raw[2], y: raw[3] },
    { x: raw[4], y: raw[5] },
    { x: raw[6], y: raw[7] },
  ])
}

function isDuplicate(candidate, candidates) {
  const a = boundingBox(candidate.points)
  return candidates.some((item) => {
    const b = boundingBox(item.points)
    const left = Math.max(a.x, b.x)
    const top = Math.max(a.y, b.y)
    const right = Math.min(a.x + a.width, b.x + b.width)
    const bottom = Math.min(a.y + a.height, b.y + b.height)
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top)
    const union = a.width * a.height + b.width * b.height - intersection
    return union > 0 && intersection / union > 0.72
  })
}

function edgeDensityInQuad(cv, edges, points) {
  const box = boundingBox(points)
  const x = Math.max(0, Math.floor(box.x))
  const y = Math.max(0, Math.floor(box.y))
  const width = Math.min(edges.cols - x, Math.max(1, Math.ceil(box.width)))
  const height = Math.min(edges.rows - y, Math.max(1, Math.ceil(box.height)))
  if (width <= 1 || height <= 1) return 0
  const roi = edges.roi(new cv.Rect(x, y, width, height))
  try {
    return cv.countNonZero(roi) / (width * height)
  } finally {
    roi.delete()
  }
}

export function detectPlate(cv, sourceCanvas, debugCanvas) {
  const src = cv.imread(sourceCanvas)
  const gray = new cv.Mat()
  const blurred = new cv.Mat()
  const edges = new cv.Mat()
  const edgesSoft = new cv.Mat()
  const gradient = new cv.Mat()
  const gradient8 = new cv.Mat()
  const gradientMask = new cv.Mat()
  const adaptive = new cv.Mat()
  const combined = new cv.Mat()
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(9, 3))
  const wideKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(19, 5))
  const candidates = []

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
    cv.equalizeHist(gray, gray)
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT)
    cv.Canny(blurred, edges, 55, 165)
    cv.Canny(blurred, edgesSoft, 28, 105)
    cv.Sobel(blurred, gradient, cv.CV_16S, 1, 0, 3)
    cv.convertScaleAbs(gradient, gradient8)
    cv.threshold(gradient8, gradientMask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.adaptiveThreshold(blurred, adaptive, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 7)
    cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2)
    cv.morphologyEx(edgesSoft, edgesSoft, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2)
    cv.morphologyEx(gradientMask, gradientMask, cv.MORPH_CLOSE, wideKernel, new cv.Point(-1, -1), 2)
    cv.morphologyEx(adaptive, adaptive, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 1)
    cv.bitwise_or(edges, edgesSoft, combined)
    cv.bitwise_or(combined, gradientMask, combined)

    const masks = [edges, edgesSoft, gradientMask, adaptive]
    for (const mask of masks) {
      const contours = new cv.MatVector()
      const hierarchy = new cv.Mat()
      try {
        cv.findContours(mask, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for (let i = 0; i < contours.size(); i += 1) {
          const contour = contours.get(i)
          const perimeter = cv.arcLength(contour, true)
          if (perimeter < Math.min(src.cols, src.rows) * 0.06) {
            contour.delete()
            continue
          }
          for (const epsilon of [0.018, 0.026, 0.038, 0.055, 0.075]) {
            const approx = new cv.Mat()
            cv.approxPolyDP(contour, approx, epsilon * perimeter, true)
            if (approx.rows === 4 && cv.isContourConvex(approx)) {
              const points = pointsFromApprox(approx)
              const density = edgeDensityInQuad(cv, edgesSoft, points)
              const metrics = scorePlateCandidate(points, src.cols, src.rows, density)
              if (
                metrics.coverage >= 0.00065
                && metrics.coverage <= 0.38
                && metrics.aspect >= 1.15
                && metrics.aspect <= 7.8
                && metrics.rectangularity >= 0.32
              ) {
                const candidate = { points, ...metrics }
                if (!isDuplicate(candidate, candidates)) candidates.push(candidate)
              }
            }
            approx.delete()
          }

          // Textured plates do not always preserve a clean outer border. A bounding
          // box around a horizontally-connected character region gives the user a
          // useful editable proposal instead of failing with no result at all.
          if (mask === gradientMask || mask === adaptive) {
            const rect = cv.boundingRect(contour)
            const points = orderPoints([
              { x: rect.x, y: rect.y },
              { x: rect.x + rect.width, y: rect.y },
              { x: rect.x + rect.width, y: rect.y + rect.height },
              { x: rect.x, y: rect.y + rect.height },
            ])
            const density = edgeDensityInQuad(cv, edgesSoft, points)
            const metrics = scorePlateCandidate(points, src.cols, src.rows, density)
            if (
              metrics.coverage >= 0.00065
              && metrics.coverage <= 0.2
              && metrics.aspect >= 1.45
              && metrics.aspect <= 7.8
            ) {
              const candidate = { points, ...metrics, score: metrics.score * 0.88 }
              if (!isDuplicate(candidate, candidates)) candidates.push(candidate)
            }
          }
          contour.delete()
        }
      } finally {
        safeDelete(contours, hierarchy)
      }
    }

    cv.imshow(debugCanvas, combined)
    candidates.sort((a, b) => b.score - a.score)
    return { candidate: candidates[0] ?? null, candidates: candidates.slice(0, 12) }
  } finally {
    safeDelete(
      src,
      gray,
      blurred,
      edges,
      edgesSoft,
      gradient,
      gradient8,
      gradientMask,
      adaptive,
      combined,
      kernel,
      wideKernel,
    )
  }
}

export function rectifyPlate(cv, sourceCanvas, outputCanvas, inputPoints) {
  const points = orderPoints(inputPoints)
  let { width, height } = targetSize(points)
  const rotate = height > width
  if (rotate) [width, height] = [height, width]

  const src = cv.imread(sourceCanvas)
  const sourcePoints = cv.matFromArray(4, 1, cv.CV_32FC2, points.flatMap((p) => [p.x, p.y]))
  const destinationData = rotate
    ? [width - 1, 0, width - 1, height - 1, 0, height - 1, 0, 0]
    : [0, 0, width - 1, 0, width - 1, height - 1, 0, height - 1]
  const destinationPoints = cv.matFromArray(4, 1, cv.CV_32FC2, destinationData)
  const homography = cv.getPerspectiveTransform(sourcePoints, destinationPoints)
  const output = new cv.Mat()
  try {
    cv.warpPerspective(
      src,
      output,
      homography,
      new cv.Size(width, height),
      cv.INTER_LINEAR,
      cv.BORDER_REPLICATE,
      new cv.Scalar(),
    )
    cv.imshow(outputCanvas, output)
    return {
      width,
      height,
      aspect: Math.max(width, height) / Math.max(1, Math.min(width, height)),
      coverage: imageCoverage(points, src.cols, src.rows),
      homography: Array.from(homography.data64F),
    }
  } finally {
    safeDelete(src, sourcePoints, destinationPoints, homography, output)
  }
}

export function preprocessForOcr(cv, sourceCanvas, outputCanvas, mode = 'adaptive') {
  const src = cv.imread(sourceCanvas)
  const gray = new cv.Mat()
  const resized = new cv.Mat()
  const denoised = new cv.Mat()
  const output = new cv.Mat()
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
    const scale = Math.max(1, 640 / Math.max(1, gray.cols))
    cv.resize(gray, resized, new cv.Size(0, 0), scale, scale, cv.INTER_CUBIC)
    cv.equalizeHist(resized, resized)
    cv.bilateralFilter(resized, denoised, 7, 45, 45, cv.BORDER_DEFAULT)
    if (mode === 'adaptive') {
      cv.adaptiveThreshold(
        denoised,
        output,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        31,
        9,
      )
    } else {
      denoised.copyTo(output)
    }
    cv.imshow(outputCanvas, output)
  } finally {
    safeDelete(src, gray, resized, denoised, output)
  }
}

function drawMatchVisualization(queryCanvas, referenceCanvas, matches, targetCanvas) {
  const ctx = targetCanvas.getContext('2d')
  const maxWidth = 1280
  const naturalWidth = queryCanvas.width + referenceCanvas.width
  const scale = Math.min(1, maxWidth / naturalWidth)
  const qWidth = Math.round(queryCanvas.width * scale)
  const qHeight = Math.round(queryCanvas.height * scale)
  const rWidth = Math.round(referenceCanvas.width * scale)
  const rHeight = Math.round(referenceCanvas.height * scale)
  targetCanvas.width = qWidth + rWidth
  targetCanvas.height = Math.max(qHeight, rHeight)
  ctx.fillStyle = '#08131d'
  ctx.fillRect(0, 0, targetCanvas.width, targetCanvas.height)
  ctx.drawImage(queryCanvas, 0, 0, qWidth, qHeight)
  ctx.drawImage(referenceCanvas, qWidth, 0, rWidth, rHeight)

  ctx.lineWidth = 1.5
  for (const [index, match] of matches.entries()) {
    ctx.strokeStyle = `hsla(${(index * 47) % 360}, 82%, 64%, .72)`
    ctx.beginPath()
    ctx.moveTo(match.query.x * scale, match.query.y * scale)
    ctx.lineTo(qWidth + match.reference.x * scale, match.reference.y * scale)
    ctx.stroke()
    ctx.fillStyle = ctx.strokeStyle
    ctx.beginPath()
    ctx.arc(match.query.x * scale, match.query.y * scale, 2.4, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(qWidth + match.reference.x * scale, match.reference.y * scale, 2.4, 0, Math.PI * 2)
    ctx.fill()
  }
}

export function matchAndRectify(
  cv,
  queryCanvas,
  referenceCanvas,
  outputCanvas,
  matchesCanvas,
  ratioThreshold = 0.75,
  ransacThreshold = 4,
) {
  const query = cv.imread(queryCanvas)
  const reference = cv.imread(referenceCanvas)
  const queryGray = new cv.Mat()
  const referenceGray = new cv.Mat()
  const queryKeypoints = new cv.KeyPointVector()
  const referenceKeypoints = new cv.KeyPointVector()
  const queryDescriptors = new cv.Mat()
  const referenceDescriptors = new cv.Mat()
  const emptyMask = new cv.Mat()
  const matchPairs = new cv.DMatchVectorVector()
  const orb = new cv.ORB()
  const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false)

  let sourcePoints
  let destinationPoints
  let inlierMask
  let homography
  let output

  try {
    orb.setMaxFeatures(2500)
    orb.setFastThreshold(10)
    cv.cvtColor(query, queryGray, cv.COLOR_RGBA2GRAY)
    cv.cvtColor(reference, referenceGray, cv.COLOR_RGBA2GRAY)
    orb.detectAndCompute(queryGray, emptyMask, queryKeypoints, queryDescriptors)
    orb.detectAndCompute(referenceGray, emptyMask, referenceKeypoints, referenceDescriptors)
    if (queryDescriptors.empty() || referenceDescriptors.empty()) {
      throw new Error('Not enough visual detail was found in one of the images.')
    }

    matcher.knnMatch(queryDescriptors, referenceDescriptors, matchPairs, 2)
    const good = []
    for (let i = 0; i < matchPairs.size(); i += 1) {
      const pair = matchPairs.get(i)
      if (pair.size() >= 2) {
        const best = pair.get(0)
        const second = pair.get(1)
        if (best.distance < ratioThreshold * second.distance) {
          const queryPoint = queryKeypoints.get(best.queryIdx).pt
          const referencePoint = referenceKeypoints.get(best.trainIdx).pt
          good.push({
            query: { x: queryPoint.x, y: queryPoint.y },
            reference: { x: referencePoint.x, y: referencePoint.y },
          })
        }
      }
      safeDelete(pair)
    }
    if (good.length < 4) {
      throw new Error(`Only ${good.length} reliable matches were found; at least 4 are required.`)
    }

    sourcePoints = cv.matFromArray(good.length, 1, cv.CV_32FC2, good.flatMap((m) => [m.query.x, m.query.y]))
    destinationPoints = cv.matFromArray(
      good.length,
      1,
      cv.CV_32FC2,
      good.flatMap((m) => [m.reference.x, m.reference.y]),
    )
    inlierMask = new cv.Mat()
    homography = cv.findHomography(
      sourcePoints,
      destinationPoints,
      cv.RANSAC,
      ransacThreshold,
      inlierMask,
    )
    if (homography.empty()) throw new Error('RANSAC could not estimate a stable homography.')

    const inlierMatches = good.filter((_, index) => inlierMask.data[index] === 1)
    if (inlierMatches.length < 4) {
      throw new Error('Fewer than 4 geometrically consistent matches remained after RANSAC.')
    }
    output = new cv.Mat()
    cv.warpPerspective(
      query,
      output,
      homography,
      new cv.Size(reference.cols, reference.rows),
      cv.INTER_LINEAR,
      cv.BORDER_CONSTANT,
      new cv.Scalar(),
    )
    cv.imshow(outputCanvas, output)
    drawMatchVisualization(queryCanvas, referenceCanvas, inlierMatches, matchesCanvas)

    return {
      queryKeypoints: queryKeypoints.size(),
      referenceKeypoints: referenceKeypoints.size(),
      rawMatches: matchPairs.size(),
      goodMatches: good.length,
      inliers: inlierMatches.length,
      inlierRatio: inlierMatches.length / good.length,
      homography: Array.from(homography.data64F),
    }
  } finally {
    safeDelete(
      query,
      reference,
      queryGray,
      referenceGray,
      queryKeypoints,
      referenceKeypoints,
      queryDescriptors,
      referenceDescriptors,
      emptyMask,
      matchPairs,
      orb,
      matcher,
      sourcePoints,
      destinationPoints,
      inlierMask,
      homography,
      output,
    )
  }
}
