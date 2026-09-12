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

export function detectPlate(cv, sourceCanvas, debugCanvas) {
  const src = cv.imread(sourceCanvas)
  const gray = new cv.Mat()
  const blurred = new cv.Mat()
  const edges = new cv.Mat()
  const closed = new cv.Mat()
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 3))
  const candidates = []

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
    cv.equalizeHist(gray, gray)
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT)
    cv.Canny(blurred, edges, 55, 165)
    cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2)
    cv.findContours(closed, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    const imageArea = src.cols * src.rows
    for (let i = 0; i < contours.size(); i += 1) {
      const contour = contours.get(i)
      const perimeter = cv.arcLength(contour, true)
      const approx = new cv.Mat()
      cv.approxPolyDP(contour, approx, 0.025 * perimeter, true)

      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const area = Math.abs(cv.contourArea(approx, false))
        const coverage = area / imageArea
        const raw = approx.data32S
        const points = orderPoints([
          { x: raw[0], y: raw[1] },
          { x: raw[2], y: raw[3] },
          { x: raw[4], y: raw[5] },
          { x: raw[6], y: raw[7] },
        ])
        const aspect = plateAspect(points)
        if (coverage >= 0.0015 && coverage <= 0.42 && aspect >= 1.45 && aspect <= 6.8) {
          const aspectScore = Math.exp(-((aspect - 2.8) ** 2) / 4.8)
          const sizeScore = Math.min(coverage / 0.035, 1)
          candidates.push({
            points,
            aspect,
            coverage,
            score: 0.68 * aspectScore + 0.32 * sizeScore,
          })
        }
      }
      safeDelete(approx, contour)
    }

    cv.imshow(debugCanvas, closed)
    candidates.sort((a, b) => b.score - a.score)
    return { candidate: candidates[0] ?? null, candidates: candidates.slice(0, 8) }
  } finally {
    safeDelete(src, gray, blurred, edges, closed, contours, hierarchy, kernel)
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
