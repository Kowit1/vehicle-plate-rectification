export function drawReferencePlate(canvas) {
  canvas.width = 640
  canvas.height = 220
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = '#eef2f4'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.strokeStyle = '#18222b'
  ctx.lineWidth = 9
  ctx.strokeRect(7, 7, canvas.width - 14, canvas.height - 14)
  ctx.lineWidth = 2
  ctx.strokeStyle = '#6a747c'
  ctx.strokeRect(22, 22, canvas.width - 44, canvas.height - 44)
  ctx.fillStyle = '#151b20'
  ctx.font = '700 78px Arial, sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('3AB 7824', canvas.width / 2, 126)
  ctx.font = '600 24px Arial, sans-serif'
  ctx.fillText('BANGKOK', canvas.width / 2, 181)
  for (let x = 38; x < canvas.width - 20; x += 61) {
    ctx.beginPath()
    ctx.arc(x, 34, 4.5, 0, Math.PI * 2)
    ctx.fill()
  }
}

export function createDemoAssets(cv, sceneCanvas, queryCanvas, referenceCanvas) {
  drawReferencePlate(referenceCanvas)
  sceneCanvas.width = 1100
  sceneCanvas.height = 650
  const ctx = sceneCanvas.getContext('2d')
  const gradient = ctx.createLinearGradient(0, 0, 0, sceneCanvas.height)
  gradient.addColorStop(0, '#536171')
  gradient.addColorStop(1, '#1d2731')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, sceneCanvas.width, sceneCanvas.height)
  ctx.fillStyle = '#394550'
  ctx.beginPath()
  ctx.roundRect(80, 100, 940, 470, 64)
  ctx.fill()
  ctx.fillStyle = '#202a33'
  ctx.beginPath()
  ctx.roundRect(160, 170, 780, 290, 28)
  ctx.fill()
  ctx.fillStyle = '#7e2630'
  ctx.fillRect(102, 225, 84, 125)
  ctx.fillRect(914, 225, 84, 125)
  ctx.fillStyle = '#111820'
  ctx.beginPath(); ctx.arc(230, 560, 94, 0, Math.PI * 2); ctx.fill()
  ctx.beginPath(); ctx.arc(870, 560, 94, 0, Math.PI * 2); ctx.fill()

  const plate = cv.imread(referenceCanvas)
  const scene = cv.imread(sceneCanvas)
  const warped = cv.Mat.zeros(scene.rows, scene.cols, scene.type())
  const maskInput = new cv.Mat(referenceCanvas.height, referenceCanvas.width, cv.CV_8UC1, new cv.Scalar(255))
  const maskWarped = cv.Mat.zeros(scene.rows, scene.cols, cv.CV_8UC1)
  const source = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, 639, 0, 639, 219, 0, 219])
  const destination = cv.matFromArray(4, 1, cv.CV_32FC2, [290, 385, 828, 342, 796, 523, 263, 548])
  const matrix = cv.getPerspectiveTransform(source, destination)
  try {
    cv.warpPerspective(plate, warped, matrix, new cv.Size(scene.cols, scene.rows))
    cv.warpPerspective(maskInput, maskWarped, matrix, new cv.Size(scene.cols, scene.rows))
    warped.copyTo(scene, maskWarped)
    cv.imshow(sceneCanvas, scene)

    const rect = new cv.Rect(235, 314, 625, 270)
    const roi = scene.roi(rect)
    queryCanvas.width = rect.width
    queryCanvas.height = rect.height
    cv.imshow(queryCanvas, roi)
    roi.delete()
  } finally {
    ;[plate, scene, warped, maskInput, maskWarped, source, destination, matrix].forEach((item) => item.delete())
  }
}
