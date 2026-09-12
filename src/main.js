import './style.css'
import { createDemoAssets } from './demo.js'
import { orderPoints } from './geometry.js'
import { loadOpenCv } from './opencv.js'
import { detectPlate, matchAndRectify, preprocessForOcr, rectifyPlate } from './pipeline.js'

document.querySelector('#app').innerHTML = `
  <header class="site-header">
    <a class="brand" href="#top" aria-label="PlateRect CV home">
      <span class="brand-mark">PR</span>
      <span><strong>PlateRect</strong><small>Computer Vision Lab</small></span>
    </a>
    <div class="header-meta"><span>CP461</span><span class="dot"></span><span>Tier 3 Web App</span></div>
  </header>

  <main id="top">
    <section class="hero">
      <div>
        <p class="eyebrow">Vehicle plate rectification for LPR</p>
        <h1>Straighten angled plates.<br><em>Expose the geometry.</em></h1>
        <p class="hero-copy">Upload a vehicle photo, let the app find the plate, fine-tune its four corners, and export a straightened OCR-ready image. Processing stays on your device.</p>
      </div>
      <div class="hero-diagram" aria-label="Pipeline overview">
        <span>Input</span><i>→</i><span>Features</span><i>→</i><span>RANSAC</span><i>→</i><span>Rectified</span>
      </div>
    </section>

    <div id="runtime-status" class="runtime-status loading"><span class="pulse"></span><strong>Loading OpenCV.js</strong><small>Initializing the vision engine in your browser…</small></div>

    <nav class="tabs" aria-label="Application modes">
      <button class="tab active" data-tab="automatic">01 · Automatic</button>
      <button class="tab" data-tab="matching">02 · Feature Matching</button>
      <button class="tab" data-tab="method">03 · Method</button>
    </nav>

    <section id="automatic" class="panel active">
      <div class="section-head">
        <div><p class="step">MODE 01</p><h2>Automatic plate rectification</h2><p>Find a plate-shaped quadrilateral, adjust the corners if needed, then warp it to a frontal view.</p></div>
        <div class="actions">
          <label class="button secondary">Upload vehicle<input id="auto-upload" type="file" accept="image/*" hidden></label>
          <button id="auto-demo" class="button ghost">Reset demo</button>
          <button id="auto-detect" class="button primary" disabled>Detect plate</button>
        </div>
      </div>

      <div class="workspace-grid">
        <article class="card image-card span-2" id="image-drop-zone">
          <div class="card-title"><span>Input image</span><small id="image-meta">Drop an image here or use Upload vehicle</small></div>
          <canvas id="auto-source" hidden></canvas>
          <canvas id="auto-preview" class="main-canvas" aria-label="Vehicle input image"></canvas>
          <div class="corner-toolbar">
            <button id="manual-corners" class="text-button">Select 4 corners manually</button>
            <button id="clear-corners" class="text-button" disabled>Clear points</button>
            <span class="candidate-nav" id="candidate-nav" hidden>
              <button id="previous-candidate" class="text-button" aria-label="Previous candidate">← Previous</button>
              <strong id="candidate-position">1 / 1</strong>
              <button id="next-candidate" class="text-button" aria-label="Next candidate">Next →</button>
            </span>
            <button id="rectify-manual" class="text-button accent" disabled>Rectify selected area</button>
          </div>
        </article>
        <article class="card">
          <div class="card-title"><span>Rectified plate</span><small>Perspective transform</small></div>
          <div class="canvas-well"><canvas id="rectified-output"></canvas><p class="placeholder">Run detection to see the result</p></div>
          <a id="download-rectified" class="download disabled" download="rectified-plate.png">Download PNG ↓</a>
        </article>
        <article class="card">
          <div class="card-title"><span>OCR-ready</span><small>Equalize · Denoise · Threshold</small></div>
          <div class="canvas-well"><canvas id="ocr-output"></canvas><p class="placeholder">Preprocessed output appears here</p></div>
          <a id="download-ocr" class="download disabled" download="ocr-ready-plate.png">Download PNG ↓</a>
        </article>
      </div>

      <div id="auto-feedback" class="feedback neutral">Ready for an image.</div>
      <div class="metric-grid" id="auto-metrics">
        <div><small>Candidate score</small><strong>—</strong></div><div><small>Aspect ratio</small><strong>—</strong></div><div><small>Image coverage</small><strong>—</strong></div><div><small>Candidates</small><strong>—</strong></div>
      </div>
      <details class="diagnostics"><summary>Detection diagnostics</summary><div><p>The edge mask reveals the regions inspected by the contour-based plate proposal stage.</p><canvas id="debug-output"></canvas></div></details>
    </section>

    <section id="matching" class="panel">
      <div class="section-head">
        <div><p class="step">MODE 02</p><h2>ORB feature matching lab</h2><p>Align an angled view to a frontal reference of the <strong>same physical plate</strong>.</p></div>
        <button id="run-matching" class="button primary" disabled>Run ORB + RANSAC</button>
      </div>
      <div class="two-up">
        <article class="card">
          <div class="card-title"><span>Query</span><label class="mini-upload">Replace image<input id="query-upload" type="file" accept="image/*" hidden></label></div>
          <canvas id="query-canvas"></canvas><small>Angled CCTV view</small>
        </article>
        <article class="card">
          <div class="card-title"><span>Reference</span><label class="mini-upload">Replace image<input id="reference-upload" type="file" accept="image/*" hidden></label></div>
          <canvas id="reference-canvas"></canvas><small>Front-facing view of the same plate</small>
        </article>
      </div>
      <div class="control-strip">
        <label><span>Lowe ratio <output id="ratio-value">0.75</output></span><input id="ratio" type="range" min="0.55" max="0.90" value="0.75" step="0.01"></label>
        <label><span>RANSAC threshold <output id="ransac-value">4.0 px</output></span><input id="ransac" type="range" min="1" max="10" value="4" step="0.5"></label>
        <div class="method-pill"><small>Detector / matcher</small><strong>ORB · Hamming KNN</strong></div>
      </div>
      <div id="match-feedback" class="feedback neutral">Demo pair is ready.</div>
      <div class="metric-grid five" id="match-metrics">
        <div><small>Query keypoints</small><strong>—</strong></div><div><small>Reference keypoints</small><strong>—</strong></div><div><small>Good matches</small><strong>—</strong></div><div><small>RANSAC inliers</small><strong>—</strong></div><div><small>Inlier ratio</small><strong>—</strong></div>
      </div>
      <div class="two-up outputs">
        <article class="card"><div class="card-title"><span>Geometric inliers</span><small>Outliers rejected by RANSAC</small></div><div class="canvas-well large"><canvas id="matches-output"></canvas><p class="placeholder">Run matching to visualize inliers</p></div></article>
        <article class="card"><div class="card-title"><span>Homography output</span><small>Aligned to reference plane</small></div><div class="canvas-well large"><canvas id="match-output"></canvas><p class="placeholder">Aligned plate appears here</p></div><pre id="homography-matrix">Homography matrix: —</pre></article>
      </div>
    </section>

    <section id="method" class="panel">
      <div class="section-head"><div><p class="step">MODE 03</p><h2>How the pipeline works</h2><p>Every stage maps directly to the CP461 evaluation rubric.</p></div></div>
      <div class="method-grid">
        <article><b>01</b><h3>Plate proposal</h3><p>Grayscale conversion, histogram equalization, Canny edges and morphological closing produce rectangular candidates.</p></article>
        <article><b>02</b><h3>ORB descriptors</h3><p>Oriented FAST keypoints and rotated BRIEF descriptors capture distinctive local patterns efficiently in the browser.</p></article>
        <article><b>03</b><h3>Ratio test</h3><p>K-nearest-neighbor Hamming matching keeps a pair only when the best match is sufficiently better than the second-best.</p></article>
        <article><b>04</b><h3>RANSAC</h3><p>Repeated robust estimation separates geometrically consistent inliers from incorrect descriptor matches.</p></article>
        <article><b>05</b><h3>Homography</h3><p>A 3 × 3 projective transformation maps the angled plate plane onto a frontal rectangular plane.</p></article>
        <article><b>06</b><h3>OCR preparation</h3><p>Upscaling, equalization, bilateral denoising and adaptive thresholding improve character contrast.</p></article>
      </div>
      <aside class="limitation"><strong>Known limitation</strong><p>Automatic contour detection can fail on tiny, dark, blurred or occluded plates. Feature matching requires a frontal reference of the same plate. Manual four-corner selection provides a safe fallback during real-world demonstrations.</p></aside>
    </section>
  </main>
  <footer><span>Vehicle Plate Rectification for LPR</span><span>Client-side OpenCV.js · No image uploads</span></footer>
`

const state = { cv: null, points: [], manual: false, candidates: [], candidateIndex: 0, draggingPoint: -1 }
const $ = (selector) => document.querySelector(selector)

function setFeedback(selector, message, type = 'neutral') {
  const element = $(selector)
  element.className = `feedback ${type}`
  element.textContent = message
}

function fillMetrics(selector, values) {
  const elements = document.querySelectorAll(`${selector} strong`)
  values.forEach((value, index) => { if (elements[index]) elements[index].textContent = value })
}

function copyCanvas(source, target) {
  target.width = source.width
  target.height = source.height
  target.getContext('2d').drawImage(source, 0, 0)
}

function renderCorners() {
  const source = $('#auto-source')
  const preview = $('#auto-preview')
  copyCanvas(source, preview)
  $('#clear-corners').disabled = !state.points.length
  $('#rectify-manual').disabled = state.points.length !== 4
  preview.classList.toggle('editing', state.manual || state.points.length === 4)
  if (!state.points.length) return
  const ctx = preview.getContext('2d')
  const points = state.points.length === 4 ? orderPoints(state.points) : state.points
  ctx.lineWidth = Math.max(3, source.width / 360)
  ctx.strokeStyle = '#31e6a1'
  ctx.fillStyle = '#31e6a1'
  ctx.font = `700 ${Math.max(16, source.width / 45)}px Inter, sans-serif`
  if (points.length === 4) {
    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y)
    points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y))
    ctx.closePath(); ctx.stroke()
  }
  points.forEach((point, index) => {
    ctx.beginPath(); ctx.arc(point.x, point.y, Math.max(7, source.width / 110), 0, Math.PI * 2); ctx.fill()
    ctx.fillText(points.length === 4 ? ['TL', 'TR', 'BR', 'BL'][index] : String(index + 1), point.x + 12, point.y - 12)
  })
}

function updateCandidateNavigation() {
  const navigation = $('#candidate-nav')
  navigation.hidden = state.candidates.length < 2
  $('#candidate-position').textContent = `${state.candidateIndex + 1} / ${state.candidates.length}`
}

function enableDownload(selector, canvas) {
  const link = $(selector)
  link.href = canvas.toDataURL('image/png')
  link.classList.remove('disabled')
}

function clearCanvasOutput(selector) {
  const canvas = $(selector)
  canvas.width = 0; canvas.height = 0
}

function applyRectification(points, candidateCount = 'Manual') {
  const result = rectifyPlate(state.cv, $('#auto-source'), $('#rectified-output'), points)
  preprocessForOcr(state.cv, $('#rectified-output'), $('#ocr-output'))
  document.querySelectorAll('#automatic .canvas-well .placeholder').forEach((p) => { p.hidden = true })
  enableDownload('#download-rectified', $('#rectified-output'))
  enableDownload('#download-ocr', $('#ocr-output'))
  fillMetrics('#auto-metrics', [
    candidateCount === 'Manual' ? 'Manual' : candidateCount.score.toFixed(2),
    `${result.aspect.toFixed(2)} : 1`,
    `${(result.coverage * 100).toFixed(2)}%`,
    candidateCount === 'Manual' ? 'Manual' : String(candidateCount.count),
  ])
  setFeedback('#auto-feedback', 'Plate rectified successfully. The OCR-ready image can now be downloaded.', 'success')
}

async function loadFileToCanvas(file, canvas, maxWidth = 1600) {
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, maxWidth / bitmap.width)
  canvas.width = Math.round(bitmap.width * scale)
  canvas.height = Math.round(bitmap.height * scale)
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close()
}

async function useVehicleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    setFeedback('#auto-feedback', 'Please choose a JPG, PNG or another image file.', 'error')
    return
  }
  await loadFileToCanvas(file, $('#auto-source'))
  $('#image-meta').textContent = `${file.name} · ${$('#auto-source').width} × ${$('#auto-source').height}px`
  state.points = []; state.manual = false; state.candidates = []; renderCorners(); updateCandidateNavigation()
  if (!state.cv) {
    setFeedback('#auto-feedback', 'Image loaded. Detection will be available when OpenCV.js is ready.', 'working')
    return
  }
  setFeedback('#auto-feedback', 'Image loaded. Detecting the plate…', 'working')
  runAutomaticDetection()
}

function resetAutomaticDemo() {
  createDemoAssets(state.cv, $('#auto-source'), $('#query-canvas'), $('#reference-canvas'))
  state.points = []
  state.manual = false
  state.candidates = []
  state.candidateIndex = 0
  updateCandidateNavigation()
  $('#image-meta').textContent = 'Built-in sample · drag a green corner to fine-tune'
  renderCorners()
  clearCanvasOutput('#rectified-output'); clearCanvasOutput('#ocr-output')
  document.querySelectorAll('#automatic .canvas-well .placeholder').forEach((p) => { p.hidden = false })
  fillMetrics('#auto-metrics', ['—', '—', '—', '—'])
  setFeedback('#auto-feedback', 'Built-in angled vehicle demo loaded.', 'neutral')
}

document.querySelectorAll('.tab').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tab, .panel').forEach((element) => element.classList.remove('active'))
    button.classList.add('active')
    $(`#${button.dataset.tab}`).classList.add('active')
  })
})

$('#ratio').addEventListener('input', (event) => { $('#ratio-value').textContent = Number(event.target.value).toFixed(2) })
$('#ransac').addEventListener('input', (event) => { $('#ransac-value').textContent = `${Number(event.target.value).toFixed(1)} px` })

$('#auto-demo').addEventListener('click', resetAutomaticDemo)
$('#auto-upload').addEventListener('change', async (event) => {
  if (!event.target.files[0]) return
  await useVehicleFile(event.target.files[0])
  event.target.value = ''
})


const dropZone = $('#image-drop-zone')
for (const eventName of ['dragenter', 'dragover']) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault()
    dropZone.classList.add('dragging')
  })
}
for (const eventName of ['dragleave', 'drop']) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault()
    dropZone.classList.remove('dragging')
  })
}
dropZone.addEventListener('drop', (event) => useVehicleFile(event.dataTransfer.files[0]))

function selectCandidate(index) {
  if (!state.candidates.length) return
  state.candidateIndex = (index + state.candidates.length) % state.candidates.length
  const candidate = state.candidates[state.candidateIndex]
  state.points = candidate.points.map((point) => ({ ...point }))
  state.manual = false
  renderCorners()
  updateCandidateNavigation()
  applyRectification(state.points, { ...candidate, count: state.candidates.length })
  setFeedback('#auto-feedback', `Candidate ${state.candidateIndex + 1} selected. Drag any green corner if it needs adjustment.`, 'success')
}

function runAutomaticDetection() {
  if (!state.cv) {
    setFeedback('#auto-feedback', 'The vision engine is still loading. Please try again in a moment.', 'working')
    return
  }
  setFeedback('#auto-feedback', 'Searching for plate-shaped quadrilaterals…', 'working')
  try {
    const result = detectPlate(state.cv, $('#auto-source'), $('#debug-output'))
    if (!result.candidate) throw new Error('No reliable plate-shaped region was found.')
    state.candidates = result.candidates
    selectCandidate(0)
  } catch (error) {
    state.candidates = []
    updateCandidateNavigation()
    setFeedback('#auto-feedback', `${error.message} Use manual four-corner selection as a fallback.`, 'error')
  }
}

$('#auto-detect').addEventListener('click', runAutomaticDetection)
$('#previous-candidate').addEventListener('click', () => selectCandidate(state.candidateIndex - 1))
$('#next-candidate').addEventListener('click', () => selectCandidate(state.candidateIndex + 1))

$('#manual-corners').addEventListener('click', () => {
  state.points = []; state.manual = true; state.candidates = []; renderCorners(); updateCandidateNavigation()
  $('#corner-help').textContent = 'Click the 4 corners in any order'
  setFeedback('#auto-feedback', 'Manual mode: click the four visible plate corners. You can drag a point to correct it.', 'working')
})
$('#clear-corners').addEventListener('click', () => { state.points = []; state.manual = true; renderCorners() })
$('#auto-preview').addEventListener('click', (event) => {
  if (!state.manual || state.points.length >= 4) return
  const rect = event.currentTarget.getBoundingClientRect()
  state.points.push({
    x: (event.clientX - rect.left) * event.currentTarget.width / rect.width,
    y: (event.clientY - rect.top) * event.currentTarget.height / rect.height,
  })
  renderCorners()
})

function pointerPosition(event) {
  const rect = event.currentTarget.getBoundingClientRect()
  return {
    x: (event.clientX - rect.left) * event.currentTarget.width / rect.width,
    y: (event.clientY - rect.top) * event.currentTarget.height / rect.height,
  }
}

$('#auto-preview').addEventListener('pointerdown', (event) => {
  if (state.points.length !== 4) return
  const pointer = pointerPosition(event)
  const threshold = Math.max(28, event.currentTarget.width / 25)
  let closest = -1
  let closestDistance = Infinity
  state.points.forEach((point, index) => {
    const distance = Math.hypot(point.x - pointer.x, point.y - pointer.y)
    if (distance < closestDistance && distance < threshold) {
      closest = index
      closestDistance = distance
    }
  })
  if (closest >= 0) {
    state.draggingPoint = closest
    state.manual = true
    event.currentTarget.setPointerCapture(event.pointerId)
  }
})

$('#auto-preview').addEventListener('pointermove', (event) => {
  if (state.draggingPoint < 0) return
  state.points[state.draggingPoint] = pointerPosition(event)
  renderCorners()
})

function stopDragging(event) {
  if (state.draggingPoint < 0) return
  state.draggingPoint = -1
  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  setFeedback('#auto-feedback', 'Corners adjusted. Click “Rectify selected area” to apply the correction.', 'working')
}

$('#auto-preview').addEventListener('pointerup', stopDragging)
$('#auto-preview').addEventListener('pointercancel', stopDragging)
$('#rectify-manual').addEventListener('click', () => {
  try { applyRectification(state.points, 'Manual') } catch (error) { setFeedback('#auto-feedback', error.message, 'error') }
})

for (const [inputSelector, canvasSelector] of [['#query-upload', '#query-canvas'], ['#reference-upload', '#reference-canvas']]) {
  $(inputSelector).addEventListener('change', async (event) => {
    if (!event.target.files[0]) return
    await loadFileToCanvas(event.target.files[0], $(canvasSelector), 1100)
    setFeedback('#match-feedback', 'Image replaced. Run ORB + RANSAC to calculate a new alignment.', 'neutral')
  })
}

$('#run-matching').addEventListener('click', () => {
  setFeedback('#match-feedback', 'Extracting ORB descriptors and estimating robust geometry…', 'working')
  try {
    const result = matchAndRectify(
      state.cv,
      $('#query-canvas'),
      $('#reference-canvas'),
      $('#match-output'),
      $('#matches-output'),
      Number($('#ratio').value),
      Number($('#ransac').value),
    )
    document.querySelectorAll('#matching .canvas-well .placeholder').forEach((p) => { p.hidden = true })
    fillMetrics('#match-metrics', [result.queryKeypoints, result.referenceKeypoints, result.goodMatches, result.inliers, `${(result.inlierRatio * 100).toFixed(1)}%`])
    const rows = [0, 1, 2].map((row) => result.homography.slice(row * 3, row * 3 + 3).map((v) => v.toFixed(5)).join('   '))
    $('#homography-matrix').textContent = `Homography matrix\n${rows.join('\n')}`
    setFeedback('#match-feedback', 'Stable homography estimated. Only RANSAC inlier matches are shown.', 'success')
  } catch (error) {
    fillMetrics('#match-metrics', ['—', '—', '—', '—', '—'])
    setFeedback('#match-feedback', `${error.message} Try a less extreme angle or a higher ratio threshold.`, 'error')
  }
})

async function initialize() {
  try {
    state.cv = await loadOpenCv()
    resetAutomaticDemo()
    $('#auto-detect').disabled = false
    $('#run-matching').disabled = false
    const status = $('#runtime-status')
    status.className = 'runtime-status ready'
    status.innerHTML = '<span class="status-check">✓</span><strong>OpenCV.js ready</strong><small>All processing runs locally in this browser.</small>'
  } catch (error) {
    const status = $('#runtime-status')
    status.className = 'runtime-status failed'
    status.innerHTML = `<strong>Vision engine failed to load</strong><small>${error.message}</small>`
  }
}

initialize()
