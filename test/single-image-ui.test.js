// @vitest-environment happy-dom
import { expect, it, vi } from 'vitest'

const runtime = vi.hoisted(() => {
  let ready
  return { promise: new Promise(resolve => { ready = resolve }), resolve: value => ready(value) }
})
vi.mock('../src/opencv.js', () => ({ loadOpenCv: () => runtime.promise }))
vi.mock('../src/demo.js', () => ({ createDemoAssets: (_cv, scene, query, reference) => {
  for (const canvas of [scene, query, reference]) { canvas.width = 1100; canvas.height = 650 }
} }))
vi.mock('../src/pipeline.js', () => ({
  detectPlate: vi.fn(() => ({ candidate: true, candidates: [{ points: [{x:20,y:20},{x:300,y:20},{x:300,y:120},{x:20,y:120}], score:0.8 }] })),
  rectifyPlate: (_cv, _source, output) => { output.width=280; output.height=100; return { aspect:2.8, coverage:0.1 } },
  preprocessForOcr: (_cv, _source, output) => { output.width=640; output.height=220 },
  matchAndRectify: vi.fn(),
}))

it('keeps a single uploaded image through initialization, displays three outputs, and clears failed/stale results', async () => {
  const context = { drawImage: vi.fn(), beginPath(){}, moveTo(){}, lineTo(){}, closePath(){}, stroke(){}, arc(){}, fill(){}, fillText(){} }
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context)
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/png;base64,test')
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width:800, height:500, close(){} })))
  document.body.innerHTML = '<div id="app"></div>'
  await import('../src/main.js')
  const $ = selector => document.querySelector(selector)
  expect($('#automatic').classList.contains('active')).toBe(true)
  expect($('#matching').classList.contains('active')).toBe(false)
  expect($('#auto-demo').disabled).toBe(true)
  const input = $('#auto-upload')
  Object.defineProperty(input, 'files', { configurable:true, value:[new File(['image'], 'car.png', {type:'image/png'})] })
  input.dispatchEvent(new Event('change'))
  await vi.waitFor(() => expect($('#auto-source').width).toBe(800))
  runtime.resolve({})
  await vi.waitFor(() => expect($('#results-section').hidden).toBe(false))
  expect($('#auto-source').width).toBe(800)
  expect($('#plate-before').width).toBe(280)
  expect($('#rectified-output').width).toBe(280)
  expect($('#ocr-output').width).toBe(640)
  expect($('#download-rectified').hasAttribute('href')).toBe(true)
  const { detectPlate } = await import('../src/pipeline.js')
  detectPlate.mockReturnValue({ candidate:null, candidates:[] })
  $('#auto-detect').click()
  await vi.waitFor(() => expect($('#auto-feedback').classList.contains('error')).toBe(true))
  expect($('#results-section').hidden).toBe(true)
  expect($('#download-rectified').hasAttribute('href')).toBe(false)
  expect($('#replace-vehicle').disabled).toBe(false)
  $('#manual-corners').click()
  expect($('#rectify-manual').disabled).toBe(true)
})
