import { beforeAll, describe, expect, it } from 'vitest'
import { createRequire } from 'node:module'
import { createCanvas, ImageData } from '@napi-rs/canvas'
import { createDemoAssets } from '../src/demo.js'
import { detectPlate, rectifyPlate, preprocessForOcr } from '../src/pipeline.js'

let cv
beforeAll(async () => {
  cv = await createRequire(import.meta.url)('@techstark/opencv-js')
  // Native canvas adapter: all detection, geometry and image processing use real OpenCV.
  cv.imread = canvas => cv.matFromImageData(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height))
  cv.imshow = (canvas, mat) => {
    const rgba = new cv.Mat()
    try {
      if (mat.channels() === 1) cv.cvtColor(mat, rgba, cv.COLOR_GRAY2RGBA)
      else mat.copyTo(rgba)
      canvas.width = mat.cols; canvas.height = mat.rows
      canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba.data), mat.cols, mat.rows), 0, 0)
    } finally { rgba.delete() }
  }
})

describe('single vehicle image rectification', () => {
  it('detects the plate in a vehicle image, warps it and produces grayscale output', () => {
    const scene = createCanvas(1, 1), query = createCanvas(1, 1), reference = createCanvas(1, 1)
    createDemoAssets(cv, scene, query, reference)
    const result = detectPlate(cv, scene, createCanvas(1, 1))
    expect(result.candidate).not.toBeNull()
    const points = result.candidate.points
    const expected = [{ x: 290, y: 385 }, { x: 828, y: 342 }, { x: 796, y: 523 }, { x: 263, y: 548 }]
    const averageError = points.reduce((sum, p, i) => sum + Math.hypot(p.x - expected[i].x, p.y - expected[i].y), 0) / 4
    expect(averageError).toBeLessThan(40)
    const output = createCanvas(1, 1), gray = createCanvas(1, 1)
    const warp = rectifyPlate(cv, scene, output, points)
    expect(warp.width).toBeGreaterThan(400)
    expect(warp.homography.every(Number.isFinite)).toBe(true)
    // H must map each detected corner to its corresponding output corner.
    const h = warp.homography
    const targets = [[0, 0], [warp.width - 1, 0], [warp.width - 1, warp.height - 1], [0, warp.height - 1]]
    points.forEach((p, i) => {
      const z = h[6] * p.x + h[7] * p.y + h[8]
      expect((h[0] * p.x + h[1] * p.y + h[2]) / z).toBeCloseTo(targets[i][0], 2)
      expect((h[3] * p.x + h[4] * p.y + h[5]) / z).toBeCloseTo(targets[i][1], 2)
    })
    preprocessForOcr(cv, output, gray, 'grayscale')
    const pixels = gray.getContext('2d').getImageData(0, 0, gray.width, gray.height).data
    expect(gray.width).toBeGreaterThanOrEqual(640)
    expect(pixels.some((value, i) => i % 4 === 0 && value > 160)).toBe(true)
    for (let i = 0; i < pixels.length; i += 4) { expect(pixels[i]).toBe(pixels[i + 1]); expect(pixels[i]).toBe(pixels[i + 2]) }
  })

  it('returns no plate for a blank image', () => {
    const canvas = createCanvas(640, 480)
    canvas.getContext('2d').fillRect(0, 0, 640, 480)
    expect(detectPlate(cv, canvas, createCanvas(1, 1)).candidate).toBeNull()
  })
})
