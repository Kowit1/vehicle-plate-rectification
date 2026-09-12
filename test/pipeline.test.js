import { describe, expect, it } from 'vitest'
import { scorePlateCandidate } from '../src/pipeline.js'

describe('plate candidate scoring', () => {
  it('prefers a textured plate-like rectangle over a square', () => {
    const plate = scorePlateCandidate([
      { x: 300, y: 440 },
      { x: 700, y: 430 },
      { x: 710, y: 570 },
      { x: 290, y: 580 },
    ], 1000, 700, 0.15)
    const square = scorePlateCandidate([
      { x: 400, y: 400 },
      { x: 550, y: 400 },
      { x: 550, y: 550 },
      { x: 400, y: 550 },
    ], 1000, 700, 0.03)

    expect(plate.score).toBeGreaterThan(square.score)
    expect(plate.aspect).toBeGreaterThan(2.5)
  })

  it('accepts a perspective-skewed quadrilateral', () => {
    const candidate = scorePlateCandidate([
      { x: 240, y: 390 },
      { x: 760, y: 350 },
      { x: 730, y: 520 },
      { x: 260, y: 550 },
    ], 1000, 700, 0.13)

    expect(candidate.rectangularity).toBeGreaterThan(0.7)
    expect(candidate.score).toBeGreaterThan(0.7)
  })
})
