import { describe, expect, it } from 'vitest'
import { orderPoints, plateAspect, targetSize } from '../src/geometry.js'

describe('geometry helpers', () => {
  it('orders four shuffled corners clockwise from top-left', () => {
    const points = [{ x: 95, y: 85 }, { x: 10, y: 10 }, { x: 12, y: 88 }, { x: 98, y: 13 }]
    expect(orderPoints(points)).toEqual([
      { x: 10, y: 10 }, { x: 98, y: 13 }, { x: 95, y: 85 }, { x: 12, y: 88 },
    ])
  })

  it('calculates a useful target size and normalized plate aspect', () => {
    const points = [{ x: 0, y: 0 }, { x: 300, y: 0 }, { x: 300, y: 100 }, { x: 0, y: 100 }]
    expect(targetSize(points)).toEqual({ width: 300, height: 100 })
    expect(plateAspect(points)).toBe(3)
  })
})
