export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

export function orderPoints(points) {
  if (!Array.isArray(points) || points.length !== 4) {
    throw new Error('Exactly four corner points are required.')
  }
  const pts = points.map(({ x, y }) => ({ x: Number(x), y: Number(y) }))
  const bySum = [...pts].sort((a, b) => (a.x + a.y) - (b.x + b.y))
  const byDiff = [...pts].sort((a, b) => (a.y - a.x) - (b.y - b.x))
  return [bySum[0], byDiff[0], bySum[3], byDiff[3]]
}

export function targetSize(points) {
  const [tl, tr, br, bl] = orderPoints(points)
  return {
    width: Math.max(2, Math.round(Math.max(distance(br, bl), distance(tr, tl)))),
    height: Math.max(2, Math.round(Math.max(distance(tr, br), distance(tl, bl)))),
  }
}

export function plateAspect(points) {
  const { width, height } = targetSize(points)
  return Math.max(width, height) / Math.max(1, Math.min(width, height))
}
