export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

export function orderPoints(points) {
  if (!Array.isArray(points) || points.length !== 4) {
    throw new Error('Exactly four corner points are required.')
  }
  const pts = points.map(({ x, y }) => ({ x: Number(x), y: Number(y) }))
  if (pts.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) throw new Error('พิกัดมุมไม่ถูกต้อง')
  const center = { x: pts.reduce((s, p) => s + p.x, 0) / 4, y: pts.reduce((s, p) => s + p.y, 0) / 4 }
  pts.sort((a, b) => Math.atan2(a.y - center.y, a.x - center.x) - Math.atan2(b.y - center.y, b.x - center.x))
  let start = 0
  pts.forEach((p, i) => { if (p.x + p.y < pts[start].x + pts[start].y) start = i })
  const ordered = pts.slice(start).concat(pts.slice(0, start))
  for (let i = 0; i < 4; i++) {
    const a = ordered[i], b = ordered[(i + 1) % 4], c = ordered[(i + 2) % 4]
    if ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 1) {
      throw new Error('เลือกมุมป้าย 4 จุดที่ไม่ซ้ำและไม่อยู่บนเส้นตรงเดียวกัน')
    }
  }
  return ordered
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
