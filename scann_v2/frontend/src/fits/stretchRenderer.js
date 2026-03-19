export function calculatePixelRange(pixels) {
  if (!pixels || pixels.length === 0) {
    return { min: 0, max: 1 }
  }

  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < pixels.length; i += 1) {
    const value = pixels[i]
    if (value < min) min = value
    if (value > max) max = value
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 }
  }

  if (min === max) {
    return { min, max: min + 1 }
  }

  return { min, max }
}

function clamp01(value) {
  if (value < 0) return 0
  if (value > 1) return 1
  return value
}

export function renderStretchToRgba(pixels, min, max, invert = false) {
  if (!pixels || pixels.length === 0) {
    return new Uint8ClampedArray()
  }

  const safeMin = Number.isFinite(min) ? min : 0
  const safeMax = Number.isFinite(max) && max > safeMin ? max : safeMin + 1
  const range = safeMax - safeMin

  const out = new Uint8ClampedArray(pixels.length * 4)
  for (let i = 0; i < pixels.length; i += 1) {
    const normalized = clamp01((pixels[i] - safeMin) / range)
    const scaled = Math.round(normalized * 255)
    const gray = invert ? 255 - scaled : scaled
    const base = i * 4
    out[base] = gray
    out[base + 1] = gray
    out[base + 2] = gray
    out[base + 3] = 255
  }

  return out
}
