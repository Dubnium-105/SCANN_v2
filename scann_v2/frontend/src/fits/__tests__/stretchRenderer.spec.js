import { calculatePixelRange, renderStretchToRgba } from '../stretchRenderer'

describe('stretchRenderer', () => {
  it('maps float pixels to clamped 0-255 grayscale RGBA', () => {
    const pixels = new Float32Array([-1, 0, 0.5, 2])
    const rgba = renderStretchToRgba(pixels, 0, 1, false)

    expect(Array.from(rgba)).toEqual([
      0, 0, 0, 255,
      0, 0, 0, 255,
      128, 128, 128, 255,
      255, 255, 255, 255,
    ])
  })

  it('applies inversion as 255 - value', () => {
    const pixels = new Float32Array([0, 0.25, 1])
    const rgba = renderStretchToRgba(pixels, 0, 1, true)

    expect(Array.from(rgba)).toEqual([
      255, 255, 255, 255,
      191, 191, 191, 255,
      0, 0, 0, 255,
    ])
  })

  it('calculates min/max range from pixel array', () => {
    const range = calculatePixelRange(new Float32Array([3, -5, 8, 2]))
    expect(range).toEqual({ min: -5, max: 8 })
  })
})
