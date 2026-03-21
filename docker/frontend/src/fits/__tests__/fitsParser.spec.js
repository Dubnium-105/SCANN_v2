import { parseFitsArrayBuffer } from '../fitsParser'

function makeCard(keyword, value) {
  let line = keyword.padEnd(8, ' ')
  if (value !== undefined) {
    line += '= '
    line += String(value).padEnd(70, ' ')
  }
  return line.padEnd(80, ' ')
}

function createMinimalFitsBuffer() {
  const cards = [
    makeCard('SIMPLE', 'T'),
    makeCard('BITPIX', '-32'),
    makeCard('NAXIS', '2'),
    makeCard('NAXIS1', '2'),
    makeCard('NAXIS2', '2'),
    'END'.padEnd(80, ' '),
  ]

  let headerText = cards.join('')
  const headerPadding = (2880 - (headerText.length % 2880)) % 2880
  headerText += ' '.repeat(headerPadding)

  const encoder = new TextEncoder()
  const headerBytes = encoder.encode(headerText)

  const dataBuffer = new ArrayBuffer(16)
  const view = new DataView(dataBuffer)
  const values = [1.5, 2.5, 3.5, 4.5]
  values.forEach((v, i) => view.setFloat32(i * 4, v, false))
  const dataBytes = new Uint8Array(dataBuffer)

  const merged = new Uint8Array(headerBytes.length + dataBytes.length)
  merged.set(headerBytes, 0)
  merged.set(dataBytes, headerBytes.length)

  return merged.buffer
}

describe('parseFitsArrayBuffer', () => {
  it('parses headers and float32 pixel data from a FITS buffer', () => {
    const buffer = createMinimalFitsBuffer()
    const parsed = parseFitsArrayBuffer(buffer)

    expect(parsed.headers.BITPIX).toBe(-32)
    expect(parsed.headers.NAXIS1).toBe(2)
    expect(parsed.headers.NAXIS2).toBe(2)

    expect(parsed.width).toBe(2)
    expect(parsed.height).toBe(2)
    expect(Array.from(parsed.pixels)).toEqual([1.5, 2.5, 3.5, 4.5])
  })
})
