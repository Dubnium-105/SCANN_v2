function parseHeaderValue(raw) {
  const trimmed = raw.trim()
  if (trimmed === 'T') return true
  if (trimmed === 'F') return false
  if (trimmed.startsWith("'") && trimmed.endsWith("'")) {
    return trimmed.slice(1, -1).trim()
  }
  const numeric = Number(trimmed)
  if (!Number.isNaN(numeric)) {
    return numeric
  }
  return trimmed
}

export function parseFitsArrayBuffer(buffer) {
  const bytes = new Uint8Array(buffer)
  const decoder = new TextDecoder('ascii')

  const headers = {}
  let headerOffset = 0

  while (headerOffset + 80 <= bytes.length) {
    const cardBytes = bytes.slice(headerOffset, headerOffset + 80)
    const card = decoder.decode(cardBytes)
    const keyword = card.slice(0, 8).trim()

    headerOffset += 80

    if (!keyword) {
      continue
    }

    if (keyword === 'END') {
      break
    }

    const hasValue = card[8] === '='
    if (!hasValue) {
      continue
    }

    const valueField = card.slice(10, 80)
    const commentSplit = valueField.split('/')
    headers[keyword] = parseHeaderValue(commentSplit[0])
  }

  const dataStart = Math.ceil(headerOffset / 2880) * 2880
  const bitpix = Number(headers.BITPIX)
  const naxis = Number(headers.NAXIS)
  const width = Number(headers.NAXIS1)
  const height = naxis >= 2 ? Number(headers.NAXIS2) : 1

  if (bitpix !== -32) {
    throw new Error(`Unsupported BITPIX: ${bitpix}`)
  }
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Invalid FITS dimensions')
  }

  const pixelCount = width * height
  const dataByteLength = pixelCount * 4
  if (dataStart + dataByteLength > bytes.length) {
    throw new Error('FITS data section is truncated')
  }

  const view = new DataView(buffer, dataStart, dataByteLength)
  const pixels = new Float32Array(pixelCount)
  for (let i = 0; i < pixelCount; i += 1) {
    pixels[i] = view.getFloat32(i * 4, false)
  }

  return {
    headers,
    width,
    height,
    pixels,
  }
}
