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

  // 支持多种FITS数据类型
  const bytesPerPixel = Math.abs(bitpix) / 8
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Invalid FITS dimensions')
  }

  const pixelCount = width * height
  const dataByteLength = pixelCount * bytesPerPixel
  if (dataStart + dataByteLength > bytes.length) {
    throw new Error('FITS data section is truncated')
  }

  let pixels
  const view = new DataView(buffer, dataStart, dataByteLength)

  switch (bitpix) {
    case 8: // 8位无符号字节
      pixels = new Float32Array(pixelCount)
      for (let i = 0; i < pixelCount; i += 1) {
        pixels[i] = view.getUint8(i)
      }
      break
    case 16: // 16位有符号整数
      pixels = new Float32Array(pixelCount)
      for (let i = 0; i < pixelCount; i += 1) {
        pixels[i] = view.getInt16(i * 2, false)
      }
      break
    case 32: // 32位有符号整数
      pixels = new Float32Array(pixelCount)
      for (let i = 0; i < pixelCount; i += 1) {
        pixels[i] = view.getInt32(i * 4, false)
      }
      break
    case -32: // 32位浮点数
      pixels = new Float32Array(pixelCount)
      for (let i = 0; i < pixelCount; i += 1) {
        pixels[i] = view.getFloat32(i * 4, false)
      }
      break
    case -64: // 64位双精度浮点数
      const doublePixels = new Float64Array(pixelCount)
      for (let i = 0; i < pixelCount; i += 1) {
        doublePixels[i] = view.getFloat64(i * 8, false)
      }
      // 转换为Float32以保持一致性
      pixels = new Float32Array(doublePixels)
      break
    default:
      throw new Error(`Unsupported BITPIX: ${bitpix}`)
  }

  return {
    headers,
    width,
    height,
    pixels,
  }
}
