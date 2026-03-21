import { useFitsImagePool } from '../useFitsImagePool'

function makeCard(keyword, value) {
  let line = keyword.padEnd(8, ' ')
  if (value !== undefined) {
    line += '= '
    line += String(value).padEnd(70, ' ')
  }
  return line.padEnd(80, ' ')
}

function createFitsBuffer() {
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
  const values = [0.1, 0.2, 0.3, 0.4]
  values.forEach((v, i) => view.setFloat32(i * 4, v, false))
  const dataBytes = new Uint8Array(dataBuffer)

  const merged = new Uint8Array(headerBytes.length + dataBytes.length)
  merged.set(headerBytes, 0)
  merged.set(dataBytes, headerBytes.length)

  return merged.buffer
}

describe('useFitsImagePool', () => {
  it('loads old/new/new_marked FITS nodes and parses headers/pixels', async () => {
    const fitsBuffer = createFitsBuffer()
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => fitsBuffer,
    }))

    const { fitsNodes, preloadTaskFits, fitsError } = useFitsImagePool(fetchImpl)

    await preloadTaskFits({
      old_path: 'old/PGC 17069.fts',
      new_path: 'new/PGC 17069.fts',
      new_marked_path: 'new_marked/PGC 17069.fts',
    })

    expect(fetchImpl).toHaveBeenCalledTimes(3)
    expect(fitsError.value).toBe('')
    expect(fitsNodes.value).toHaveLength(3)

    for (const node of fitsNodes.value) {
      expect(node.headers.BITPIX).toBe(-32)
      expect(node.width).toBe(2)
      expect(node.height).toBe(2)
      expect(Array.from(node.pixels)).toEqual([
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645,
      ])
    }
  })
})
