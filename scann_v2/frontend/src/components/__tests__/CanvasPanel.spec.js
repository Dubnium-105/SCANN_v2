import { flushPromises, mount } from '@vue/test-utils'
import CanvasPanel from '../CanvasPanel.vue'

function mockImageFetch() {
  globalThis.fetch = vi.fn((url) => {
    if (url === '/api/tasks') {
      return Promise.resolve({
        ok: true,
        json: async () => [
          {
            task_id: 'PGC 17069',
            old_path: 'old/PGC 17069.fts',
            new_path: 'new/PGC 17069.fts',
            new_marked_path: 'new_marked/PGC 17069.fts',
          },
        ],
      })
    }

    return Promise.resolve({
      ok: true,
      blob: async () => new Blob(['png-bytes'], { type: 'image/png' }),
    })
  })
}

describe('CanvasPanel', () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn((blob) => `blob:${blob.type}`)
    URL.revokeObjectURL = vi.fn()
    mockImageFetch()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('creates 3 image state nodes and defaults to new layer visible', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: {
          'v-stage': { template: '<div><slot /></div>' },
          'v-layer': { template: '<div><slot /></div>' },
          'v-image': { template: '<div />' },
        },
      },
    })

    await flushPromises()

    const debugItems = wrapper.findAll('[data-testid="image-state-item"]')
    expect(debugItems).toHaveLength(3)

    const visibleViews = debugItems
      .filter((item) => item.attributes('data-visible') === 'true')
      .map((item) => item.attributes('data-view'))

    expect(visibleViews).toEqual(['new'])
  })

  it('cycles current view with Tab/Space keydown in new -> new_marked -> old order', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: {
          'v-stage': { template: '<div><slot /></div>' },
          'v-layer': { template: '<div><slot /></div>' },
          'v-image': { template: '<div />' },
        },
      },
    })

    await flushPromises()

    const currentVisible = () =>
      wrapper
        .findAll('[data-testid="image-state-item"]')
        .filter((item) => item.attributes('data-visible') === 'true')
        .map((item) => item.attributes('data-view'))

    expect(currentVisible()).toEqual(['new'])

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab' }))
    await flushPromises()
    expect(currentVisible()).toEqual(['new_marked'])

    window.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }))
    await flushPromises()
    expect(currentVisible()).toEqual(['old'])
  })
})
