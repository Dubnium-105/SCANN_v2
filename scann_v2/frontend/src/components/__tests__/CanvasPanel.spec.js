import { flushPromises, mount } from '@vue/test-utils'
import CanvasPanel from '../CanvasPanel.vue'

function mockImageFetch() {
  const calls = []
  globalThis.fetch = vi.fn((url, options) => {
    calls.push({ url, options })
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
      json: async () => ({
        task_id: 'PGC 17069',
        saved_count: 1,
      }),
    })
  })
  return calls
}

const StageStub = {
  template:
    '<div data-testid="stage" @mousedown="$emit(\'mousedown\', $event)" @mousemove="$emit(\'mousemove\', $event)" @mouseup="$emit(\'mouseup\', $event)"><slot /></div>',
}

describe('CanvasPanel', () => {
  let fetchCalls = []

  beforeEach(() => {
    URL.createObjectURL = vi.fn((blob) => `blob:${blob.type}`)
    URL.revokeObjectURL = vi.fn()
    fetchCalls = mockImageFetch()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('creates 3 image state nodes and defaults to new layer visible', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: {
          'v-stage': StageStub,
          'v-layer': { template: '<div><slot /></div>' },
          'v-image': { template: '<div />' },
          'v-rect': { template: '<div data-testid="bbox-rect" />' },
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
          'v-stage': StageStub,
          'v-layer': { template: '<div><slot /></div>' },
          'v-image': { template: '<div />' },
          'v-rect': { template: '<div data-testid="bbox-rect" />' },
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

  it('creates a bbox annotation via mousedown/mousemove/mouseup in bbox mode', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: {
          'v-stage': StageStub,
          'v-layer': { template: '<div><slot /></div>' },
          'v-image': { template: '<div />' },
          'v-rect': { template: '<div data-testid="bbox-rect" />' },
        },
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="tool-bbox"]').trigger('click')

    const stage = wrapper.get('[data-testid="stage"]')
    await stage.trigger('mousedown', { clientX: 10, clientY: 20 })
    await stage.trigger('mousemove', { clientX: 60, clientY: 80 })
    await stage.trigger('mouseup', { clientX: 60, clientY: 80 })
    await flushPromises()

    const rects = wrapper.findAll('[data-testid="bbox-rect"]')
    expect(rects.length).toBeGreaterThanOrEqual(1)
  })

  it('submits drawn bbox annotations to backend endpoint', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: {
          'v-stage': StageStub,
          'v-layer': { template: '<div><slot /></div>' },
          'v-image': { template: '<div />' },
          'v-rect': { template: '<div data-testid="bbox-rect" />' },
        },
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="tool-bbox"]').trigger('click')
    const stage = wrapper.get('[data-testid="stage"]')
    await stage.trigger('mousedown', { clientX: 12, clientY: 18 })
    await stage.trigger('mousemove', { clientX: 42, clientY: 58 })
    await stage.trigger('mouseup', { clientX: 42, clientY: 58 })
    await flushPromises()

    await wrapper.get('[data-testid="submit-annotations"]').trigger('click')
    await flushPromises()

    const submitCall = fetchCalls.find((item) => String(item.url).startsWith('/api/annotations/'))
    expect(submitCall).toBeTruthy()
    expect(submitCall.options?.method).toBe('POST')

    const payload = JSON.parse(submitCall.options?.body)
    expect(payload.bucket).toBe('positive')
    expect(payload.annotations).toHaveLength(1)
    expect(payload.annotations[0]).toMatchObject({ x: 12, y: 18, width: 30, height: 40 })
  })
})
