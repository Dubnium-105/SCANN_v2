import { flushPromises, mount } from '@vue/test-utils'
import CanvasPanel from '../CanvasPanel.vue'
import { authState } from '../../services/authStore'

function createTask(taskId, overrides = {}) {
  return {
    task_id: taskId,
    old_path: `old/${taskId}.fts`,
    new_path: `new/${taskId}.fts`,
    new_marked_path: `new_marked/${taskId}.fts`,
    ...overrides,
  }
}

function createPrelabel(taskId, overrides = {}) {
  return {
    prelabel_id: `prelabel-${taskId}`,
    task_id: taskId,
    status: 'available',
    source_view: 'new',
    ai_suggestion: 'asteroid',
    ai_confidence: 0.91,
    model_version: 'detector-v1',
    model_id: 'model-20260413-001',
    model_backbone: 'ViT_B_16',
    box_count: 1,
    annotations: [
      {
        x: 16,
        y: 24,
        width: 20,
        height: 28,
        label: null,
        detail_type: 'asteroid',
        confidence: 0.91,
      },
    ],
    ...overrides,
  }
}

function makeCard(keyword, value) {
  let line = keyword.padEnd(8, ' ')
  if (value !== undefined) {
    line += '= '
    line += String(value).padEnd(70, ' ')
  }
  return line.padEnd(80, ' ')
}

function createFitsBuffer(values = [0, 0.5, 0.75, 1]) {
  const cards = [
    makeCard('SIMPLE', 'T'),
    makeCard('BITPIX', '-32'),
    makeCard('NAXIS', '2'),
    makeCard('NAXIS1', '2'),
    makeCard('NAXIS2', '2'),
    'END'.padEnd(80, ' '),
  ]

  let header = cards.join('')
  header += ' '.repeat((2880 - (header.length % 2880)) % 2880)

  const encoder = new TextEncoder()
  const headerBytes = encoder.encode(header)
  const dataBuffer = new ArrayBuffer(values.length * 4)
  const view = new DataView(dataBuffer)
  values.forEach((value, index) => {
    view.setFloat32(index * 4, value, false)
  })

  const merged = new Uint8Array(headerBytes.length + dataBuffer.byteLength)
  merged.set(headerBytes, 0)
  merged.set(new Uint8Array(dataBuffer), headerBytes.length)
  return merged.buffer
}

function mockImageFetch(pathValues = {}, options = {}) {
  const tasks = options.tasks || [createTask('PGC 17069')]
  const nextTaskId = options.nextTaskId || tasks[0]?.task_id || 'PGC 17069'
  const prelabels = options.prelabels || {}
  const calls = []
  globalThis.fetch = vi.fn((url, options) => {
    calls.push({ url, options })
    if (url === '/api/tasks' || String(url).startsWith('/api/tasks?')) {
      return Promise.resolve({
        ok: true,
        json: async () => tasks,
      })
    }

    if (String(url).startsWith('/api/tasks/next?')) {
      const task = tasks.find((item) => item.task_id === nextTaskId) || createTask(nextTaskId)
      return Promise.resolve({
        ok: true,
        json: async () => ({
          ...task,
          client_id: 'test-client',
          lock_expires_at: '2026-03-19T21:30:00+00:00',
        }),
      })
    }

    if (String(url).includes('/api/tasks/') && String(url).includes('/claim?')) {
      const taskId = decodeURIComponent(String(url).split('/api/tasks/')[1].split('/claim?')[0])
      const task = tasks.find((item) => item.task_id === taskId) || createTask(taskId)
      return Promise.resolve({
        ok: true,
        json: async () => ({
          ...task,
          client_id: 'test-client',
          lock_expires_at: '2026-03-19T21:30:00+00:00',
        }),
      })
    }

    if (String(url).includes('/api/tasks/') && String(url).includes('/heartbeat?')) {
      const taskId = decodeURIComponent(String(url).split('/api/tasks/')[1].split('/heartbeat?')[0])
      return Promise.resolve({
        ok: true,
        json: async () => ({
          task_id: taskId,
          client_id: 'test-client',
          lock_expires_at: '2026-03-19T21:30:00+00:00',
        }),
      })
    }

    if (String(url).includes('/api/tasks/') && String(url).includes('/release?')) {
      const taskId = decodeURIComponent(String(url).split('/api/tasks/')[1].split('/release?')[0])
      return Promise.resolve({
        ok: true,
        json: async () => ({
          task_id: taskId,
          client_id: 'test-client',
          released: true,
        }),
      })
    }

    if (String(url).startsWith('/api/fits/')) {
      const decodedUrl = decodeURIComponent(String(url))
      let fitsBuffer = createFitsBuffer()
      for (const [fragment, values] of Object.entries(pathValues)) {
        if (decodedUrl.includes(fragment)) {
          fitsBuffer = createFitsBuffer(values)
          break
        }
      }
      return Promise.resolve({
        ok: true,
        arrayBuffer: async () => fitsBuffer,
      })
    }

    if (String(url) === '/api/prelabels/enqueue') {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          requested_count: 1,
          enqueued_count: 1,
          skipped_count: 0,
          job_ids: ['job-1'],
          skipped_task_ids: [],
        }),
      })
    }

    if (String(url).startsWith('/api/prelabels/')) {
      const taskId = decodeURIComponent(String(url).split('/api/prelabels/')[1])
      const payload = prelabels[taskId]
      if (!payload) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: async () => ({ detail: 'Prelabel not found' }),
        })
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => payload,
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

  const globalStubs = {
    'v-stage': StageStub,
    'v-layer': { template: '<div><slot /></div>' },
    'v-image': { template: '<div />' },
    'v-rect': {
      props: ['config'],
      template: '<div data-testid="bbox-rect" :data-stroke="config?.stroke || \'\'" :data-fill="config?.fill || \'\'" />',
    },
    'v-circle': { template: '<div data-testid="point-shape" />' },
    'v-line': { template: '<div data-testid="polygon-shape" />' },
  }

  beforeEach(() => {
    URL.createObjectURL = vi.fn((blob) => `blob:${blob.type}`)
    URL.revokeObjectURL = vi.fn()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => null)
    sessionStorage.clear()
    authState.token = ''
    authState.username = ''
    authState.role = ''
    fetchCalls = mockImageFetch()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('creates 3 image state nodes and defaults to new layer visible', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const debugItems = wrapper.findAll('[data-testid="image-state-item"]')
    expect(debugItems).toHaveLength(3)

    const visibleViews = debugItems
      .filter((item) => item.attributes('data-visible') === 'true')
      .map((item) => item.attributes('data-view'))

    expect(visibleViews).toEqual(['new'])
    expect(fetchCalls.some((item) => String(item.url).startsWith('/api/tasks/next?'))).toBe(true)
  })

  it('cycles current view with Tab/Space keydown in new -> new_marked -> old order', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
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

  it('switches tools with H/C keyboard shortcuts', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const moveButton = () => wrapper.get('[data-testid="tool-move"]')
    const bboxButton = () => wrapper.get('[data-testid="tool-bbox"]')

    expect(moveButton().classes()).toContain('border-sky-400')
    expect(bboxButton().classes()).not.toContain('border-emerald-400')

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'c' }))
    await flushPromises()

    expect(bboxButton().classes()).toContain('border-emerald-400')
    expect(moveButton().classes()).not.toContain('border-sky-400')

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'H' }))
    await flushPromises()

    expect(moveButton().classes()).toContain('border-sky-400')
    expect(bboxButton().classes()).not.toContain('border-emerald-400')
  })

  it('blinks only checked views in queue order', async () => {
    vi.useFakeTimers()

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="blink-queue-new-marked"]').setValue(false)
    await flushPromises()

    expect(wrapper.get('[data-testid="blink-queue-text"]').text()).toContain('新图 -> 旧图')

    const currentVisible = () =>
      wrapper
        .findAll('[data-testid="image-state-item"]')
        .filter((item) => item.attributes('data-visible') === 'true')
        .map((item) => item.attributes('data-view'))

    expect(currentVisible()).toEqual(['new'])

    await wrapper.get('[data-testid="blink-toggle"]').trigger('click')
    await flushPromises()

    vi.advanceTimersByTime(500)
    await flushPromises()
    expect(currentVisible()).toEqual(['old'])

    vi.advanceTimersByTime(500)
    await flushPromises()
    expect(currentVisible()).toEqual(['new'])

    wrapper.unmount()
  })

  it('creates a bbox annotation via mousedown/mousemove/mouseup in bbox mode', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
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

  it.skip('loads AI prelabel summary and can apply/remove AI draft boxes', async () => {
    fetchCalls = mockImageFetch({}, {
      prelabels: {
        'PGC 17069': createPrelabel('PGC 17069'),
      },
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    expect(wrapper.get('[data-testid="prelabel-summary"]').text()).toContain('AI 草稿可用')
    expect(wrapper.findAll('[data-testid="annotation-item"]')).toHaveLength(0)

    await wrapper.get('[data-testid="apply-prelabel"]').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('[data-testid="annotation-item"]')).toHaveLength(1)
    expect(wrapper.get('[data-testid="prelabel-message"]').text()).toContain('已导入 1 个 AI 草稿框')

    await wrapper.get('[data-testid="remove-applied-prelabel"]').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('[data-testid="annotation-item"]')).toHaveLength(0)
    expect(wrapper.get('[data-testid="prelabel-message"]').text()).toContain('已移除 1 个 AI 导入框')
    expect(fetchCalls.some((item) => String(item.url).startsWith('/api/prelabels/PGC%2017069'))).toBe(true)
  })

  it('reviews AI prelabel boxes before importing them into manual annotations', async () => {
    fetchCalls = mockImageFetch({}, {
      prelabels: {
        'PGC 17069': createPrelabel('PGC 17069'),
      },
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    expect(wrapper.findAll('[data-testid="annotation-item"]')).toHaveLength(0)

    await wrapper.get('[data-testid="apply-prelabel"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="prelabel-review-panel"]').exists()).toBe(true)
    expect(wrapper.findAll('[data-testid="prelabel-review-item"]')).toHaveLength(1)

    await wrapper.get('[data-testid="prelabel-review-item"]').trigger('click')
    window.dispatchEvent(new KeyboardEvent('keydown', { key: '2' }))
    await flushPromises()

    const selectedPrelabelOverlay = wrapper
      .findAll('[data-testid="bbox-rect"]')
      .find((item) => item.attributes('data-stroke') === '#fbbf24')
    expect(selectedPrelabelOverlay).toBeTruthy()

    await wrapper.get('[data-testid="confirm-prelabel-review"]').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('[data-testid="annotation-item"]')).toHaveLength(1)
    expect(wrapper.get('[data-testid="annotation-label-select"]').element.value).toBe('supernova')
    expect(wrapper.get('[data-testid="prelabel-message"]').text()).toContain('AI')

    await wrapper.get('[data-testid="remove-applied-prelabel"]').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('[data-testid="annotation-item"]')).toHaveLength(0)
    expect(fetchCalls.some((item) => String(item.url).startsWith('/api/prelabels/PGC%2017069'))).toBe(true)
  })

  it('uses target_type from prelabel response when detail_type is missing', async () => {
    fetchCalls = mockImageFetch({}, {
      prelabels: {
        'PGC 17069': createPrelabel('PGC 17069', {
          ai_suggestion: null,
          annotations: [
            {
              x: 16,
              y: 24,
              width: 20,
              height: 28,
              label: null,
              target_type: 'asteroid',
              confidence: 0.91,
            },
          ],
        }),
      },
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="apply-prelabel"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="prelabel-review-item"]').text()).toContain('asteroid')

    await wrapper.get('[data-testid="confirm-prelabel-review"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="annotation-label-select"]').element.value).toBe('asteroid')
  })

  it('uses target_type from prelabel response when detail_type is unlabeled', async () => {
    fetchCalls = mockImageFetch({}, {
      prelabels: {
        'PGC 17069': createPrelabel('PGC 17069', {
          ai_suggestion: null,
          annotations: [
            {
              x: 16,
              y: 24,
              width: 20,
              height: 28,
              label: null,
              detail_type: 'unlabeled',
              target_type: 'asteroid',
              confidence: 0.91,
            },
          ],
        }),
      },
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="apply-prelabel"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="prelabel-review-item"]').text()).toContain('asteroid')

    await wrapper.get('[data-testid="confirm-prelabel-review"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="annotation-label-select"]').element.value).toBe('asteroid')
  })

  it('allows admins to request prelabel regeneration for the current task', async () => {
    authState.username = 'admin'
    authState.role = 'admin'
    fetchCalls = mockImageFetch({}, {
      tasks: [
        createTask('PGC 17069', {
          prelabel_status: 'accepted',
          prelabel_model_version: 'detector-v1',
          prelabel_model_id: 'model-20260413-001',
          prelabel_model_backbone: 'ViT_B_16',
        }),
      ],
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const button = wrapper.get('[data-testid="regenerate-prelabel"]')
    expect(button.attributes('disabled')).toBeUndefined()

    await button.trigger('click')
    await flushPromises()

    const enqueueCall = fetchCalls.find(
      (item) => String(item.url) === '/api/prelabels/enqueue' && item.options?.method === 'POST',
    )
    expect(enqueueCall).toBeTruthy()
    expect(JSON.parse(enqueueCall.options?.body)).toEqual({
      model_version: 'detector-v1',
      model_id: 'model-20260413-001',
      model_backbone: 'ViT_B_16',
      candidate_limit: null,
      confidence_threshold: null,
      task_ids: ['PGC 17069'],
      priority: 100,
      force: true,
    })
    expect(wrapper.get('[data-testid="prelabel-message"]').text()).toContain('已请求重新生成 AI 草稿')
  })

  it('submits drawn bbox annotations to backend endpoint', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="tool-bbox"]').trigger('click')
    const stage = wrapper.get('[data-testid="stage"]')
    await stage.trigger('mousedown', { clientX: 12, clientY: 18 })
    await stage.trigger('mousemove', { clientX: 42, clientY: 58 })
    await stage.trigger('mouseup', { clientX: 42, clientY: 58 })
    await flushPromises()

    const item = wrapper.findAll('[data-testid="annotation-item"]')[0]
    await item.trigger('click')
    const select = wrapper.get('[data-testid="annotation-label-select"]')
    await select.setValue('asteroid')
    await flushPromises()

    await wrapper.get('[data-testid="submit-annotations"]').trigger('click')
    await flushPromises()

    const submitCall = fetchCalls.find(
      (item) => String(item.url).startsWith('/api/annotations/') && item.options?.method === 'POST',
    )
    expect(submitCall).toBeTruthy()
    expect(submitCall.options?.method).toBe('POST')
    expect(String(submitCall.url)).toContain('client_id=')
    expect(String(submitCall.url)).toContain('release_after_save=false')

    const payload = JSON.parse(submitCall.options?.body)
    expect(payload.source_view).toBe('new')
    expect(payload.bucket).toBe('positive')
    expect(payload.annotations).toHaveLength(1)
    expect(payload.annotations[0]).toMatchObject({
      x: 12,
      y: 18,
      width: 30,
      height: 40,
      detail_type: 'asteroid',
    })
    expect(payload.annotations[0].label).toBeUndefined()
  })

  it('allows submitting manual crop boundary without bbox annotations', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="tool-crop"]').trigger('click')
    const stage = wrapper.get('[data-testid="stage"]')
    await stage.trigger('mousedown', { clientX: 0, clientY: 0 })
    await stage.trigger('mousemove', { clientX: 90, clientY: 70 })
    await stage.trigger('mouseup', { clientX: 90, clientY: 70 })
    await flushPromises()

    await wrapper.get('[data-testid="submit-annotations"]').trigger('click')
    await flushPromises()

    const submitCall = fetchCalls.find(
      (item) => String(item.url).startsWith('/api/annotations/') && item.options?.method === 'POST',
    )
    expect(submitCall).toBeTruthy()

    const payload = JSON.parse(submitCall.options?.body)
    expect(payload.annotations).toHaveLength(0)
    expect(payload.metadata?.manual_crop?.x).toBe(0)
    expect(payload.metadata?.manual_crop?.y).toBe(0)
    expect(payload.metadata?.manual_crop?.width).toBeGreaterThan(1)
    expect(payload.metadata?.manual_crop?.height).toBeGreaterThan(1)
  })

  it('auto-submits current task at the configured interval', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-09T00:00:00.000Z'))

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="tool-bbox"]').trigger('click')
    const stage = wrapper.get('[data-testid="stage"]')
    await stage.trigger('mousedown', { clientX: 20, clientY: 30 })
    await stage.trigger('mousemove', { clientX: 60, clientY: 90 })
    await stage.trigger('mouseup', { clientX: 60, clientY: 90 })
    await flushPromises()

    const item = wrapper.findAll('[data-testid="annotation-item"]')[0]
    await item.trigger('click')
    await wrapper.get('[data-testid="annotation-label-select"]').setValue('asteroid')
    await flushPromises()

    await wrapper.get('[data-testid="auto-submit-interval"]').setValue('30')
    await wrapper.get('[data-testid="auto-submit-toggle"]').setValue(true)
    await flushPromises()

    expect(wrapper.get('[data-testid="auto-submit-countdown"]').text()).toContain('00:00:30')

    await vi.advanceTimersByTimeAsync(30_000)
    await flushPromises()

    const submitCalls = fetchCalls.filter(
      (item) => String(item.url).startsWith('/api/annotations/') && item.options?.method === 'POST',
    )
    expect(submitCalls).toHaveLength(1)
    expect(String(submitCalls[0].url)).toContain('release_after_save=false')
    expect(JSON.parse(submitCalls[0].options?.body).bucket).toBe('positive')
    expect(wrapper.get('[data-testid="auto-submit-status"]').text()).toContain('自动提交成功')
  })

  it('updates rendered rgba with stretch sliders and invert toggle', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const debug = () => wrapper.get('[data-testid="stretch-debug"]').attributes('data-rgba')

    const initial = debug()
    expect(initial).toContain('0,0,0,255')

    await wrapper.get('[data-testid="stretch-min-slider"]').setValue('0.5')
    await flushPromises()
    const afterMin = debug()
    expect(afterMin).toContain('0,0,0,255')

    await wrapper.get('[data-testid="invert-toggle"]').setValue(true)
    await flushPromises()
    const afterInvert = debug()
    expect(afterInvert.startsWith('255,255,255,255')).toBe(true)
  })

  it('renders FITS pixels through a non-smoothed pixelated canvas without CSS scale transforms', async () => {
    const originalImageData = globalThis.ImageData
    try {
      if (typeof globalThis.ImageData === 'undefined') {
        globalThis.ImageData = class ImageDataMock {
          constructor(data, width, height) {
            this.data = data
            this.width = width
            this.height = height
          }
        }
      }

      const context = {
        clearRect: vi.fn(),
        putImageData: vi.fn(),
        imageSmoothingEnabled: true,
      }
      HTMLCanvasElement.prototype.getContext.mockImplementation(() => context)

      const wrapper = mount(CanvasPanel, {
        global: {
          stubs: globalStubs,
        },
      })

      await flushPromises()

      const canvas = wrapper.get('[data-testid="fits-render-canvas"]').element
      expect(canvas.style.imageRendering).toBe('pixelated')
      expect(canvas.style.transform).toBe('')
      expect(canvas.style.width).toBe('2px')
      expect(canvas.style.height).toBe('2px')
      expect(context.imageSmoothingEnabled).toBe(false)
      expect(context.clearRect).toHaveBeenCalledWith(0, 0, 2, 2)
      expect(context.putImageData).toHaveBeenCalled()
    } finally {
      if (originalImageData === undefined) {
        delete globalThis.ImageData
      } else {
        globalThis.ImageData = originalImageData
      }
    }
  })

  it('auto-applies brightness-matched stretch per task view and can sync other views from current min/max', async () => {
    mockImageFetch({
      'new/PGC 17069.fts': [0, 1, 2, 3],
      'old/PGC 17069.fts': [10, 11, 12, 13],
      'new_marked/PGC 17069.fts': [100, 101, 102, 103],
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="switch-view-old"]').trigger('click')
    await flushPromises()
    const initialOldMin = Number(wrapper.get('[data-testid="stretch-min-slider"]').element.value)
    const initialOldMax = Number(wrapper.get('[data-testid="stretch-max-slider"]').element.value)
    expect(initialOldMin).not.toBe(10)
    expect(initialOldMax).not.toBe(13)

    await wrapper.get('[data-testid="switch-view-new"]').trigger('click')
    await flushPromises()
    await wrapper.get('[data-testid="stretch-min-slider"]').setValue('1.2')
    await wrapper.get('[data-testid="stretch-max-slider"]').setValue('1.8')
    await flushPromises()
    await wrapper.get('[data-testid="match-group-stretch"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-testid="switch-view-old"]').trigger('click')
    await flushPromises()
    const oldMin = Number(wrapper.get('[data-testid="stretch-min-slider"]').element.value)
    const oldMax = Number(wrapper.get('[data-testid="stretch-max-slider"]').element.value)
    expect(oldMin).not.toBe(initialOldMin)
    expect(oldMax).not.toBe(initialOldMax)
  })

  it('renders point and polygon annotations after switching tools', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const stage = wrapper.get('[data-testid="stage"]')

    await wrapper.get('[data-testid="tool-point"]').trigger('click')
    await stage.trigger('mouseup', { clientX: 30, clientY: 40 })
    await flushPromises()
    expect(wrapper.findAll('[data-testid="point-shape"]').length).toBeGreaterThanOrEqual(1)

    await wrapper.get('[data-testid="tool-polygon"]').trigger('click')
    await stage.trigger('mouseup', { clientX: 10, clientY: 10 })
    await stage.trigger('mouseup', { clientX: 20, clientY: 10 })
    await stage.trigger('mouseup', { clientX: 20, clientY: 20 })
    await wrapper.get('[data-testid="finish-polygon"]').trigger('click')
    await flushPromises()

    const polygonItems = wrapper
      .findAll('[data-testid="annotation-item"]')
      .filter((item) => item.attributes('data-ann-type') === 'polygon')
    expect(polygonItems.length).toBeGreaterThanOrEqual(1)
  })

  it('virtualizes large annotation side lists while retaining scroll access', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const stage = wrapper.get('[data-testid="stage"]')
    await wrapper.get('[data-testid="tool-point"]').trigger('click')
    for (let index = 0; index < 120; index += 1) {
      await stage.trigger('mouseup', { clientX: 10 + index, clientY: 20 + index })
    }
    await flushPromises()

    const annotationList = wrapper.get('[data-testid="annotation-list"]')
    expect(wrapper.findAll('[data-testid="annotation-item"]').length).toBeLessThan(40)

    annotationList.element.scrollTop = 70 * 36
    await annotationList.trigger('scroll')
    await flushPromises()

    const visibleIds = wrapper
      .findAll('[data-testid="annotation-item"]')
      .map((item) => item.attributes('data-ann-display-id'))
    expect(visibleIds).toContain('A0071')
  })

  it('updates selected annotation label to target type from annotation list', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const stage = wrapper.get('[data-testid="stage"]')
    await wrapper.get('[data-testid="tool-point"]').trigger('click')
    await stage.trigger('mouseup', { clientX: 44, clientY: 52 })
    await flushPromises()

    const item = wrapper.findAll('[data-testid="annotation-item"]')[0]
    expect(item).toBeTruthy()
    await item.trigger('click')

    const select = wrapper.get('[data-testid="annotation-label-select"]')
    await select.setValue('asteroid')
    await flushPromises()

    expect(wrapper.findAll('[data-testid="annotation-item"]')[0].attributes('data-ann-label')).toBe('asteroid')
  })

  it('selects an existing annotation by clicking it on the canvas in move mode', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const stage = wrapper.get('[data-testid="stage"]')
    await wrapper.get('[data-testid="tool-bbox"]').trigger('click')
    await stage.trigger('mousedown', { clientX: 10, clientY: 20 })
    await stage.trigger('mousemove', { clientX: 30, clientY: 40 })
    await stage.trigger('mouseup', { clientX: 30, clientY: 40 })
    await stage.trigger('mousedown', { clientX: 70, clientY: 80 })
    await stage.trigger('mousemove', { clientX: 100, clientY: 110 })
    await stage.trigger('mouseup', { clientX: 100, clientY: 110 })
    await flushPromises()

    const items = () => wrapper.findAll('[data-testid="annotation-item"]')
    expect(items()).toHaveLength(2)
    await items()[0].trigger('click')
    await flushPromises()

    expect(items()[0].classes()).toContain('border-sky-500')
    expect(items()[1].classes()).not.toContain('border-sky-500')

    await wrapper.get('[data-testid="tool-move"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[data-testid="tool-move"]').classes()).toContain('border-sky-400')

    await stage.trigger('mousedown', { clientX: 85, clientY: 95 })
    await stage.trigger('mouseup', { clientX: 85, clientY: 95 })
    await flushPromises()

    expect(items()).toHaveLength(2)
    expect(items()[1].classes()).toContain('border-sky-500')
    expect(items()[0].classes()).not.toContain('border-sky-500')
  })

  it('prefers the smaller bbox when overlapping boxes both contain the click point', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const stage = wrapper.get('[data-testid="stage"]')
    await wrapper.get('[data-testid="tool-bbox"]').trigger('click')

    await stage.trigger('mousedown', { clientX: 40, clientY: 40 })
    await stage.trigger('mousemove', { clientX: 60, clientY: 60 })
    await stage.trigger('mouseup', { clientX: 60, clientY: 60 })

    await stage.trigger('mousedown', { clientX: 20, clientY: 20 })
    await stage.trigger('mousemove', { clientX: 100, clientY: 100 })
    await stage.trigger('mouseup', { clientX: 100, clientY: 100 })
    await flushPromises()

    const items = () => wrapper.findAll('[data-testid="annotation-item"]')
    expect(items()).toHaveLength(2)
    expect(items()[1].classes()).toContain('border-sky-500')

    await wrapper.get('[data-testid="tool-move"]').trigger('click')
    await stage.trigger('mousedown', { clientX: 50, clientY: 50 })
    await stage.trigger('mouseup', { clientX: 50, clientY: 50 })
    await flushPromises()

    expect(items()[0].attributes('data-ann-display-id')).toBe('A0001')
    expect(items()[0].classes()).toContain('border-sky-500')
    expect(items()[1].classes()).not.toContain('border-sky-500')
  })

  it('supports disappeared target label options and keyboard shortcuts', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    const stage = wrapper.get('[data-testid="stage"]')
    await wrapper.get('[data-testid="tool-point"]').trigger('click')
    await stage.trigger('mouseup', { clientX: 30, clientY: 36 })
    await flushPromises()

    const item = wrapper.findAll('[data-testid="annotation-item"]')[0]
    await item.trigger('click')

    const select = wrapper.get('[data-testid="annotation-label-select"]')
    const optionValues = select.findAll('option').map((option) => option.element.value)
    expect(optionValues).toContain('disappeared_asteroid')
    expect(optionValues).toContain('disappeared_star')
    expect(optionValues).toContain('disappeared_galaxy')

    window.dispatchEvent(new KeyboardEvent('keydown', { key: '9' }))
    await flushPromises()
    expect(wrapper.get('[data-testid="annotation-label-select"]').element.value).toBe('disappeared_asteroid')

    window.dispatchEvent(new KeyboardEvent('keydown', { key: '0' }))
    await flushPromises()
    expect(wrapper.get('[data-testid="annotation-label-select"]').element.value).toBe('disappeared_star')

    window.dispatchEvent(new KeyboardEvent('keydown', { key: '-' }))
    await flushPromises()
    expect(wrapper.get('[data-testid="annotation-label-select"]').element.value).toBe('disappeared_galaxy')
  })

  it('requires second click to delete and supports undo for annotation item', async () => {
    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="tool-point"]').trigger('click')
    const stage = wrapper.get('[data-testid="stage"]')
    await stage.trigger('mouseup', { clientX: 44, clientY: 52 })
    await flushPromises()

    const countBeforeDelete = wrapper.findAll('[data-testid="annotation-item"]').length
    expect(countBeforeDelete).toBeGreaterThanOrEqual(1)

    await wrapper.get('[data-testid="annotation-remove"]').trigger('click')
    await flushPromises()
    expect(wrapper.findAll('[data-testid="annotation-item"]').length).toBe(countBeforeDelete)

    await wrapper.get('[data-testid="annotation-remove"]').trigger('click')
    await flushPromises()

    const countAfterDelete = wrapper.findAll('[data-testid="annotation-item"]').length
    expect(countAfterDelete).toBe(countBeforeDelete - 1)
    expect(wrapper.find('[data-testid="undo-delete-banner"]').exists()).toBe(true)

    await wrapper.get('[data-testid="undo-delete"]').trigger('click')
    await flushPromises()

    const countAfterUndo = wrapper.findAll('[data-testid="annotation-item"]').length
    expect(countAfterUndo).toBe(countBeforeDelete)
  })

  it('shows lock status in task catalog only for occupied task groups', async () => {
    fetchCalls = mockImageFetch({}, {
      tasks: [
        createTask('PGC 17069', {
          lock_expires_at: '2026-03-19T21:30:00+00:00',
          locked_by_current_client: true,
        }),
        createTask('PGC 35671'),
      ],
      nextTaskId: 'PGC 17069',
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="task-catalog-open"]').trigger('click')
    await flushPromises()

    const lockBadges = wrapper.findAll('[data-testid="task-catalog-lock-status"]')
    expect(lockBadges).toHaveLength(1)
    expect(lockBadges[0].text()).toBe('当前会话占用')
  })

  it('shows prelabel status badges in task catalog when available', async () => {
    fetchCalls = mockImageFetch({}, {
      tasks: [
        createTask('PGC 17069', {
          prelabel_status: 'available',
          prelabel_model_version: 'detector-v1',
        }),
        createTask('PGC 35671'),
      ],
      nextTaskId: 'PGC 17069',
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="task-catalog-open"]').trigger('click')
    await flushPromises()

    const prelabelBadges = wrapper.findAll('[data-testid="task-catalog-prelabel-status"]')
    expect(prelabelBadges).toHaveLength(1)
    expect(prelabelBadges[0].text()).toContain('AI 草稿可用')
  })

  it('blocks switching when target task group is occupied by another user', async () => {
    fetchCalls = mockImageFetch({}, {
      tasks: [
        createTask('PGC 17069'),
        createTask('PGC 35671', {
          lock_expires_at: '2026-03-19T21:30:00+00:00',
          locked_by_current_client: false,
        }),
      ],
      nextTaskId: 'PGC 17069',
    })

    const wrapper = mount(CanvasPanel, {
      global: {
        stubs: globalStubs,
      },
    })

    await flushPromises()

    await wrapper.get('[data-testid="task-next"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="task-switch-error"]').text()).toContain('PGC 35671')
    expect(wrapper.get('[data-testid="task-switch-error"]').text()).toContain('当前被其他用户占用')
    expect(fetchCalls.some((item) => String(item.url).includes('/api/tasks/PGC%2035671/claim?'))).toBe(false)
    expect(wrapper.text()).toContain('PGC 17069')
  })
})
