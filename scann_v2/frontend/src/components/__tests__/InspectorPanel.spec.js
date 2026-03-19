import { flushPromises, mount } from '@vue/test-utils'

import InspectorPanel from '../InspectorPanel.vue'

describe('InspectorPanel', () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        task_id: 'PGC 17069',
        revisions: [
          {
            revision_id: 'rev-1',
            task_id: 'PGC 17069',
            submitted_by: 'annotator',
            saved_at: '2026-03-19T21:00:00+00:00',
            bucket: 'positive',
            annotation_count: 2,
          },
        ],
      }),
    }))
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('loads and renders revision timeline for task', async () => {
    const wrapper = mount(InspectorPanel, {
      props: {
        taskId: 'PGC 17069',
        refreshKey: 0,
      },
    })

    await flushPromises()

    expect(globalThis.fetch).toHaveBeenCalledWith('/api/annotations/PGC%2017069/history', expect.any(Object))
    expect(wrapper.findAll('[data-testid="history-item-user"]')).toHaveLength(1)
    expect(wrapper.text()).toContain('annotator')
  })
})
