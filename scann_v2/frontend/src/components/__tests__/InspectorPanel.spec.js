import { flushPromises, mount } from '@vue/test-utils'

import InspectorPanel from '../InspectorPanel.vue'

describe('InspectorPanel', () => {
  let revisionChangedItems = []

  beforeEach(() => {
    revisionChangedItems = []
    globalThis.fetch = vi.fn(async (url, options) => {
      if (String(url).includes('/rollback/')) {
        return {
          ok: true,
          json: async () => ({
            task_id: 'PGC 17069',
            rolled_back_to_revision_id: 'rev-1',
            new_revision_id: 'rev-2',
            saved_count: 2,
          }),
        }
      }

      if (String(url).includes('/history/rev-2')) {
        return {
          ok: true,
          json: async () => ({
            revision_id: 'rev-2',
            task_id: 'PGC 17069',
            submitted_by: 'admin',
            saved_at: '2026-03-19T21:10:00+00:00',
            annotations: [],
            change_summary: { added: 0, modified: 0, removed: 0 },
            changed_items: [],
          }),
        }
      }

      if (String(url).includes('/history/rev-1')) {
        return {
          ok: true,
          json: async () => ({
            revision_id: 'rev-1',
            task_id: 'PGC 17069',
            submitted_by: 'annotator',
            saved_at: '2026-03-19T21:00:00+00:00',
            annotations: [],
            change_summary: { added: revisionChangedItems.length || 1, modified: 0, removed: 0 },
            changed_items: revisionChangedItems,
          }),
        }
      }

      if (String(url).includes('/history')) {
        return {
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
                change_summary: { added: 1, modified: 0, removed: 0 },
              },
            ],
          }),
        }
      }

      return {
        ok: true,
        json: async () => ({ ok: true, url, options }),
      }
    })
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

  it('rolls back a revision for admin users and emits refresh event', async () => {
    const wrapper = mount(InspectorPanel, {
      props: {
        taskId: 'PGC 17069',
        refreshKey: 0,
        userRole: 'admin',
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="history-row-rollback"]').trigger('click')
    await flushPromises()

    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/annotations/PGC%2017069/rollback/rev-1',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(wrapper.emitted('history-mutated')).toBeTruthy()
  })

  it('virtualizes large revision detail lists while retaining scroll access', async () => {
    revisionChangedItems = Array.from({ length: 500 }, (_, index) => ({
      change_type: 'added',
      changed_fields: [`field-${index}`],
    }))

    const wrapper = mount(InspectorPanel, {
      props: {
        taskId: 'PGC 17069',
        refreshKey: 0,
      },
    })

    await flushPromises()
    await wrapper.get('[data-testid="history-item"] button').trigger('click')
    await flushPromises()

    const detailList = wrapper.get('[data-testid="history-detail-list"]')
    expect(wrapper.text()).toContain('500')
    expect(detailList.findAll('li').length).toBeLessThan(50)
    expect(wrapper.text()).toContain('field-0')

    detailList.element.scrollTop = 400 * 28
    await detailList.trigger('scroll')

    expect(wrapper.text()).toContain('field-400')
  })
})
