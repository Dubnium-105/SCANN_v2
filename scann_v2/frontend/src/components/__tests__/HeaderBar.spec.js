import { flushPromises, mount } from '@vue/test-utils'
import { nextTick } from 'vue'

import HeaderBar from '../HeaderBar.vue'

function createToken(payload = {}) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString('base64url')
  return `${encode({ alg: 'none', typ: 'JWT' })}.${encode(payload)}.signature`
}

describe('HeaderBar', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('shows username in header', () => {
    const wrapper = mount(HeaderBar, {
      props: {
        username: 'annotator',
      },
    })

    expect(wrapper.get('[data-testid="header-username"]').text()).toContain('annotator')
  })

  it('shows session remaining time from token expiration', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-09T00:00:00.000Z'))

    const wrapper = mount(HeaderBar, {
      props: {
        username: 'annotator',
        token: createToken({
          sub: 'annotator',
          exp: Math.floor(Date.now() / 1000) + 61,
        }),
      },
    })

    expect(wrapper.get('[data-testid="header-session-remaining"]').text()).toBe('会话剩余 00:01:01')

    await vi.advanceTimersByTimeAsync(1000)
    await nextTick()

    expect(wrapper.get('[data-testid="header-session-remaining"]').text()).toBe('会话剩余 00:01:00')
  })

  it('shows manual annotation sync menu only for admin users', async () => {
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        enabled: true,
        configured: true,
        scheduled: false,
        running: false,
        interval_seconds: 300,
        dataset_id: 'dataset-a',
        schema_name: 'scann_backup',
        scope: 'annotations_only',
        last_result: null,
      }),
    }))

    const annotatorWrapper = mount(HeaderBar, {
      props: {
        username: 'annotator',
        role: 'annotator',
      },
    })
    expect(annotatorWrapper.find('[data-testid="header-sync-menu-toggle"]').exists()).toBe(false)

    const adminWrapper = mount(HeaderBar, {
      props: {
        username: 'admin',
        role: 'admin',
      },
    })
    expect(adminWrapper.find('[data-testid="header-sync-menu-toggle"]').exists()).toBe(true)

    await adminWrapper.get('[data-testid="header-sync-menu-toggle"]').trigger('click')
    await flushPromises()

    expect(adminWrapper.find('[data-testid="header-sync-menu"]').exists()).toBe(true)
    expect(adminWrapper.text()).toContain('dataset-a')
  })

  it('runs manual incremental and full sync from admin menu', async () => {
    globalThis.fetch = vi.fn(async (url, options) => {
      if (String(url).startsWith('/api/annotation-sync/run')) {
        return {
          ok: true,
          json: async () => ({
            success: true,
            dataset_id: 'dataset-a',
            started_at: '2026-04-10T00:00:00+00:00',
            finished_at: '2026-04-10T00:00:01+00:00',
            sync_mode: String(url).includes('full=true') ? 'full' : 'incremental',
            previous_revision_rowid: 0,
            last_revision_rowid: 12,
            tasks_synced: 1,
            revisions_synced: 2,
            current_boxes_synced: 3,
            revision_boxes_synced: 4,
          }),
        }
      }

      return {
        ok: true,
        json: async () => ({
          enabled: true,
          configured: true,
          scheduled: false,
          running: false,
          interval_seconds: 300,
          dataset_id: 'dataset-a',
          schema_name: 'scann_backup',
          scope: 'annotations_only',
          last_result: null,
        }),
      }
    })

    const wrapper = mount(HeaderBar, {
      props: {
        username: 'admin',
        role: 'admin',
      },
    })

    await wrapper.get('[data-testid="header-sync-menu-toggle"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-testid="header-sync-run"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-testid="header-sync-run-full"]').trigger('click')
    await flushPromises()

    expect(globalThis.fetch).toHaveBeenCalledWith('/api/annotation-sync/run', expect.objectContaining({ method: 'POST' }))
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/annotation-sync/run?full=true', expect.objectContaining({ method: 'POST' }))
    expect(wrapper.get('[data-testid="header-sync-message"]').text()).toContain('全量同步完成')
  })
})
