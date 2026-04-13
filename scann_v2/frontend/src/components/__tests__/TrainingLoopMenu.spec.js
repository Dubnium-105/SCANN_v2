import { flushPromises, mount } from '@vue/test-utils'

import TrainingLoopMenu from '../TrainingLoopMenu.vue'

function jsonResponse(payload, ok = true, status = 200) {
  return {
    ok,
    status,
    json: async () => payload,
  }
}

function buildBaseFetchMock(extraHandler = null) {
  return vi.fn(async (url, options = {}) => {
    if (extraHandler) {
      const handled = await extraHandler(url, options)
      if (handled) {
        return handled
      }
    }

    if (String(url).startsWith('/api/training/snapshots') && !options.method) {
      return jsonResponse([
        {
          snapshot_id: 'snapshot-1',
          snapshot_name: 'round-1',
          annotation_count: 12,
          task_count: 10,
        },
      ])
    }
    if (String(url).startsWith('/api/training/jobs') && !options.method) {
      return jsonResponse([
        {
          job_id: 'job-1',
          snapshot_id: 'snapshot-1',
          model_version: 'cls-v3',
          model_id: 'model-1',
          model_backbone: 'ViT_B_16',
          status: 'queued',
          attempt_count: 1,
        },
      ])
    }
    if (String(url).startsWith('/api/training/runs')) {
      return jsonResponse([
        {
          run_id: 'run-1',
          model_version: 'cls-v2',
          status: 'completed',
          metrics: { f1: 0.91 },
        },
      ])
    }
    if (String(url).startsWith('/api/training/models/promoted')) {
      return jsonResponse({
        model_id: 'model-0',
        model_version: 'cls-v2',
        model_backbone: 'ResNet18',
        task_type: 'classification',
        promoted_at: '2026-04-13T08:00:00+00:00',
      })
    }
    if (String(url).startsWith('/api/training/models') && !options.method) {
      return jsonResponse([
        {
          model_id: 'model-0',
          model_version: 'cls-v2',
          model_backbone: 'ResNet18',
          is_promoted: true,
          metrics: { f1: 0.91 },
        },
        {
          model_id: 'model-1',
          model_version: 'cls-v3',
          model_backbone: 'ViT_B_16',
          is_promoted: false,
          metrics: { f1: 0.93 },
        },
      ])
    }
    if (String(url).startsWith('/api/tasks')) {
      return jsonResponse([
        {
          task_id: 'PGC 17069',
          new_path: 'new/PGC 17069.fts',
          field_name: 'Field A',
          field_key: 'field-a',
          capture_key: 'capture-1',
          prelabel_status: 'queued',
          prelabel_model_version: 'cls-v2',
          prelabel_model_id: 'model-0',
          prelabel_model_backbone: 'ResNet18',
        },
        {
          task_id: 'PGC 17070',
          new_path: 'new/PGC 17070.fts',
          field_name: 'Field A',
          field_key: 'field-a',
          capture_key: 'capture-1',
          prelabel_status: 'failed',
          prelabel_model_version: 'cls-v1',
          prelabel_model_id: 'model-old',
          prelabel_model_backbone: 'ResNet18',
        },
        {
          task_id: 'PGC 17071',
          new_path: 'new/PGC 17071.fts',
          field_name: 'Field B',
          field_key: 'field-b',
          capture_key: 'capture-2',
          prelabel_status: 'available',
          prelabel_model_version: 'cls-v2',
          prelabel_model_id: 'model-0',
          prelabel_model_backbone: 'ResNet18',
        },
      ])
    }
    if (String(url).startsWith('/api/prelabels/jobs')) {
      return jsonResponse([
        {
          job_id: 'prelabel-job-1',
          task_id: 'PGC 17069',
          requested_by: 'admin',
          status: 'queued',
          model_version: 'cls-v2',
          model_id: 'model-0',
          model_backbone: 'ResNet18',
          input_fingerprint: 'fingerprint-1',
          attempt_count: 0,
        },
      ])
    }
    if (String(url).startsWith('/api/prelabels/workers')) {
      return jsonResponse([
        {
          worker_id: 'gpu-worker-1',
          display_name: 'GPU Worker',
          host_name: 'pc-01',
          device_label: 'RTX-4090',
          status: 'online',
          capabilities: {
            model_versions: ['cls-v2'],
            model_ids: ['model-0'],
            model_backbones: ['ResNet18'],
          },
          last_seen_at: '2026-04-13T08:10:00+00:00',
        },
      ])
    }

    throw new Error(`Unexpected URL: ${url}`)
  })
}

describe('TrainingLoopMenu', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('loads training control state for admin workflow', async () => {
    globalThis.fetch = buildBaseFetchMock()

    const wrapper = mount(TrainingLoopMenu, {
      props: {
        activeTaskId: 'PGC 17069',
      },
    })

    await wrapper.get('[data-testid="header-training-menu-toggle"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="header-training-menu"]').text()).toContain('round-1')
    expect(wrapper.get('[data-testid="training-promoted-model"]').text()).toContain('cls-v2')
    expect(wrapper.get('[data-testid="prelabel-control-panel"]').text()).toContain('GPU Worker')
    expect(wrapper.get('[data-testid="prelabel-selected-summary"]').text()).toContain('当前队列 1 个')
  })

  it('creates snapshots and training jobs, then can promote a model with targeted prelabels', async () => {
    const fetchMock = buildBaseFetchMock(async (url, options = {}) => {
      if (String(url) === '/api/training/snapshots' && options.method === 'POST') {
        return jsonResponse({
          snapshot_id: 'snapshot-2',
          snapshot_name: 'round-2',
          annotation_count: 3,
          task_count: 1,
        })
      }
      if (String(url) === '/api/training/jobs' && options.method === 'POST') {
        return jsonResponse({
          job_id: 'job-2',
          snapshot_id: 'snapshot-2',
          model_version: 'cls-v4',
          model_id: 'model-2',
          model_backbone: 'ViT_B_16',
          status: 'queued',
          attempt_count: 0,
        })
      }
      if (
        String(url) === '/api/training/models/model-1/promote?enqueue_prelabels=true&force_prelabel=true&task_ids=PGC+17069'
        && options.method === 'POST'
      ) {
        return jsonResponse({
          model: {
            model_id: 'model-1',
            model_version: 'cls-v3',
            model_backbone: 'ViT_B_16',
            task_type: 'classification',
            is_promoted: true,
            promoted_at: '2026-04-13T08:10:00+00:00',
          },
          prelabel_enqueue: {
            requested_count: 1,
            enqueued_count: 1,
            skipped_count: 0,
          },
        })
      }
      return null
    })
    globalThis.fetch = fetchMock

    const wrapper = mount(TrainingLoopMenu, {
      props: {
        activeTaskId: 'PGC 17069',
      },
    })

    await wrapper.get('[data-testid="header-training-menu-toggle"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-testid="training-use-current-task"]').trigger('click')
    await wrapper.get('[data-testid="training-snapshot-name"]').setValue('round-2')
    await wrapper.get('[data-testid="training-create-snapshot"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="training-menu-message"]').text()).toContain('round-2')

    await wrapper.get('[data-testid="training-job-model-version"]').setValue('cls-v4')
    await wrapper.get('[data-testid="training-job-model-backbone"]').setValue('ViT_B_16')
    await wrapper.get('[data-testid="training-create-job"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="training-menu-message"]').text()).toContain('cls-v4')

    await wrapper.get('[data-testid="training-use-current-task-for-job"]').trigger('click')
    await wrapper.findAll('[data-testid="training-promote-and-enqueue"]')[1].trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="training-menu-message"]').text()).toContain('1')
  })

  it('manages prelabel task groups, enqueue, and cancel from the menu', async () => {
    const fetchMock = buildBaseFetchMock(async (url, options = {}) => {
      if (String(url) === '/api/prelabels/enqueue' && options.method === 'POST') {
        const body = JSON.parse(String(options.body || '{}'))
        expect(body.task_ids).toEqual(['PGC 17069', 'PGC 17070'])
        expect(body.model_version).toBe('cls-v2')
        expect(body.model_id).toBe('model-0')
        expect(body.model_backbone).toBe('ResNet18')
        expect(body.candidate_limit).toBe(15)
        expect(body.confidence_threshold).toBe(0.65)
        return jsonResponse({
          requested_count: 2,
          enqueued_count: 2,
          skipped_count: 0,
          job_ids: ['prelabel-job-2', 'prelabel-job-3'],
        })
      }
      if (String(url) === '/api/prelabels/jobs/cancel' && options.method === 'POST') {
        const body = JSON.parse(String(options.body || '{}'))
        expect(body.task_ids).toEqual(['PGC 17069', 'PGC 17070'])
        expect(body.statuses).toEqual(['queued', 'claimed'])
        return jsonResponse({
          requested_job_count: 0,
          requested_task_count: 2,
          cancelled_count: 2,
          jobs: [
            { job_id: 'prelabel-job-1', task_id: 'PGC 17069', status: 'cancelled' },
            { job_id: 'prelabel-job-2', task_id: 'PGC 17070', status: 'cancelled' },
          ],
        })
      }
      return null
    })
    globalThis.fetch = fetchMock

    const wrapper = mount(TrainingLoopMenu, {
      props: {
        activeTaskId: 'PGC 17069',
      },
    })

    await wrapper.get('[data-testid="header-training-menu-toggle"]').trigger('click')
    await flushPromises()

    await wrapper.get('[data-testid="prelabel-select-same-field"]').trigger('click')
    expect(wrapper.get('[data-testid="prelabel-selected-summary"]').text()).toContain('已选 2 个任务')

    await wrapper.get('[data-testid="prelabel-use-promoted-model"]').trigger('click')
    expect(wrapper.get('[data-testid="prelabel-model-version"]').element.value).toBe('cls-v2')
    expect(wrapper.get('[data-testid="prelabel-model-id"]').element.value).toBe('model-0')
    expect(wrapper.get('[data-testid="prelabel-model-backbone"]').element.value).toBe('ResNet18')
    await wrapper.get('[data-testid="prelabel-candidate-limit"]').setValue('15')
    await wrapper.get('[data-testid="prelabel-confidence-threshold"]').setValue('0.65')

    await wrapper.get('[data-testid="prelabel-bulk-enqueue"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[data-testid="prelabel-message"]').text()).toContain('2 个预标注任务')

    await wrapper.get('[data-testid="prelabel-bulk-cancel"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[data-testid="prelabel-message"]').text()).toContain('已取消 2 个预标注任务')
  })
})
