import { flushPromises, mount } from '@vue/test-utils'

import TrainingLoopMenu from '../TrainingLoopMenu.vue'

function jsonResponse(payload, ok = true, status = 200) {
  return {
    ok,
    status,
    json: async () => payload,
  }
}

describe('TrainingLoopMenu', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('loads training control state for admin workflow', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      if (String(url).startsWith('/api/training/snapshots')) {
        return jsonResponse([
          {
            snapshot_id: 'snapshot-1',
            snapshot_name: 'round-1',
            annotation_count: 12,
            task_count: 10,
          },
        ])
      }
      if (String(url).startsWith('/api/training/jobs')) {
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
      if (String(url).startsWith('/api/training/models')) {
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
      throw new Error(`Unexpected URL: ${url}`)
    })

    const wrapper = mount(TrainingLoopMenu, {
      props: {
        activeTaskId: 'PGC 17069',
      },
    })

    await wrapper.get('[data-testid="header-training-menu-toggle"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="header-training-menu"]').text()).toContain('round-1')
    expect(wrapper.get('[data-testid="training-promoted-model"]').text()).toContain('cls-v2')
    expect(wrapper.text()).toContain('cls-v3 / ViT_B_16')
  })

  it('creates snapshots and training jobs, then can promote a model with targeted prelabels', async () => {
    const fetchMock = vi.fn(async (url, options = {}) => {
      if (String(url).startsWith('/api/training/snapshots') && !options.method) {
        return jsonResponse([])
      }
      if (String(url).startsWith('/api/training/jobs') && !options.method) {
        return jsonResponse([])
      }
      if (String(url).startsWith('/api/training/runs')) {
        return jsonResponse([])
      }
      if (String(url).startsWith('/api/training/models/promoted')) {
        return {
          ok: false,
          status: 404,
          json: async () => ({ detail: 'Promoted model not found' }),
        }
      }
      if (String(url).startsWith('/api/training/models') && !options.method) {
        return jsonResponse([
          {
            model_id: 'model-1',
            model_version: 'cls-v3',
            model_backbone: 'ViT_B_16',
            is_promoted: false,
            metrics: { f1: 0.93 },
          },
        ])
      }
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
      throw new Error(`Unexpected URL: ${url}`)
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

    expect(wrapper.get('[data-testid="training-menu-message"]').text()).toContain('已创建训练快照 round-2')

    await wrapper.get('[data-testid="training-job-model-version"]').setValue('cls-v4')
    await wrapper.get('[data-testid="training-job-model-backbone"]').setValue('ViT_B_16')
    await wrapper.get('[data-testid="training-create-job"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="training-menu-message"]').text()).toContain('已创建训练作业 cls-v4')

    await wrapper.get('[data-testid="training-use-current-task-for-job"]').trigger('click')
    await wrapper.get('[data-testid="training-promote-and-enqueue"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="training-menu-message"]').text()).toContain('已推广模型并排入 1 个预标注任务')
  })
})
