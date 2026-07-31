import { flushPromises, mount } from '@vue/test-utils'

import DiscoveryControlMenu from '../DiscoveryControlMenu.vue'


function jsonResponse(payload) {
  return {
    ok: true,
    status: 200,
    json: async () => payload,
  }
}


describe('DiscoveryControlMenu', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('loads the discovery lifecycle overview without mutating state', async () => {
    globalThis.fetch = vi.fn(async (url) => {
      if (url === '/api/evaluations') {
        return jsonResponse([{ run_id: 'evaluation-1', status: 'completed' }])
      }
      if (url === '/api/active-learning/batches') {
        return jsonResponse([{ batch_id: 'batch-1', batch_name: 'round-1' }])
      }
      if (url === '/api/review-feedback') {
        return jsonResponse([{ event_id: 'review-1', outcome: 'partial_accept' }])
      }
      if (url === '/api/training/model-deployments') {
        return jsonResponse([{ deployment_id: 'deployment-1', stage: 'shadow' }])
      }
      throw new Error(`Unexpected URL: ${url}`)
    })

    const wrapper = mount(DiscoveryControlMenu)
    await wrapper.get('[data-testid="header-discovery-menu-toggle"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('[data-testid="discovery-evaluation-count"]').text()).toBe('1')
    expect(wrapper.get('[data-testid="discovery-batch-count"]').text()).toBe('1')
    expect(wrapper.get('[data-testid="discovery-feedback-count"]').text()).toBe('1')
    expect(wrapper.get('[data-testid="discovery-deployment-count"]').text()).toBe('1')
    expect(wrapper.text()).toContain('自动推广已关闭')
    expect(globalThis.fetch).toHaveBeenCalledTimes(4)
    expect(
      globalThis.fetch.mock.calls.every(([, options]) => !options?.method),
    ).toBe(true)
  })
})
