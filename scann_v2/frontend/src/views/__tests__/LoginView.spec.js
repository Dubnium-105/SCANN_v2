import { flushPromises, mount } from '@vue/test-utils'

import LoginView from '../LoginView.vue'
import { authState, clearAuthSession } from '../../services/authStore'

const push = vi.fn()

vi.mock('vue-router', () => ({
  useRouter: () => ({ push }),
}))

describe('LoginView', () => {
  beforeEach(() => {
    push.mockReset()
    clearAuthSession()
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        access_token:
          'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbm5vdGF0b3IiLCJyb2xlIjoiYW5ub3RhdG9yIn0.signature',
      }),
    }))
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('submits login form, stores token, and navigates to annotation page', async () => {
    const wrapper = mount(LoginView, {
    })

    await wrapper.get('[data-testid="login-username"]').setValue('annotator')
    await wrapper.get('[data-testid="login-password"]').setValue('scann123')
    await wrapper.get('[data-testid="login-submit"]').trigger('submit')
    await flushPromises()

    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/login',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(authState.token).toBeTruthy()
    expect(authState.username).toBe('annotator')
    expect(push).toHaveBeenCalledWith({ name: 'annotation' })
  })
})
