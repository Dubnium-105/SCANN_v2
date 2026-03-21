import { flushPromises, mount } from '@vue/test-utils'

import RegisterView from '../RegisterView.vue'
import { authState, clearAuthSession } from '../../services/authStore'

const push = vi.fn()

vi.mock('vue-router', () => ({
  useRouter: () => ({ push }),
}))

describe('RegisterView', () => {
  beforeEach(() => {
    push.mockReset()
    clearAuthSession()
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        access_token:
          'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJuZXdfdXNlciIsInJvbGUiOiJhbm5vdGF0b3IifQ.signature',
      }),
    }))
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('submits register form and navigates to annotation page', async () => {
    const wrapper = mount(RegisterView)

    await wrapper.get('[data-testid="register-username"]').setValue('new_user')
    await wrapper.get('[data-testid="register-password"]').setValue('newpass123')
    await wrapper.get('[data-testid="register-confirm-password"]').setValue('newpass123')
    await wrapper.get('[data-testid="register-submit"]').trigger('submit')
    await flushPromises()

    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/register',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(authState.token).toBeTruthy()
    expect(authState.username).toBe('new_user')
    expect(push).toHaveBeenCalledWith({ name: 'annotation' })
  })

  it('shows validation error when passwords do not match', async () => {
    const wrapper = mount(RegisterView)

    await wrapper.get('[data-testid="register-username"]').setValue('new_user')
    await wrapper.get('[data-testid="register-password"]').setValue('newpass123')
    await wrapper.get('[data-testid="register-confirm-password"]').setValue('x')
    await wrapper.get('[data-testid="register-submit"]').trigger('submit')
    await flushPromises()

    expect(globalThis.fetch).not.toHaveBeenCalled()
    expect(wrapper.get('[data-testid="register-error"]').text()).toContain('Passwords do not match')
  })
})
