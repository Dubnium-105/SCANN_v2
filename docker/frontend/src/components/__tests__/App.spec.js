import { mount } from '@vue/test-utils'
import App from '../../App.vue'

describe('App', () => {
  it('renders router outlet', () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          RouterView: { template: '<div data-testid="router-outlet" />' },
        },
      },
    })
    expect(wrapper.find('[data-testid="router-outlet"]').exists()).toBe(true)
  })
})
