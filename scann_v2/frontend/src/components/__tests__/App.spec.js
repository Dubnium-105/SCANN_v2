import { mount } from '@vue/test-utils'
import App from '../../App.vue'

describe('App', () => {
  it('renders title text', () => {
    const wrapper = mount(App)
    expect(wrapper.text()).toContain('SCANN Native Annotation')
  })
})
