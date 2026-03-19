import { mount } from '@vue/test-utils'
import App from '../../App.vue'

describe('App', () => {
  it('renders title text', () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          CanvasPanel: true,
          InspectorPanel: true,
        },
      },
    })
    expect(wrapper.text()).toContain('SCANN Native Annotation')
  })
})
