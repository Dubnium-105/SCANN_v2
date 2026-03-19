import { mount } from '@vue/test-utils'

import HeaderBar from '../HeaderBar.vue'

describe('HeaderBar', () => {
  it('shows username in header', () => {
    const wrapper = mount(HeaderBar, {
      props: {
        username: 'annotator',
      },
    })

    expect(wrapper.get('[data-testid="header-username"]').text()).toContain('annotator')
  })
})
