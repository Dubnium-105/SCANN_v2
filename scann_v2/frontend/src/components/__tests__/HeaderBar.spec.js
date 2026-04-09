import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'

import HeaderBar from '../HeaderBar.vue'

function createToken(payload = {}) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString('base64url')
  return `${encode({ alg: 'none', typ: 'JWT' })}.${encode(payload)}.signature`
}

describe('HeaderBar', () => {
  afterEach(() => {
    vi.useRealTimers()
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
})
