import {
  authFetch,
  authState,
  clearAuthSession,
  setAuthSession,
} from '../authStore'

function createToken(payload = {}) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString('base64url')
  return `${encode({ alg: 'none', typ: 'JWT' })}.${encode(payload)}.signature`
}

describe('authStore', () => {
  beforeEach(() => {
    clearAuthSession()
    localStorage.clear()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('blocks authenticated requests when token has already expired', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-10T00:00:00.000Z'))

    setAuthSession(createToken({
      sub: 'annotator',
      role: 'annotator',
      exp: Math.floor(Date.now() / 1000) - 1,
    }))

    const fetchImpl = vi.fn()

    await expect(authFetch('/api/tasks', {}, fetchImpl)).rejects.toThrow('Session expired. Please log in again')
    expect(fetchImpl).not.toHaveBeenCalled()
    expect(authState.token).toBe('')
  })

  it('clears local session after backend returns 401', async () => {
    setAuthSession(createToken({
      sub: 'annotator',
      role: 'annotator',
      exp: Math.floor(Date.now() / 1000) + 3600,
    }))

    const response = await authFetch('/api/tasks', {}, vi.fn(async () => ({
      ok: false,
      status: 401,
    })))

    expect(response.status).toBe(401)
    expect(authState.token).toBe('')
  })
})
