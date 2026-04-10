import { submitAnnotations } from '../annotationApi'

describe('annotationApi', () => {
  it('surfaces validation detail for 422 responses', async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [
          {
            loc: ['body', 'bucket'],
            msg: 'Field required',
          },
        ],
      }),
    }))

    await expect(
      submitAnnotations('PGC 17069', { annotations: [] }, {}, fetchImpl),
    ).rejects.toThrow('body.bucket: Field required')
  })

  it('turns 401 responses into re-login guidance', async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      status: 401,
      json: async () => ({
        detail: 'Invalid token',
      }),
    }))

    await expect(
      submitAnnotations('PGC 17069', { annotations: [] }, {}, fetchImpl),
    ).rejects.toThrow('会话已过期，请重新登录')
  })
})
