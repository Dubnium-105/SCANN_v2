import { authFetch } from './authStore'

function formatErrorDetail(detail) {
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (!item || typeof item !== 'object') {
          return ''
        }
        const location = Array.isArray(item.loc) ? item.loc.join('.') : ''
        const message = typeof item.msg === 'string' ? item.msg : ''
        return location ? `${location}: ${message}` : message
      })
      .filter(Boolean)
      .join('; ')
  }
  if (typeof detail === 'string') {
    return detail
  }
  return ''
}

async function readErrorMessage(response, fallback) {
  try {
    const payload = await response.json()
    const detail = formatErrorDetail(payload?.detail)
    if (detail) {
      return detail
    }
  } catch {
    // Ignore malformed payloads and fall back to caller-provided text.
  }
  return fallback
}

export async function preprocessDataset(fetchImpl = authFetch) {
  const response = await fetchImpl('/api/dataset/preprocess', {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, '目录检测失败'))
  }
  return response.json()
}

export async function fetchDatasetStats(options = {}, fetchImpl = authFetch) {
  if (typeof options === 'function') {
    fetchImpl = options
    options = {}
  }
  const params = new URLSearchParams()
  if (options.noCache) {
    params.set('fresh', '1')
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/dataset/stats${suffix}`, {
    headers: options.noCache ? { 'Cache-Control': 'no-store' } : {},
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, '获取数据集统计失败'))
  }
  return response.json()
}
