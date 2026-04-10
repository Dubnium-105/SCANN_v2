import { authFetch } from './authStore'

function formatValidationDetail(detail) {
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

export async function submitAnnotations(taskId, payload, options = {}, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const params = new URLSearchParams()
  if (options.clientId) {
    params.set('client_id', options.clientId)
  }
  if (typeof options.releaseAfterSave === 'boolean') {
    params.set('release_after_save', String(options.releaseAfterSave))
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/annotations/${encodedTaskId}${suffix}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    let detail = ''
    try {
      const payload = await response.json()
      detail = formatValidationDetail(payload?.detail)
    } catch {
      detail = ''
    }

    if (response.status === 401) {
      throw new Error('会话已过期，请重新登录')
    }
    if (response.status === 422) {
      throw new Error(detail ? `提交请求格式无效：${detail}` : '提交请求格式无效')
    }
    throw new Error(detail || 'Failed to submit annotations')
  }

  return response.json()
}
