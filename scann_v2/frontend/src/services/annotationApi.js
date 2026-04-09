import { authFetch } from './authStore'

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
    throw new Error('Failed to submit annotations')
  }

  return response.json()
}
