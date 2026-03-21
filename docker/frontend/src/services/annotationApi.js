import { authFetch } from './authStore'

export async function submitAnnotations(taskId, payload, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const response = await fetchImpl(`/api/annotations/${encodedTaskId}`, {
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