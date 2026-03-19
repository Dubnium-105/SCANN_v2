import { authFetch } from './authStore'

export async function fetchAnnotationHistory(taskId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const response = await fetchImpl(`/api/annotations/${encodedTaskId}/history`)
  if (!response.ok) {
    throw new Error('Failed to load annotation history')
  }
  return response.json()
}
