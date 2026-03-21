import { authFetch } from './authStore'

export async function fetchAnnotationHistory(taskId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const response = await fetchImpl(`/api/annotations/${encodedTaskId}/history`)
  if (!response.ok) {
    throw new Error('Failed to load annotation history')
  }
  return response.json()
}

export async function fetchAnnotationRevision(taskId, revisionId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const encodedRevisionId = encodeURIComponent(revisionId)
  const response = await fetchImpl(`/api/annotations/${encodedTaskId}/history/${encodedRevisionId}`)
  if (!response.ok) {
    throw new Error('Failed to load annotation revision detail')
  }
  return response.json()
}

export async function rollbackAnnotationRevision(taskId, revisionId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const encodedRevisionId = encodeURIComponent(revisionId)
  const response = await fetchImpl(`/api/annotations/${encodedTaskId}/rollback/${encodedRevisionId}`, {
    method: 'POST',
  })
  if (!response.ok) {
    if (response.status === 403) {
      throw new Error('Only admin can rollback revisions')
    }
    throw new Error('Failed to rollback revision')
  }
  return response.json()
}
