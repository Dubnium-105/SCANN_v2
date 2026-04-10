import { authFetch } from './authStore'

async function readErrorMessage(response, fallback) {
  try {
    const payload = await response.json()
    if (typeof payload?.detail === 'string') {
      return payload.detail
    }
    if (typeof payload?.error_message === 'string') {
      return payload.error_message
    }
  } catch {
    // Ignore invalid response bodies and use the caller's fallback.
  }
  return fallback
}

export async function fetchAnnotationSyncStatus(fetchImpl = authFetch) {
  const response = await fetchImpl('/api/annotation-sync/status')
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load annotation sync status'))
  }
  return response.json()
}

export async function runAnnotationSync(options = {}, fetchImpl = authFetch) {
  const params = new URLSearchParams()
  if (options.full) {
    params.set('full', 'true')
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/annotation-sync/run${suffix}`, {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to run annotation sync'))
  }
  return response.json()
}
