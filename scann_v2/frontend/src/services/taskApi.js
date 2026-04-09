import { authFetch } from './authStore'

const CLIENT_ID_STORAGE_KEY = 'scann_native_client_id'
let memoryClientId = ''

function getClientIdStorage() {
  try {
    if (typeof sessionStorage !== 'undefined') {
      return sessionStorage
    }
  } catch {
    // Ignore storage access failures and fall back to in-memory id.
  }
  return null
}

function createClientId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `client-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function getTaskClientId() {
  const storage = getClientIdStorage()
  if (!storage) {
    if (!memoryClientId) {
      memoryClientId = createClientId()
    }
    return memoryClientId
  }

  const existing = storage.getItem(CLIENT_ID_STORAGE_KEY)
  if (existing) {
    return existing
  }

  const created = createClientId()
  storage.setItem(CLIENT_ID_STORAGE_KEY, created)
  return created
}

export async function fetchTasks(fetchImpl = authFetch) {
  const response = await fetchImpl('/api/tasks')
  if (!response.ok) {
    throw new Error('Failed to load tasks')
  }
  return response.json()
}

export async function claimNextTask(clientId, fetchImpl = authFetch) {
  const params = new URLSearchParams({ client_id: clientId })
  const response = await fetchImpl(`/api/tasks/next?${params.toString()}`)
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('No available task')
    }
    if (response.status === 409) {
      throw new Error('Task locked by another client')
    }
    throw new Error('Failed to claim next task')
  }
  return response.json()
}

export async function claimTask(taskId, clientId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const params = new URLSearchParams({ client_id: clientId })
  const response = await fetchImpl(`/api/tasks/${encodedTaskId}/claim?${params.toString()}`, {
    method: 'POST',
  })
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('Task not found')
    }
    if (response.status === 409) {
      throw new Error('Task locked by another client')
    }
    throw new Error('Failed to claim task')
  }
  return response.json()
}

export async function heartbeatTask(taskId, clientId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const params = new URLSearchParams({ client_id: clientId })
  const response = await fetchImpl(`/api/tasks/${encodedTaskId}/heartbeat?${params.toString()}`, {
    method: 'POST',
  })
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('Task lock not found')
    }
    if (response.status === 409) {
      throw new Error('Task locked by another client')
    }
    throw new Error('Failed to refresh task lock')
  }
  return response.json()
}

export async function releaseTask(taskId, clientId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const params = new URLSearchParams({ client_id: clientId })
  const response = await fetchImpl(`/api/tasks/${encodedTaskId}/release?${params.toString()}`, {
    method: 'POST',
  })
  if (!response.ok) {
    if (response.status === 404) {
      return { released: false }
    }
    if (response.status === 409) {
      throw new Error('Task locked by another client')
    }
    throw new Error('Failed to release task')
  }
  return response.json()
}
