import { authFetch } from './authStore'

export async function fetchTasks(fetchImpl = authFetch) {
  const response = await fetchImpl('/api/tasks')
  if (!response.ok) {
    throw new Error('Failed to load tasks')
  }
  return response.json()
}
