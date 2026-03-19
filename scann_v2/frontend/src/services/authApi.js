import { clearAuthSession, setAuthSession } from './authStore'

export async function loginWithPassword(username, password, fetchImpl = fetch) {
  const response = await fetchImpl('/api/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  })

  if (!response.ok) {
    throw new Error('Invalid username or password')
  }

  const payload = await response.json()
  setAuthSession(payload.access_token)
  return payload
}

export function logout() {
  clearAuthSession()
}
