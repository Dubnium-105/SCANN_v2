import { reactive } from 'vue'

const STORAGE_KEY = 'scann_native_auth'

function decodeJwtPayload(token) {
  try {
    const payloadPart = token.split('.')[1]
    if (!payloadPart) return null
    const normalized = payloadPart.replace(/-/g, '+').replace(/_/g, '/')
    const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4)
    return JSON.parse(atob(padded))
  } catch {
    return null
  }
}

function loadInitialState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return { token: '', username: '', role: '' }
    }
    const parsed = JSON.parse(raw)
    return {
      token: parsed.token || '',
      username: parsed.username || '',
      role: parsed.role || '',
    }
  } catch {
    return { token: '', username: '', role: '' }
  }
}

export const authState = reactive(loadInitialState())

function persistState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      token: authState.token,
      username: authState.username,
      role: authState.role,
    }),
  )
}

export function setAuthSession(token) {
  const payload = decodeJwtPayload(token)
  authState.token = token
  authState.username = payload?.sub || ''
  authState.role = payload?.role || ''
  persistState()
}

export function clearAuthSession() {
  authState.token = ''
  authState.username = ''
  authState.role = ''
  persistState()
}

export function isAuthenticated() {
  return Boolean(authState.token)
}

export function getAuthToken() {
  return authState.token
}

export async function authFetch(url, options = {}, fetchImpl = fetch) {
  const headers = {
    ...(options.headers || {}),
  }
  const token = getAuthToken()
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  return fetchImpl(url, {
    ...options,
    headers,
  })
}
