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

export function getAuthTokenExpiresAtMs(token = authState.token) {
  const payload = decodeJwtPayload(token)
  const exp = Number(payload?.exp)
  return Number.isFinite(exp) ? exp * 1000 : 0
}

export function isAuthTokenExpired(token = authState.token, bufferMs = 0) {
  const expiresAtMs = getAuthTokenExpiresAtMs(token)
  if (!expiresAtMs) {
    return false
  }
  return Date.now() + Math.max(0, Number(bufferMs) || 0) >= expiresAtMs
}

export async function authFetch(url, options = {}, fetchImpl = fetch) {
  const headers = {
    ...(options.headers || {}),
  }
  const token = getAuthToken()
  if (token) {
    if (isAuthTokenExpired(token)) {
      clearAuthSession()
      throw new Error('Session expired. Please log in again')
    }
    headers.Authorization = `Bearer ${token}`
  }
  const response = await fetchImpl(url, {
    ...options,
    headers,
  })
  if (response.status === 401) {
    clearAuthSession()
  }
  return response
}
