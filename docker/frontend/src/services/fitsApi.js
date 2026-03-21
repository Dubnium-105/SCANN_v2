import { authFetch } from './authStore'

export async function fetchFitsArrayBuffer(relativePath, fetchImpl = authFetch) {
  const encodedPath = relativePath
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')

  const response = await fetchImpl(`/api/fits/${encodedPath}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch FITS: ${relativePath}`)
  }

  return response.arrayBuffer()
}
