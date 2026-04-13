import { authFetch } from './authStore'

function normalizeTaskIds(taskIds) {
  if (Array.isArray(taskIds)) {
    return taskIds
      .map((item) => String(item || '').trim())
      .filter(Boolean)
  }
  return String(taskIds || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function normalizeStatuses(statuses) {
  if (Array.isArray(statuses)) {
    return statuses
      .map((item) => String(item || '').trim())
      .filter(Boolean)
  }
  return String(statuses || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function normalizePositiveInteger(value) {
  if (value === null || value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return null
  }
  const rounded = Math.round(parsed)
  return rounded > 0 ? rounded : null
}

function normalizeThreshold(value) {
  if (value === null || value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return null
  }
  return Math.max(0, Math.min(1, parsed))
}

function formatErrorDetail(detail) {
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (!item || typeof item !== 'object') {
          return ''
        }
        const location = Array.isArray(item.loc) ? item.loc.join('.') : ''
        const message = typeof item.msg === 'string' ? item.msg : ''
        return location ? `${location}: ${message}` : message
      })
      .filter(Boolean)
      .join('; ')
  }
  if (typeof detail === 'string') {
    return detail
  }
  return ''
}

async function readErrorMessage(response, fallback) {
  try {
    const payload = await response.json()
    const detail = formatErrorDetail(payload?.detail)
    if (detail) {
      return detail
    }
  } catch {
    // Ignore malformed payloads and fall back to caller-provided text.
  }
  return fallback
}

export async function fetchTaskPrelabel(taskId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const response = await fetchImpl(`/api/prelabels/${encodedTaskId}`)
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load AI prelabel'))
  }
  return response.json()
}

export async function enqueuePrelabels(options = {}, fetchImpl = authFetch) {
  const taskIds = normalizeTaskIds(options.taskIds)
  const modelVersion = String(options.modelVersion || '').trim()
  const modelId = String(options.modelId || '').trim()
  const modelBackbone = String(options.modelBackbone || '').trim()
  const candidateLimit = normalizePositiveInteger(options.candidateLimit)
  const confidenceThreshold = normalizeThreshold(options.confidenceThreshold)
  if (!modelVersion) {
    throw new Error('当前缺少模型版本，无法创建预标注任务')
  }

  const response = await fetchImpl('/api/prelabels/enqueue', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_version: modelVersion,
      model_id: modelId || null,
      model_backbone: modelBackbone || null,
      candidate_limit: candidateLimit,
      confidence_threshold: confidenceThreshold,
      task_ids: taskIds,
      priority: Number.isFinite(Number(options.priority)) ? Number(options.priority) : 100,
      force: options.force !== false,
    }),
  })

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, '请求预标注任务失败'))
  }
  return response.json()
}

export async function enqueueTaskPrelabel(taskId, options = {}, fetchImpl = authFetch) {
  const normalizedTaskId = String(taskId || '').trim()
  if (!normalizedTaskId) {
    throw new Error('当前任务不存在，无法重新生成 AI 草稿')
  }
  return enqueuePrelabels(
    {
      ...options,
      taskIds: [normalizedTaskId],
    },
    fetchImpl,
  )
}

export async function fetchPrelabelJobs(options = {}, fetchImpl = authFetch) {
  const params = new URLSearchParams()
  if (Number.isFinite(Number(options.limit)) && Number(options.limit) > 0) {
    params.set('limit', String(Math.round(Number(options.limit))))
  }
  const statuses = normalizeStatuses(options.statuses)
  if (statuses.length > 0) {
    params.set('statuses', statuses.join(','))
  }
  const taskIds = normalizeTaskIds(options.taskIds)
  if (taskIds.length > 0) {
    params.set('task_ids', taskIds.join(','))
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/prelabels/jobs${suffix}`)
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load prelabel jobs'))
  }
  return response.json()
}

export async function fetchPrelabelWorkers(options = {}, fetchImpl = authFetch) {
  const params = new URLSearchParams()
  if (Number.isFinite(Number(options.limit)) && Number(options.limit) > 0) {
    params.set('limit', String(Math.round(Number(options.limit))))
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/prelabels/workers${suffix}`)
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load prelabel workers'))
  }
  return response.json()
}

export async function cancelPrelabelJobs(options = {}, fetchImpl = authFetch) {
  const response = await fetchImpl('/api/prelabels/jobs/cancel', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      job_ids: Array.isArray(options.jobIds)
        ? options.jobIds.map((item) => String(item || '').trim()).filter(Boolean)
        : [],
      task_ids: normalizeTaskIds(options.taskIds),
      statuses: normalizeStatuses(options.statuses),
      reason: String(options.reason || '').trim() || null,
    }),
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to cancel prelabel jobs'))
  }
  return response.json()
}
