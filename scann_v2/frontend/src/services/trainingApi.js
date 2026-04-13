import { authFetch } from './authStore'

async function readErrorMessage(response, fallback) {
  try {
    const payload = await response.json()
    if (typeof payload?.detail === 'string' && payload.detail.trim()) {
      return payload.detail.trim()
    }
    if (typeof payload?.error_message === 'string' && payload.error_message.trim()) {
      return payload.error_message.trim()
    }
  } catch {
    // Ignore malformed payloads and fall back to caller-provided text.
  }
  return fallback
}

function withLimit(path, limit) {
  const normalizedLimit = Number(limit)
  if (!Number.isFinite(normalizedLimit) || normalizedLimit <= 0) {
    return path
  }
  const params = new URLSearchParams({ limit: String(Math.round(normalizedLimit)) })
  return `${path}?${params.toString()}`
}

function parseTaskIds(taskIds) {
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

export async function fetchTrainingSnapshots(options = {}, fetchImpl = authFetch) {
  const response = await fetchImpl(withLimit('/api/training/snapshots', options.limit))
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load training snapshots'))
  }
  return response.json()
}

export async function createTrainingSnapshot(payload, fetchImpl = authFetch) {
  const response = await fetchImpl('/api/training/snapshots', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      snapshot_name: String(payload?.snapshotName || '').trim() || null,
      task_ids: parseTaskIds(payload?.taskIds),
      metadata: payload?.metadata && typeof payload.metadata === 'object' ? payload.metadata : {},
    }),
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to create training snapshot'))
  }
  return response.json()
}

export async function fetchTrainingJobs(options = {}, fetchImpl = authFetch) {
  const response = await fetchImpl(withLimit('/api/training/jobs', options.limit))
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load training jobs'))
  }
  return response.json()
}

export async function createTrainingJob(payload, fetchImpl = authFetch) {
  const body = {
    snapshot_id: String(payload?.snapshotId || '').trim() || null,
    snapshot_name: String(payload?.snapshotName || '').trim() || null,
    snapshot_task_ids: parseTaskIds(payload?.snapshotTaskIds),
    snapshot_metadata: payload?.snapshotMetadata && typeof payload.snapshotMetadata === 'object'
      ? payload.snapshotMetadata
      : {},
    task_type: String(payload?.taskType || 'classification').trim() || 'classification',
    model_version: String(payload?.modelVersion || '').trim(),
    model_id: String(payload?.modelId || '').trim() || null,
    model_backbone: String(payload?.modelBackbone || '').trim(),
    train_config: payload?.trainConfig && typeof payload.trainConfig === 'object' ? payload.trainConfig : {},
    priority: Number.isFinite(Number(payload?.priority)) ? Number(payload.priority) : 100,
    promote_on_success: payload?.promoteOnSuccess === true,
    enqueue_prelabels_on_success: payload?.enqueuePrelabelsOnSuccess === true,
    prelabel_task_ids: parseTaskIds(payload?.prelabelTaskIds),
    force_prelabel: payload?.forcePrelabel === true,
  }

  const response = await fetchImpl('/api/training/jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to create training job'))
  }
  return response.json()
}

export async function fetchTrainingRuns(options = {}, fetchImpl = authFetch) {
  const response = await fetchImpl(withLimit('/api/training/runs', options.limit))
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load training runs'))
  }
  return response.json()
}

export async function fetchTrainingModels(options = {}, fetchImpl = authFetch) {
  const params = new URLSearchParams()
  if (options.taskType) {
    params.set('task_type', String(options.taskType))
  }
  if (Number.isFinite(Number(options.limit)) && Number(options.limit) > 0) {
    params.set('limit', String(Math.round(Number(options.limit))))
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/training/models${suffix}`)
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load training models'))
  }
  return response.json()
}

export async function fetchPromotedTrainingModel(options = {}, fetchImpl = authFetch) {
  const params = new URLSearchParams()
  params.set('task_type', String(options.taskType || 'classification'))
  const response = await fetchImpl(`/api/training/models/promoted?${params.toString()}`)
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to load promoted model'))
  }
  return response.json()
}

export async function promoteTrainingModel(modelId, options = {}, fetchImpl = authFetch) {
  const normalizedModelId = String(modelId || '').trim()
  if (!normalizedModelId) {
    throw new Error('Model ID is required')
  }
  const params = new URLSearchParams()
  if (options.enqueuePrelabels) {
    params.set('enqueue_prelabels', 'true')
  }
  if (options.forcePrelabel) {
    params.set('force_prelabel', 'true')
  }
  const taskIds = parseTaskIds(options.taskIds)
  if (taskIds.length > 0) {
    params.set('task_ids', taskIds.join(','))
  }
  const suffix = params.size ? `?${params.toString()}` : ''
  const response = await fetchImpl(`/api/training/models/${encodeURIComponent(normalizedModelId)}/promote${suffix}`, {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(await readErrorMessage(response, 'Failed to promote model'))
  }
  return response.json()
}
