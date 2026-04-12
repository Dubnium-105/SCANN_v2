import { authFetch } from './authStore'

export async function fetchTaskPrelabel(taskId, fetchImpl = authFetch) {
  const encodedTaskId = encodeURIComponent(taskId)
  const response = await fetchImpl(`/api/prelabels/${encodedTaskId}`)
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    throw new Error('Failed to load AI prelabel')
  }
  return response.json()
}

function formatErrorDetail(detail) {
  if (typeof detail === 'string') {
    return detail
  }
  return ''
}

export async function enqueueTaskPrelabel(taskId, options = {}, fetchImpl = authFetch) {
  const normalizedTaskId = String(taskId || '').trim()
  const modelVersion = String(options.modelVersion || '').trim()
  if (!normalizedTaskId) {
    throw new Error('任务不存在，无法重新生成 AI 草稿')
  }
  if (!modelVersion) {
    throw new Error('当前任务缺少模型版本，无法重新生成 AI 草稿')
  }

  const response = await fetchImpl('/api/prelabels/enqueue', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_version: modelVersion,
      task_ids: [normalizedTaskId],
      priority: Number.isFinite(Number(options.priority)) ? Number(options.priority) : 100,
      force: options.force !== false,
    }),
  })

  if (!response.ok) {
    let detail = ''
    try {
      const payload = await response.json()
      detail = formatErrorDetail(payload?.detail)
    } catch {
      detail = ''
    }
    if (response.status === 401) {
      throw new Error('会话已过期，请重新登录')
    }
    if (response.status === 403) {
      throw new Error(detail || '只有管理员可以重新生成 AI 草稿')
    }
    if (response.status === 422) {
      throw new Error(detail || '重新生成 AI 草稿的请求无效')
    }
    throw new Error(detail || '请求重新生成 AI 草稿失败')
  }

  return response.json()
}
