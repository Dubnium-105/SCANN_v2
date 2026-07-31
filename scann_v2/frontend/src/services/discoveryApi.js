import { authFetch } from './authStore'

async function readPayload(response, fallback) {
  if (response.ok) {
    return response.json()
  }
  try {
    const payload = await response.json()
    throw new Error(payload?.detail || fallback)
  } catch (error) {
    if (error instanceof Error && error.message !== fallback) {
      throw error
    }
    throw new Error(fallback)
  }
}

function jsonOptions(payload) {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  }
}

export async function fetchEvaluations(fetchImpl = authFetch) {
  return readPayload(
    await fetchImpl('/api/evaluations'),
    '无法加载评估运行',
  )
}

export async function createEvaluation(payload, fetchImpl = authFetch) {
  return readPayload(
    await fetchImpl('/api/evaluations', jsonOptions(payload)),
    '无法创建评估运行',
  )
}

export async function fetchActiveLearningBatches(fetchImpl = authFetch) {
  return readPayload(
    await fetchImpl('/api/active-learning/batches'),
    '无法加载主动学习批次',
  )
}

export async function createActiveLearningBatch(payload, fetchImpl = authFetch) {
  return readPayload(
    await fetchImpl('/api/active-learning/batches', jsonOptions(payload)),
    '无法创建主动学习批次',
  )
}

export async function fetchReviewFeedback(fetchImpl = authFetch) {
  return readPayload(
    await fetchImpl('/api/review-feedback'),
    '无法加载审核反馈',
  )
}

export async function fetchModelDeployments(fetchImpl = authFetch) {
  return readPayload(
    await fetchImpl('/api/training/model-deployments'),
    '无法加载模型发布记录',
  )
}

async function deploymentAction(modelId, action, payload, fetchImpl) {
  const normalizedModelId = String(modelId || '').trim()
  if (!normalizedModelId) {
    throw new Error('Model ID is required')
  }
  const path = `/api/training/models/${encodeURIComponent(normalizedModelId)}/deployments/${action}`
  return readPayload(
    await fetchImpl(path, jsonOptions(payload)),
    `无法执行模型发布动作: ${action}`,
  )
}

export function startShadowDeployment(modelId, payload, fetchImpl = authFetch) {
  return deploymentAction(modelId, 'shadow', payload, fetchImpl)
}

export function startCanaryDeployment(modelId, payload, fetchImpl = authFetch) {
  return deploymentAction(modelId, 'canary', payload, fetchImpl)
}

export function promoteDeployment(modelId, payload, fetchImpl = authFetch) {
  return deploymentAction(modelId, 'promote', payload, fetchImpl)
}

export function rollbackDeployment(modelId, payload, fetchImpl = authFetch) {
  return deploymentAction(modelId, 'rollback', payload, fetchImpl)
}
