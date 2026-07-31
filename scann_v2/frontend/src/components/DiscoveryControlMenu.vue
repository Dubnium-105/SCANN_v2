<template>
  <div class="relative">
    <button
      type="button"
      data-testid="header-discovery-menu-toggle"
      class="rounded border border-cyan-700 px-2 py-1 text-xs text-cyan-200"
      @click="toggle"
    >
      发现治理
    </button>
    <div
      v-if="open"
      data-testid="header-discovery-menu"
      class="absolute right-0 top-full z-40 mt-2 w-[32rem] max-w-[92vw] rounded border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300 shadow-xl"
    >
      <div class="flex items-center justify-between gap-2">
        <div>
          <p class="font-semibold text-slate-100">候选发现与模型治理</p>
          <p class="mt-0.5 text-[11px] text-slate-500">
            评估、审核反馈、主动学习与发布状态均保留版本历史
          </p>
        </div>
        <div class="flex gap-2">
          <button
            type="button"
            data-testid="discovery-refresh"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400 disabled:opacity-50"
            :disabled="loading"
            @click="refresh"
          >
            刷新
          </button>
          <button
            type="button"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400"
            @click="open = false"
          >
            关闭
          </button>
        </div>
      </div>

      <p v-if="error" data-testid="discovery-error" class="mt-2 text-rose-300">
        {{ error }}
      </p>
      <div class="mt-3 grid grid-cols-2 gap-2">
        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <p class="text-slate-500">评估运行</p>
          <p data-testid="discovery-evaluation-count" class="mt-1 text-lg text-slate-100">
            {{ evaluations.length }}
          </p>
          <p class="text-[10px] text-slate-500">
            最近：{{ evaluations[0]?.status || '无' }}
          </p>
        </section>
        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <p class="text-slate-500">主动学习批次</p>
          <p data-testid="discovery-batch-count" class="mt-1 text-lg text-slate-100">
            {{ batches.length }}
          </p>
          <p class="text-[10px] text-slate-500">
            最近：{{ batches[0]?.batch_name || '无' }}
          </p>
        </section>
        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <p class="text-slate-500">审核反馈事件</p>
          <p data-testid="discovery-feedback-count" class="mt-1 text-lg text-slate-100">
            {{ feedback.length }}
          </p>
          <p class="text-[10px] text-slate-500">
            最近：{{ feedback[0]?.outcome || '无' }}
          </p>
        </section>
        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <p class="text-slate-500">模型发布记录</p>
          <p data-testid="discovery-deployment-count" class="mt-1 text-lg text-slate-100">
            {{ deployments.length }}
          </p>
          <p class="text-[10px] text-slate-500">
            最近：{{ deployments[0]?.stage || '无' }}
          </p>
        </section>
      </div>
      <div class="mt-3 rounded border border-amber-800 bg-amber-950/20 p-2 text-[11px] text-amber-200">
        自动推广已关闭。shadow、canary、promote 与 rollback 只能通过显式审批接口执行。
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

import {
  fetchActiveLearningBatches,
  fetchEvaluations,
  fetchModelDeployments,
  fetchReviewFeedback,
} from '../services/discoveryApi'

const open = ref(false)
const loading = ref(false)
const error = ref('')
const evaluations = ref([])
const batches = ref([])
const feedback = ref([])
const deployments = ref([])

async function refresh() {
  loading.value = true
  error.value = ''
  try {
    const [
      evaluationResults,
      batchResults,
      feedbackResults,
      deploymentResults,
    ] = await Promise.all([
      fetchEvaluations(),
      fetchActiveLearningBatches(),
      fetchReviewFeedback(),
      fetchModelDeployments(),
    ])
    evaluations.value = evaluationResults
    batches.value = batchResults
    feedback.value = feedbackResults
    deployments.value = deploymentResults
  } catch (refreshError) {
    error.value = refreshError?.message || '加载发现治理状态失败'
  } finally {
    loading.value = false
  }
}

async function toggle() {
  open.value = !open.value
  if (open.value) {
    await refresh()
  }
}
</script>
