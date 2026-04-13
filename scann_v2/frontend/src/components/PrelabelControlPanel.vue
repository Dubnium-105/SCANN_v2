<template>
  <section class="rounded border border-slate-800 bg-slate-900/70 p-3" data-testid="prelabel-control-panel">
    <div class="flex items-center justify-between gap-2">
      <div>
        <p class="font-medium text-slate-100">预标注管理</p>
        <p class="mt-0.5 text-[10px] text-slate-500">
          批量选择任务、排队预标注、取消现有任务，并查看 worker 在线状态。
        </p>
      </div>
      <button
        type="button"
        data-testid="prelabel-refresh"
        class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400 disabled:opacity-50"
        :disabled="loading || submitting"
        @click="refreshAll"
      >
        刷新
      </button>
    </div>

    <p v-if="message" data-testid="prelabel-message" class="mt-2 text-[11px] text-emerald-300">
      {{ message }}
    </p>
    <p v-if="error" data-testid="prelabel-error" class="mt-2 text-[11px] text-rose-300">
      {{ error }}
    </p>

    <div class="mt-3 grid gap-2">
      <div class="grid grid-cols-3 gap-2">
        <label class="grid gap-1">
          <span class="text-[11px] text-slate-500">模型版本</span>
          <input
            v-model="modelForm.modelVersion"
            data-testid="prelabel-model-version"
            type="text"
            class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
            placeholder="detector-v3"
          >
        </label>
        <label class="grid gap-1">
          <span class="text-[11px] text-slate-500">模型 ID</span>
          <input
            v-model="modelForm.modelId"
            data-testid="prelabel-model-id"
            type="text"
            class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
            placeholder="run-20260413-001"
          >
        </label>
        <label class="grid gap-1">
          <span class="text-[11px] text-slate-500">Backbone</span>
          <input
            v-model="modelForm.modelBackbone"
            data-testid="prelabel-model-backbone"
            type="text"
            class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
            placeholder="ViT_B_16"
          >
        </label>
      </div>

      <div class="grid gap-2 rounded border border-slate-800 bg-slate-950/60 p-2">
        <div class="flex flex-wrap gap-2">
          <button
            v-if="activeTaskId"
            type="button"
            data-testid="prelabel-select-current-task"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300"
            @click="applyTaskPreset('current')"
          >
            当前任务
          </button>
          <button
            type="button"
            data-testid="prelabel-select-same-field"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300 disabled:opacity-50"
            :disabled="!activeTaskFieldKey"
            @click="applyTaskPreset('same-field')"
          >
            同场区
          </button>
          <button
            type="button"
            data-testid="prelabel-select-same-capture"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300 disabled:opacity-50"
            :disabled="!activeTaskCaptureKey"
            @click="applyTaskPreset('same-capture')"
          >
            同观测批次
          </button>
          <button
            type="button"
            data-testid="prelabel-select-queued"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300"
            @click="applyTaskPreset('queued')"
          >
            AI 排队/处理中
          </button>
          <button
            type="button"
            data-testid="prelabel-select-failed"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300"
            @click="applyTaskPreset('failed')"
          >
            AI 失败/取消
          </button>
          <button
            type="button"
            data-testid="prelabel-select-missing"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300"
            @click="applyTaskPreset('missing')"
          >
            尚未生成
          </button>
          <button
            type="button"
            data-testid="prelabel-clear-selection"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400"
            @click="selectedTaskIds = []"
          >
            清空
          </button>
        </div>

        <div class="grid grid-cols-[1fr_auto_auto] gap-2">
          <label class="grid gap-1">
            <span class="text-[11px] text-slate-500">任务筛选</span>
            <input
              v-model="taskQuery"
              data-testid="prelabel-task-query"
              type="text"
              class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
              placeholder="按任务 ID / 场区 / 批次筛选"
            >
          </label>
          <button
            type="button"
            data-testid="prelabel-select-visible"
            class="self-end rounded border border-slate-700 px-2 py-1 text-[10px] text-slate-300"
            @click="selectVisibleTasks"
          >
            选择可见
          </button>
          <button
            type="button"
            data-testid="prelabel-use-promoted-model"
            class="self-end rounded border border-emerald-700 px-2 py-1 text-[10px] text-emerald-200 disabled:opacity-50"
            :disabled="!promotedModel"
            @click="syncModelFormFromPromoted(true)"
          >
            使用当前模型
          </button>
        </div>

        <p class="text-[10px] text-slate-400" data-testid="prelabel-selected-summary">
          已选 {{ selectedTaskIds.length }} 个任务。可见 {{ filteredTasks.length }} 个，当前队列 {{ queuedTaskCount }} 个，失败/取消 {{ failedTaskCount }} 个。
        </p>

        <div class="max-h-40 space-y-1 overflow-y-auto rounded border border-slate-800 bg-slate-950/40 p-2">
          <label
            v-for="task in filteredTasks"
            :key="task.task_id"
            class="flex items-start gap-2 rounded px-1 py-1 text-[11px]"
            :class="isTaskSelected(task.task_id) ? 'bg-sky-950/30' : ''"
          >
            <input
              :checked="isTaskSelected(task.task_id)"
              type="checkbox"
              @change="toggleTaskSelection(task.task_id)"
            >
            <div class="min-w-0 flex-1">
              <p class="truncate text-slate-200">{{ task.task_id }}</p>
              <p class="truncate text-slate-500">
                {{ task.field_name || task.field_key || '-' }} · {{ task.capture_key || '-' }} · {{ formatTaskPrelabelStatus(task.prelabel_status) }}
              </p>
            </div>
          </label>
          <p v-if="filteredTasks.length === 0" class="text-[11px] text-slate-500">
            没有匹配的任务。
          </p>
        </div>

        <div class="grid grid-cols-2 gap-2">
          <label class="flex items-center gap-2 text-[11px] text-slate-300">
            <input v-model="forceEnqueue" data-testid="prelabel-force-enqueue" type="checkbox">
            强制覆盖已有排队任务
          </label>
          <label class="flex items-center gap-2 text-[11px] text-slate-300">
            <input v-model="cancelClaimedToo" data-testid="prelabel-cancel-claimed" type="checkbox">
            取消处理中任务
          </label>
        </div>

        <div class="flex justify-end gap-2">
          <button
            type="button"
            data-testid="prelabel-bulk-cancel"
            class="rounded border border-rose-700 px-3 py-1 text-[11px] text-rose-200 disabled:opacity-50"
            :disabled="submitting || selectedTaskIds.length === 0"
            @click="cancelSelectedTasks"
          >
            取消所选任务
          </button>
          <button
            type="button"
            data-testid="prelabel-bulk-enqueue"
            class="rounded border border-emerald-700 px-3 py-1 text-[11px] text-emerald-200 disabled:opacity-50"
            :disabled="submitting || selectedTaskIds.length === 0"
            @click="enqueueSelectedTasks"
          >
            为所选任务生成预标注
          </button>
        </div>
      </div>

      <div class="grid gap-3 lg:grid-cols-2">
        <div class="rounded border border-slate-800 bg-slate-950/60 p-2">
          <div class="flex items-center justify-between gap-2">
            <p class="text-[11px] font-medium text-slate-200">最近预标注任务</p>
            <span class="text-[10px] text-slate-500">{{ jobs.length }} 条</span>
          </div>
          <ul v-if="jobs.length > 0" class="mt-2 space-y-2">
            <li
              v-for="job in jobs"
              :key="job.job_id"
              class="rounded border border-slate-800 bg-slate-950/40 p-2 text-[10px]"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="min-w-0">
                  <p class="truncate text-slate-200">{{ job.task_id }}</p>
                  <p class="truncate text-slate-500">{{ job.model_version }} / {{ job.model_backbone || '-' }}</p>
                </div>
                <span :class="statusClass(job.status)">{{ job.status }}</span>
              </div>
              <p class="mt-1 break-all text-slate-500">{{ job.model_id || '-' }}</p>
              <p class="mt-1 text-slate-500">worker={{ job.claim_worker_id || '-' }} · attempts={{ job.attempt_count }}</p>
              <p v-if="job.error_message" class="mt-1 text-rose-300">{{ job.error_message }}</p>
              <div class="mt-2 flex justify-end">
                <button
                  v-if="job.status === 'queued' || job.status === 'claimed'"
                  type="button"
                  class="rounded border border-rose-700 px-2 py-0.5 text-[10px] text-rose-200 disabled:opacity-50"
                  :disabled="submitting"
                  @click="cancelJob(job.job_id)"
                >
                  取消
                </button>
              </div>
            </li>
          </ul>
          <p v-else class="mt-2 text-[10px] text-slate-500">暂无预标注任务。</p>
        </div>

        <div class="rounded border border-slate-800 bg-slate-950/60 p-2">
          <div class="flex items-center justify-between gap-2">
            <p class="text-[11px] font-medium text-slate-200">本地 Worker</p>
            <span class="text-[10px] text-slate-500">{{ workers.length }} 个</span>
          </div>
          <ul v-if="workers.length > 0" class="mt-2 space-y-2">
            <li
              v-for="worker in workers"
              :key="worker.worker_id"
              class="rounded border border-slate-800 bg-slate-950/40 p-2 text-[10px]"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="min-w-0">
                  <p class="truncate text-slate-200">{{ worker.display_name || worker.worker_id }}</p>
                  <p class="truncate text-slate-500">{{ worker.host_name || '-' }} · {{ worker.device_label || '-' }}</p>
                </div>
                <span :class="statusClass(worker.status)">{{ worker.status }}</span>
              </div>
              <p class="mt-1 break-all text-slate-500">
                {{ formatWorkerCapabilities(worker.capabilities) }}
              </p>
              <p class="mt-1 text-slate-500">last_seen={{ formatTime(worker.last_seen_at) }}</p>
            </li>
          </ul>
          <p v-else class="mt-2 text-[10px] text-slate-500">暂无 worker 心跳。</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

import { cancelPrelabelJobs, enqueuePrelabels, fetchPrelabelJobs, fetchPrelabelWorkers } from '../services/prelabelApi'
import { fetchTasks } from '../services/taskApi'

const props = defineProps({
  activeTaskId: {
    type: String,
    default: '',
  },
  promotedModel: {
    type: Object,
    default: null,
  },
  open: {
    type: Boolean,
    default: false,
  },
})

const loading = ref(false)
const submitting = ref(false)
const message = ref('')
const error = ref('')
const taskQuery = ref('')
const taskList = ref([])
const jobs = ref([])
const workers = ref([])
const selectedTaskIds = ref([])
const forceEnqueue = ref(true)
const cancelClaimedToo = ref(true)
const modelForm = ref({
  modelVersion: '',
  modelId: '',
  modelBackbone: '',
})

const activeTask = computed(() => taskList.value.find((task) => task.task_id === props.activeTaskId) || null)
const activeTaskFieldKey = computed(() => String(activeTask.value?.field_key || activeTask.value?.field_name || '').trim())
const activeTaskCaptureKey = computed(() => String(activeTask.value?.capture_key || '').trim())
const queuedTaskCount = computed(() => taskList.value.filter((task) => ['queued', 'processing'].includes(String(task.prelabel_status || ''))).length)
const failedTaskCount = computed(() => taskList.value.filter((task) => ['failed', 'cancelled'].includes(String(task.prelabel_status || ''))).length)

const filteredTasks = computed(() => {
  const query = String(taskQuery.value || '').trim().toLowerCase()
  const tasks = Array.isArray(taskList.value) ? taskList.value : []
  const filtered = query
    ? tasks.filter((task) => {
      const haystack = [
        task.task_id,
        task.field_name,
        task.field_key,
        task.capture_key,
        task.prelabel_status,
      ]
        .map((item) => String(item || '').toLowerCase())
        .join(' ')
      return haystack.includes(query)
    })
    : tasks
  return filtered.slice(0, 80)
})

function syncModelFormFromPromoted(force = false) {
  const promoted = props.promotedModel || {}
  if (!force && modelForm.value.modelVersion) {
    return
  }
  modelForm.value = {
    modelVersion: String(promoted.model_version || '').trim(),
    modelId: String(promoted.model_id || '').trim(),
    modelBackbone: String(promoted.model_backbone || '').trim(),
  }
}

function normalizeTaskIdList(taskIds) {
  return Array.from(new Set((Array.isArray(taskIds) ? taskIds : []).map((item) => String(item || '').trim()).filter(Boolean)))
}

function isTaskSelected(taskId) {
  return selectedTaskIds.value.includes(String(taskId || '').trim())
}

function toggleTaskSelection(taskId) {
  const normalized = String(taskId || '').trim()
  if (!normalized) {
    return
  }
  if (isTaskSelected(normalized)) {
    selectedTaskIds.value = selectedTaskIds.value.filter((item) => item !== normalized)
    return
  }
  selectedTaskIds.value = [...selectedTaskIds.value, normalized]
}

function setSelectedTasks(taskIds) {
  selectedTaskIds.value = normalizeTaskIdList(taskIds)
}

function applyTaskPreset(preset) {
  const current = activeTask.value
  switch (preset) {
    case 'current':
      setSelectedTasks(props.activeTaskId ? [props.activeTaskId] : [])
      break
    case 'same-field':
      if (!current) {
        return
      }
      setSelectedTasks(
        taskList.value
          .filter((task) => String(task.field_key || task.field_name || '') === String(current.field_key || current.field_name || ''))
          .map((task) => task.task_id),
      )
      break
    case 'same-capture':
      if (!current) {
        return
      }
      setSelectedTasks(
        taskList.value
          .filter((task) => String(task.capture_key || '') === String(current.capture_key || ''))
          .map((task) => task.task_id),
      )
      break
    case 'queued':
      setSelectedTasks(
        taskList.value
          .filter((task) => ['queued', 'processing'].includes(String(task.prelabel_status || '')))
          .map((task) => task.task_id),
      )
      break
    case 'failed':
      setSelectedTasks(
        taskList.value
          .filter((task) => ['failed', 'cancelled'].includes(String(task.prelabel_status || '')))
          .map((task) => task.task_id),
      )
      break
    case 'missing':
      setSelectedTasks(
        taskList.value
          .filter((task) => !String(task.prelabel_status || '').trim())
          .map((task) => task.task_id),
      )
      break
    default:
      break
  }
}

function selectVisibleTasks() {
  setSelectedTasks(filteredTasks.value.map((task) => task.task_id))
}

function formatTime(value) {
  const text = String(value || '').trim()
  if (!text) {
    return '--'
  }
  return text.replace('T', ' ').slice(0, 19)
}

function formatTaskPrelabelStatus(status) {
  const normalized = String(status || '').trim().toLowerCase()
  if (!normalized) {
    return '未生成'
  }
  const labels = {
    available: '可用',
    accepted: '已接受',
    queued: '排队中',
    processing: '处理中',
    failed: '失败',
    cancelled: '已取消',
    superseded: '已过期',
    completed: '已完成',
  }
  return labels[normalized] || normalized
}

function formatWorkerCapabilities(capabilities) {
  if (!capabilities || typeof capabilities !== 'object') {
    return '-'
  }
  const versions = Array.isArray(capabilities.model_versions) ? capabilities.model_versions.join(', ') : ''
  const ids = Array.isArray(capabilities.model_ids) ? capabilities.model_ids.join(', ') : ''
  const backbones = Array.isArray(capabilities.model_backbones) ? capabilities.model_backbones.join(', ') : ''
  return `versions=${versions || '-'} | ids=${ids || '-'} | backbones=${backbones || '-'}`
}

function statusClass(status) {
  const normalized = String(status || '').toLowerCase()
  if (normalized === 'completed' || normalized === 'available' || normalized === 'accepted' || normalized === 'online') {
    return 'text-emerald-300'
  }
  if (normalized === 'queued' || normalized === 'claimed' || normalized === 'processing') {
    return 'text-amber-300'
  }
  if (normalized === 'failed' || normalized === 'cancelled' || normalized === 'offline') {
    return 'text-rose-300'
  }
  return 'text-slate-400'
}

async function refreshAll() {
  loading.value = true
  error.value = ''
  try {
    const [nextTasks, nextJobs, nextWorkers] = await Promise.all([
      fetchTasks(''),
      fetchPrelabelJobs({ limit: 12 }),
      fetchPrelabelWorkers({ limit: 12 }),
    ])
    taskList.value = Array.isArray(nextTasks) ? nextTasks : []
    jobs.value = Array.isArray(nextJobs) ? nextJobs : []
    workers.value = Array.isArray(nextWorkers) ? nextWorkers : []
  } catch (err) {
    error.value = err instanceof Error ? err.message : '加载预标注管理状态失败'
  } finally {
    loading.value = false
  }
}

async function enqueueSelectedTasks() {
  submitting.value = true
  message.value = ''
  error.value = ''
  try {
    const response = await enqueuePrelabels({
      taskIds: selectedTaskIds.value,
      modelVersion: modelForm.value.modelVersion,
      modelId: modelForm.value.modelId,
      modelBackbone: modelForm.value.modelBackbone,
      force: forceEnqueue.value,
    })
    message.value = `已请求 ${response.enqueued_count} 个预标注任务，跳过 ${response.skipped_count} 个`
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : '创建预标注任务失败'
  } finally {
    submitting.value = false
  }
}

async function cancelSelectedTasks() {
  submitting.value = true
  message.value = ''
  error.value = ''
  try {
    const response = await cancelPrelabelJobs({
      taskIds: selectedTaskIds.value,
      statuses: cancelClaimedToo.value ? ['queued', 'claimed'] : ['queued'],
      reason: 'cancelled from prelabel control panel',
    })
    message.value = `已取消 ${response.cancelled_count} 个预标注任务`
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : '取消预标注任务失败'
  } finally {
    submitting.value = false
  }
}

async function cancelJob(jobId) {
  submitting.value = true
  message.value = ''
  error.value = ''
  try {
    const response = await cancelPrelabelJobs({
      jobIds: [jobId],
      statuses: ['queued', 'claimed'],
      reason: 'cancelled from prelabel control panel',
    })
    message.value = `已取消 ${response.cancelled_count} 个预标注任务`
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : '取消预标注任务失败'
  } finally {
    submitting.value = false
  }
}

watch(
  () => props.promotedModel,
  () => {
    syncModelFormFromPromoted(false)
  },
  { immediate: true, deep: true },
)

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      refreshAll()
    }
  },
  { immediate: true },
)
</script>
