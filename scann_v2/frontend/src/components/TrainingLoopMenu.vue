<template>
  <div class="relative">
    <button
      type="button"
      data-testid="header-training-menu-toggle"
      class="text-xs rounded border border-emerald-700 px-2 py-1 text-emerald-200"
      @click="toggleMenu"
    >
      训练闭环
    </button>
    <div
      v-if="menuOpen"
      data-testid="header-training-menu"
      class="absolute right-0 top-full z-40 mt-2 w-[36rem] max-w-[92vw] rounded border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300 shadow-xl"
    >
      <div class="flex items-center justify-between gap-2">
        <div>
          <p class="font-semibold text-slate-100">训练与模型</p>
          <p class="mt-0.5 text-[11px] text-slate-500">管理快照、训练作业、模型推广和预标注回流。</p>
        </div>
        <div class="flex items-center gap-2">
          <button
            type="button"
            data-testid="training-menu-refresh"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400 disabled:opacity-50"
            :disabled="loading"
            @click="refreshAll"
          >
            刷新
          </button>
          <button
            type="button"
            class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400"
            @click="menuOpen = false"
          >
            关闭
          </button>
        </div>
      </div>

      <p v-if="message" data-testid="training-menu-message" class="mt-2 text-[11px] text-emerald-300">
        {{ message }}
      </p>
      <p v-if="error" data-testid="training-menu-error" class="mt-2 text-[11px] text-rose-300">
        {{ error }}
      </p>

      <div class="mt-3 max-h-[70vh] space-y-3 overflow-y-auto pr-1">
        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-slate-100">当前生产模型</p>
            <span class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400">
              {{ promotedModel ? (promotedModel.task_type || 'classification') : '未推广' }}
            </span>
          </div>
          <div v-if="promotedModel" data-testid="training-promoted-model" class="mt-2 space-y-1 text-[11px]">
            <p class="text-slate-200">{{ promotedModel.model_version }} / {{ promotedModel.model_backbone }}</p>
            <p class="break-all text-slate-500">{{ promotedModel.model_id }}</p>
            <p class="text-slate-500">推广时间：{{ formatTime(promotedModel.promoted_at) }}</p>
          </div>
          <p v-else class="mt-2 text-[11px] text-slate-500">当前还没有 promoted model。</p>
        </section>

        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-slate-100">创建训练快照</p>
            <button
              v-if="activeTaskId"
              type="button"
              data-testid="training-use-current-task"
              class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400"
              @click="useCurrentTaskForSnapshot"
            >
              使用当前任务
            </button>
          </div>
          <div class="mt-2 grid gap-2">
            <label class="grid gap-1">
              <span class="text-[11px] text-slate-500">快照名称</span>
              <input
                v-model="snapshotForm.snapshotName"
                data-testid="training-snapshot-name"
                type="text"
                class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                placeholder="round-1"
              >
            </label>
            <label class="grid gap-1">
              <span class="text-[11px] text-slate-500">任务 ID 列表</span>
              <input
                v-model="snapshotForm.taskIdsText"
                data-testid="training-snapshot-task-ids"
                type="text"
                class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                placeholder="留空表示全部已标注任务；多个任务用逗号分隔"
              >
            </label>
            <div class="flex justify-end">
              <button
                type="button"
                data-testid="training-create-snapshot"
                class="rounded border border-emerald-700 px-3 py-1 text-[11px] text-emerald-200 disabled:opacity-50"
                :disabled="submitting"
                @click="submitSnapshot"
              >
                创建快照
              </button>
            </div>
          </div>
        </section>

        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-slate-100">创建训练作业</p>
            <button
              v-if="activeTaskId"
              type="button"
              data-testid="training-use-current-task-for-job"
              class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400"
              @click="useCurrentTaskForJob"
            >
              当前任务回流
            </button>
          </div>
          <div class="mt-2 grid gap-2">
            <label class="grid gap-1">
              <span class="text-[11px] text-slate-500">使用已有快照</span>
              <select
                v-model="jobForm.snapshotId"
                data-testid="training-job-snapshot"
                class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
              >
                <option value="">自动新建快照</option>
                <option v-for="snapshot in snapshots" :key="snapshot.snapshot_id" :value="snapshot.snapshot_id">
                  {{ snapshot.snapshot_name }} · {{ snapshot.annotation_count }} 标注
                </option>
              </select>
            </label>
            <div
              v-if="selectedSnapshotAudit"
              data-testid="training-selected-snapshot-audit"
              class="rounded border border-amber-800 bg-amber-950/20 px-2 py-1 text-[11px] text-amber-200"
            >
              {{ auditSummary(selectedSnapshotAudit) }}
            </div>
            <div v-if="!jobForm.snapshotId" class="grid gap-2 rounded border border-slate-800 bg-slate-950/60 p-2">
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">新快照名称</span>
                <input
                  v-model="jobForm.snapshotName"
                  data-testid="training-job-snapshot-name"
                  type="text"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                  placeholder="train-round-2"
                >
              </label>
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">新快照任务 ID</span>
                <input
                  v-model="jobForm.snapshotTaskIdsText"
                  data-testid="training-job-snapshot-task-ids"
                  type="text"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                  placeholder="留空表示全部已标注任务"
                >
              </label>
            </div>
            <div class="grid grid-cols-2 gap-2">
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">任务类型</span>
                <select
                  v-model="jobForm.taskType"
                  data-testid="training-job-task-type"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                >
                  <option value="classification">classification</option>
                  <option value="detection">detection</option>
                </select>
              </label>
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">Backbone</span>
                <input
                  v-model="jobForm.modelBackbone"
                  data-testid="training-job-model-backbone"
                  type="text"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                  placeholder="ViT_B_16"
                >
              </label>
            </div>
            <div class="grid grid-cols-2 gap-2">
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">模型版本</span>
                <input
                  v-model="jobForm.modelVersion"
                  data-testid="training-job-model-version"
                  type="text"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                  placeholder="cls-v3"
                >
              </label>
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">模型 ID</span>
                <input
                  v-model="jobForm.modelId"
                  data-testid="training-job-model-id"
                  type="text"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                  placeholder="可选，不填自动生成"
                >
              </label>
            </div>
            <div class="grid grid-cols-3 gap-2">
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">Epochs</span>
                <input
                  v-model.number="jobForm.epochs"
                  data-testid="training-job-epochs"
                  type="number"
                  min="1"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                >
              </label>
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">Batch</span>
                <input
                  v-model.number="jobForm.batchSize"
                  data-testid="training-job-batch-size"
                  type="number"
                  min="1"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                >
              </label>
              <label class="grid gap-1">
                <span class="text-[11px] text-slate-500">LR</span>
                <input
                  v-model.number="jobForm.learningRate"
                  data-testid="training-job-learning-rate"
                  type="number"
                  min="0"
                  step="0.0001"
                  class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                >
              </label>
            </div>
            <label class="grid gap-1">
              <span class="text-[11px] text-slate-500">Advanced JSON</span>
              <textarea
                v-model="jobForm.advancedConfigText"
                data-testid="training-job-advanced-config"
                rows="6"
                class="rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-[11px] text-slate-100"
              />
            </label>
            <label class="grid gap-1">
              <span class="text-[11px] text-slate-500">训练成功后重排预标注任务</span>
              <input
                v-model="jobForm.prelabelTaskIdsText"
                data-testid="training-job-prelabel-task-ids"
                type="text"
                class="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                :placeholder="activeTaskId ? `可填 ${activeTaskId}，留空表示按服务端默认范围` : '多个任务用逗号分隔'"
              >
            </label>
            <div class="grid grid-cols-3 gap-2 text-[11px] text-slate-300">
              <label class="flex items-center gap-2">
                <input v-model="jobForm.promoteOnSuccess" type="checkbox">
                自动推广
              </label>
              <label class="flex items-center gap-2">
                <input v-model="jobForm.enqueuePrelabelsOnSuccess" type="checkbox">
                自动预标注
              </label>
              <label class="flex items-center gap-2">
                <input v-model="jobForm.forcePrelabel" type="checkbox">
                强制重跑
              </label>
            </div>
            <div class="flex justify-end">
              <button
                type="button"
                data-testid="training-create-job"
                class="rounded border border-emerald-700 px-3 py-1 text-[11px] text-emerald-200 disabled:opacity-50"
                :disabled="submitting"
                @click="submitTrainingJob"
              >
                创建训练作业
              </button>
            </div>
          </div>
        </section>

        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-slate-100">最近训练作业</p>
            <span class="text-[10px] text-slate-500">{{ jobs.length }} 条</span>
          </div>
          <div v-if="jobs.length === 0" class="mt-2 text-[11px] text-slate-500">还没有训练作业。</div>
          <ul v-else class="mt-2 space-y-2">
            <li
              v-for="job in jobs"
              :key="job.job_id"
              class="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px]"
            >
              <div class="flex items-center justify-between gap-2">
                <span class="font-medium text-slate-200">{{ job.model_version }} / {{ job.model_backbone }}</span>
                <span :class="statusClass(job.status)">{{ job.status }}</span>
              </div>
              <p class="mt-1 break-all text-slate-500">{{ job.model_id }}</p>
              <p class="mt-1 text-slate-500">快照：{{ job.snapshot_id }} · 尝试 {{ job.attempt_count }}</p>
            </li>
          </ul>
        </section>

        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-slate-100">最近训练运行</p>
            <span class="text-[10px] text-slate-500">{{ runs.length }} 条</span>
          </div>
          <div v-if="runs.length === 0" class="mt-2 text-[11px] text-slate-500">还没有训练运行。</div>
          <ul v-else class="mt-2 space-y-2">
            <li
              v-for="run in runs"
              :key="run.run_id"
              class="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px]"
            >
              <div class="flex items-center justify-between gap-2">
                <span class="font-medium text-slate-200">{{ run.model_version || run.model_id || run.run_id }}</span>
                <span :class="statusClass(run.status)">{{ run.status }}</span>
              </div>
              <p class="mt-1 text-slate-500">{{ formatMetrics(run.metrics) }}</p>
            </li>
          </ul>
        </section>

        <section class="rounded border border-slate-800 bg-slate-900/70 p-3">
          <div class="flex items-center justify-between gap-2">
            <p class="font-medium text-slate-100">模型注册表</p>
            <span class="text-[10px] text-slate-500">{{ models.length }} 条</span>
          </div>
          <div v-if="models.length === 0" class="mt-2 text-[11px] text-slate-500">还没有模型。</div>
          <ul v-else class="mt-2 space-y-2">
            <li
              v-for="model in models"
              :key="model.model_id"
              class="rounded border border-slate-800 bg-slate-950/60 p-2 text-[11px]"
            >
              <div class="flex items-center justify-between gap-2">
                <div class="min-w-0">
                  <p class="truncate font-medium text-slate-200">
                    {{ model.model_version }} / {{ model.model_backbone }}
                  </p>
                  <p class="break-all text-slate-500">{{ model.model_id }}</p>
                </div>
                <span :class="model.is_promoted ? 'text-emerald-300' : 'text-slate-500'">
                  {{ model.is_promoted ? '当前' : '候选' }}
                </span>
              </div>
              <p class="mt-1 text-slate-500">{{ formatMetrics(model.metrics) }}</p>
              <p
                v-if="promotionWarningsForModel(model).length"
                data-testid="training-model-promotion-warnings"
                class="mt-1 text-[10px] text-amber-300"
              >
                {{ warningSummary(promotionWarningsForModel(model)) }}
              </p>
              <div class="mt-2 flex flex-wrap justify-end gap-2">
                <button
                  type="button"
                  class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300 disabled:opacity-50"
                  :disabled="submitting || model.is_promoted"
                  @click="promoteModel(model, false)"
                >
                  设为当前
                </button>
                <button
                  type="button"
                  data-testid="training-promote-and-enqueue"
                  class="rounded border border-emerald-700 px-2 py-0.5 text-[10px] text-emerald-200 disabled:opacity-50"
                  :disabled="submitting"
                  @click="promoteModel(model, true)"
                >
                  推广并重排预标注
                </button>
              </div>
            </li>
          </ul>
        </section>

        <PrelabelControlPanel
          :active-task-id="activeTaskId"
          :promoted-model="promotedModel"
          :open="menuOpen"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

import PrelabelControlPanel from './PrelabelControlPanel.vue'
import {
  createTrainingJob,
  createTrainingSnapshot,
  fetchPromotedTrainingModel,
  fetchTrainingJobs,
  fetchTrainingModels,
  fetchTrainingRuns,
  fetchTrainingSnapshots,
  promoteTrainingModel,
} from '../services/trainingApi'

const props = defineProps({
  activeTaskId: {
    type: String,
    default: '',
  },
})

const menuOpen = ref(false)
const loading = ref(false)
const submitting = ref(false)
const message = ref('')
const error = ref('')

const snapshots = ref([])
const jobs = ref([])
const runs = ref([])
const models = ref([])
const promotedModel = ref(null)

const selectedSnapshot = computed(() => {
  const snapshotId = String(jobForm.value.snapshotId || '')
  return snapshots.value.find((snapshot) => String(snapshot.snapshot_id || '') === snapshotId) || null
})

const selectedSnapshotAudit = computed(() => selectedSnapshot.value?.metadata?.class_audit || null)

const snapshotForm = ref({
  snapshotName: '',
  taskIdsText: '',
})

const jobForm = ref({
  snapshotId: '',
  snapshotName: '',
  snapshotTaskIdsText: '',
  taskType: 'classification',
  modelVersion: '',
  modelId: '',
  modelBackbone: 'ViT_B_16',
  epochs: 20,
  batchSize: 32,
  learningRate: 0.001,
  advancedConfigText: JSON.stringify({
    training_mode: 'frozen_feature_classifier',
    feature_encoder: 'dinov2_vitb14_reg',
    feature_cache_enabled: true,
    prior_logit_correction: {
      enabled: true,
      tau: 1.0,
    },
    variance_transfer: {
      enabled: true,
      synthetic_per_tail: 500,
      tail_max_support: 20,
      donor_min_support: 100,
      shrinkage: 0.2,
    },
    dbl: {
      enabled: true,
    },
    expert_distillation: {
      enabled: false,
    },
  }, null, 2),
  promoteOnSuccess: true,
  enqueuePrelabelsOnSuccess: true,
  prelabelTaskIdsText: '',
  forcePrelabel: false,
})

function parseTaskIds(raw) {
  return String(raw || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function parseAdvancedConfig(raw) {
  const text = String(raw || '').trim()
  if (!text) {
    return {}
  }
  const parsed = JSON.parse(text)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Advanced JSON must be an object')
  }
  return parsed
}

function formatTime(value) {
  const text = String(value || '').trim()
  if (!text) {
    return '--'
  }
  return text.replace('T', ' ').slice(0, 19)
}

function formatMetrics(metrics) {
  if (!metrics || typeof metrics !== 'object') {
    return '暂无指标'
  }
  const keys = ['macro_f1_supported', 'best_macro_f1', 'f1', 'best_f2', 'val_loss', 'precision', 'recall']
  const parts = keys
    .filter((key) => Object.prototype.hasOwnProperty.call(metrics, key))
    .map((key) => {
      const value = Number(metrics[key])
      return Number.isFinite(value) ? `${key}=${value.toFixed(3)}` : `${key}=${String(metrics[key])}`
    })
  return parts.length > 0 ? parts.join(' · ') : '暂无指标'
}

function auditSummary(audit) {
  if (!audit || typeof audit !== 'object') {
    return ''
  }
  const buckets = audit.bucket_counts || {}
  const missing = Array.isArray(audit.missing_classes) ? audit.missing_classes.length : 0
  const low = Array.isArray(audit.low_sample_classes) ? audit.low_sample_classes.length : 0
  return `samples=${Number(audit.total_samples || 0)} real=${Number(buckets.real || 0)} bogus=${Number(buckets.bogus || 0)} missing=${missing} low=${low}`
}

function promotionWarningsForModel(model) {
  const direct = model?.metadata?.promotion_warnings || model?.metrics?.promotion_warnings
  if (Array.isArray(direct) && direct.length > 0) {
    return direct
  }
  const nested = model?.metrics?.class_support?.promotion_warnings
  return Array.isArray(nested) ? nested : []
}

function warningSummary(warnings) {
  return Array.isArray(warnings) ? warnings.slice(0, 3).join(' | ') : ''
}

function statusClass(status) {
  const normalized = String(status || '').toLowerCase()
  if (normalized === 'completed' || normalized === 'available' || normalized === 'accepted') {
    return 'text-emerald-300'
  }
  if (normalized === 'queued' || normalized === 'claimed' || normalized === 'processing') {
    return 'text-amber-300'
  }
  if (normalized === 'failed' || normalized === 'cancelled') {
    return 'text-rose-300'
  }
  return 'text-slate-400'
}

async function refreshAll() {
  loading.value = true
  error.value = ''
  try {
    const [
      nextSnapshots,
      nextJobs,
      nextRuns,
      nextModels,
      nextPromoted,
    ] = await Promise.all([
      fetchTrainingSnapshots({ limit: 6 }),
      fetchTrainingJobs({ limit: 6 }),
      fetchTrainingRuns({ limit: 6 }),
      fetchTrainingModels({ taskType: 'classification', limit: 6 }),
      fetchPromotedTrainingModel({ taskType: 'classification' }),
    ])
    snapshots.value = Array.isArray(nextSnapshots) ? nextSnapshots : []
    jobs.value = Array.isArray(nextJobs) ? nextJobs : []
    runs.value = Array.isArray(nextRuns) ? nextRuns : []
    models.value = Array.isArray(nextModels) ? nextModels : []
    promotedModel.value = nextPromoted
    if (!jobForm.value.snapshotId && snapshots.value.length > 0) {
      jobForm.value.snapshotId = String(snapshots.value[0].snapshot_id || '')
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : '加载训练闭环状态失败'
  } finally {
    loading.value = false
  }
}

async function toggleMenu() {
  menuOpen.value = !menuOpen.value
  if (menuOpen.value) {
    await refreshAll()
  }
}

function useCurrentTaskForSnapshot() {
  if (!props.activeTaskId) {
    return
  }
  snapshotForm.value.taskIdsText = props.activeTaskId
}

function useCurrentTaskForJob() {
  if (!props.activeTaskId) {
    return
  }
  if (!jobForm.value.snapshotId && !jobForm.value.snapshotTaskIdsText) {
    jobForm.value.snapshotTaskIdsText = props.activeTaskId
  }
  jobForm.value.prelabelTaskIdsText = props.activeTaskId
}

async function submitSnapshot() {
  submitting.value = true
  message.value = ''
  error.value = ''
  try {
    const snapshot = await createTrainingSnapshot({
      snapshotName: snapshotForm.value.snapshotName,
      taskIds: parseTaskIds(snapshotForm.value.taskIdsText),
    })
    snapshots.value = [snapshot, ...snapshots.value.filter((item) => item.snapshot_id !== snapshot.snapshot_id)].slice(0, 6)
    jobForm.value.snapshotId = String(snapshot.snapshot_id || '')
    message.value = `已创建训练快照 ${snapshot.snapshot_name}`
  } catch (err) {
    error.value = err instanceof Error ? err.message : '创建训练快照失败'
  } finally {
    submitting.value = false
  }
}

async function submitTrainingJob() {
  submitting.value = true
  message.value = ''
  error.value = ''
  try {
    const advancedConfig = parseAdvancedConfig(jobForm.value.advancedConfigText)
    const job = await createTrainingJob({
      snapshotId: jobForm.value.snapshotId,
      snapshotName: jobForm.value.snapshotName,
      snapshotTaskIds: parseTaskIds(jobForm.value.snapshotTaskIdsText),
      taskType: jobForm.value.taskType,
      modelVersion: jobForm.value.modelVersion,
      modelId: jobForm.value.modelId,
      modelBackbone: jobForm.value.modelBackbone,
      trainConfig: {
        training_mode: 'frozen_feature_classifier',
        feature_encoder: 'dinov2_vitb14_reg',
        feature_cache_enabled: true,
        prior_logit_correction: { enabled: true, tau: 1.0 },
        ...advancedConfig,
        epochs: Number(jobForm.value.epochs) || 20,
        batch_size: Number(jobForm.value.batchSize) || 32,
        lr: Number(jobForm.value.learningRate) || 0.001,
      },
      promoteOnSuccess: jobForm.value.promoteOnSuccess,
      enqueuePrelabelsOnSuccess: jobForm.value.enqueuePrelabelsOnSuccess,
      prelabelTaskIds: parseTaskIds(jobForm.value.prelabelTaskIdsText),
      forcePrelabel: jobForm.value.forcePrelabel,
    })
    jobs.value = [job, ...jobs.value.filter((item) => item.job_id !== job.job_id)].slice(0, 6)
    message.value = `已创建训练作业 ${job.model_version}`
    await refreshAll()
  } catch (err) {
    error.value = err instanceof Error ? err.message : '创建训练作业失败'
  } finally {
    submitting.value = false
  }
}

async function promoteModel(model, enqueuePrelabels) {
  const modelId = String(model?.model_id || '').trim()
  if (!modelId) {
    return
  }
  const warnings = promotionWarningsForModel(model)
  if (warnings.length > 0 && typeof window !== 'undefined' && typeof window.confirm === 'function') {
    const accepted = window.confirm(`Model has class-coverage warnings:\n${warningSummary(warnings)}\nPromote anyway?`)
    if (!accepted) {
      return
    }
  }
  submitting.value = true
  message.value = ''
  error.value = ''
  try {
    const response = await promoteTrainingModel(modelId, {
      enqueuePrelabels,
      forcePrelabel: enqueuePrelabels,
      taskIds: enqueuePrelabels ? parseTaskIds(jobForm.value.prelabelTaskIdsText) : [],
    })
    promotedModel.value = response.model
    models.value = models.value.map((item) => ({
      ...item,
      is_promoted: item.model_id === response.model.model_id,
      promoted_at: item.model_id === response.model.model_id ? response.model.promoted_at : item.promoted_at,
    }))
    const responseWarnings = Array.isArray(response.promotion_warnings) ? response.promotion_warnings : []
    message.value = enqueuePrelabels && response.prelabel_enqueue
      ? `已推广模型并排入 ${response.prelabel_enqueue.enqueued_count} 个预标注任务`
      : `已将 ${response.model.model_version} 设为当前模型`
    if (responseWarnings.length > 0 && !(enqueuePrelabels && response.prelabel_enqueue)) {
      message.value = `${message.value} | ${warningSummary(responseWarnings)}`
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : '推广模型失败'
  } finally {
    submitting.value = false
  }
}
</script>
