<template>
  <header class="border-b border-slate-800 bg-slate-900 px-4 flex items-center justify-between">
    <h1 class="text-lg font-semibold">SCANN Native Annotation</h1>
    <div class="flex items-center gap-3">
      <div v-if="isAdmin" class="relative">
        <button
          type="button"
          data-testid="header-sync-menu-toggle"
          class="text-xs rounded border border-sky-700 px-2 py-1 text-sky-200"
          @click="toggleSyncMenu"
        >
          标注同步
        </button>
        <div
          v-if="syncMenuOpen"
          data-testid="header-sync-menu"
          class="absolute right-0 top-full z-40 mt-2 w-72 rounded border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300 shadow-xl"
        >
          <div class="flex items-center justify-between gap-2">
            <p class="font-semibold text-slate-100">手动同步</p>
            <button
              type="button"
              class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-400"
              @click="syncMenuOpen = false"
            >
              关闭
            </button>
          </div>
          <dl class="mt-2 space-y-1 text-[11px] text-slate-400">
            <div class="flex justify-between gap-2">
              <dt>配置</dt>
              <dd :class="syncStatus?.configured ? 'text-emerald-300' : 'text-amber-300'">
                {{ syncStatus?.configured ? '已配置' : '未配置' }}
              </dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt>数据集</dt>
              <dd class="truncate text-slate-200">{{ syncStatus?.dataset_id || '--' }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt>Schema</dt>
              <dd class="truncate text-slate-200">{{ syncStatus?.schema_name || '--' }}</dd>
            </div>
            <div class="flex justify-between gap-2">
              <dt>最近结果</dt>
              <dd :class="lastSyncSucceeded ? 'text-emerald-300' : 'text-slate-400'">
                {{ lastSyncText }}
              </dd>
            </div>
          </dl>
          <p v-if="syncMessage" data-testid="header-sync-message" class="mt-2 text-[11px] text-emerald-300">
            {{ syncMessage }}
          </p>
          <p v-if="syncError" data-testid="header-sync-error" class="mt-2 text-[11px] text-rose-300">
            {{ syncError }}
          </p>
          <div class="mt-3 grid grid-cols-3 gap-2">
            <button
              type="button"
              data-testid="header-sync-refresh"
              class="rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-300 disabled:opacity-50"
              :disabled="syncLoading"
              @click="loadSyncStatus"
            >
              刷新
            </button>
            <button
              type="button"
              data-testid="header-sync-run"
              class="rounded border border-emerald-700 px-2 py-1 text-[11px] text-emerald-200 disabled:opacity-50"
              :disabled="syncLoading"
              @click="runManualSync(false)"
            >
              增量
            </button>
            <button
              type="button"
              data-testid="header-sync-run-full"
              class="rounded border border-amber-700 px-2 py-1 text-[11px] text-amber-200 disabled:opacity-50"
              :disabled="syncLoading"
              @click="runManualSync(true)"
            >
              全量
            </button>
          </div>
        </div>
      </div>

      <div class="flex flex-col items-end">
        <span class="text-xs text-slate-400" data-testid="header-username">{{ usernameText }}</span>
        <span
          v-if="sessionRemainingText"
          class="text-[10px] text-slate-500"
          data-testid="header-session-remaining"
        >
          {{ sessionRemainingText }}
        </span>
      </div>
      <button
        type="button"
        data-testid="header-logout"
        class="text-xs rounded border border-slate-700 px-2 py-1 text-slate-300"
        @click="$emit('logout')"
      >
        退出登录
      </button>
    </div>
  </header>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'

import { fetchAnnotationSyncStatus, runAnnotationSync } from '../services/annotationSyncApi'

const props = defineProps({
  username: {
    type: String,
    default: '',
  },
  token: {
    type: String,
    default: '',
  },
  role: {
    type: String,
    default: '',
  },
})

defineEmits(['logout'])

const usernameText = computed(() => (props.username ? `用户: ${props.username}` : '用户: 访客'))
const nowMs = ref(Date.now())
const syncMenuOpen = ref(false)
const syncStatus = ref(null)
const syncLoading = ref(false)
const syncMessage = ref('')
const syncError = ref('')
let sessionTimerId = null

const isAdmin = computed(() => props.role === 'admin')
const lastSyncSucceeded = computed(() => syncStatus.value?.last_result?.success === true)
const lastSyncText = computed(() => {
  const result = syncStatus.value?.last_result
  if (!result) {
    return '--'
  }
  if (!result.success) {
    return '失败'
  }
  return `${result.sync_mode || 'sync'} r${result.last_revision_rowid || 0}`
})

function decodeTokenExpMs(token) {
  try {
    const payloadPart = token.split('.')[1]
    if (!payloadPart) {
      return 0
    }
    const normalized = payloadPart.replace(/-/g, '+').replace(/_/g, '/')
    const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4)
    const payload = JSON.parse(atob(padded))
    const exp = Number(payload?.exp)
    return Number.isFinite(exp) ? exp * 1000 : 0
  } catch {
    return 0
  }
}

function stopSessionTimer() {
  if (sessionTimerId) {
    window.clearInterval(sessionTimerId)
    sessionTimerId = null
  }
}

function startSessionTimer() {
  stopSessionTimer()
  if (!props.token) {
    return
  }
  nowMs.value = Date.now()
  sessionTimerId = window.setInterval(() => {
    nowMs.value = Date.now()
  }, 1000)
}

async function loadSyncStatus() {
  syncLoading.value = true
  syncError.value = ''
  try {
    syncStatus.value = await fetchAnnotationSyncStatus()
  } catch (err) {
    syncError.value = err instanceof Error ? err.message : '加载同步状态失败'
  } finally {
    syncLoading.value = false
  }
}

async function toggleSyncMenu() {
  syncMenuOpen.value = !syncMenuOpen.value
  if (syncMenuOpen.value && !syncStatus.value) {
    await loadSyncStatus()
  }
}

async function runManualSync(full) {
  syncLoading.value = true
  syncMessage.value = ''
  syncError.value = ''
  try {
    const result = await runAnnotationSync({ full })
    syncStatus.value = {
      ...(syncStatus.value || {}),
      last_result: result,
    }
    if (!result.success) {
      syncError.value = result.error_message || '同步失败'
      return
    }
    syncMessage.value = `${full ? '全量' : '增量'}同步完成: ${result.revisions_synced || 0} 个版本，${result.current_boxes_synced || 0} 个当前框`
  } catch (err) {
    syncError.value = err instanceof Error ? err.message : '同步失败'
  } finally {
    syncLoading.value = false
  }
}

function formatDuration(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  return [hours, minutes, seconds]
    .map((value) => String(value).padStart(2, '0'))
    .join(':')
}

const tokenExpiresAtMs = computed(() => decodeTokenExpMs(props.token))
const sessionRemainingMs = computed(() => Math.max(0, tokenExpiresAtMs.value - nowMs.value))
const sessionRemainingText = computed(() => {
  if (!props.token || !tokenExpiresAtMs.value) {
    return ''
  }
  if (sessionRemainingMs.value <= 0) {
    return '会话已过期'
  }
  return `会话剩余 ${formatDuration(sessionRemainingMs.value)}`
})

watch(
  () => props.token,
  () => {
    startSessionTimer()
  },
  { immediate: true },
)

watch(isAdmin, (value) => {
  if (!value) {
    syncMenuOpen.value = false
  }
})

onBeforeUnmount(() => {
  stopSessionTimer()
})
</script>
