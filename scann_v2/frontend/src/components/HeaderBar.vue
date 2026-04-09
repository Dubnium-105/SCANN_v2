<template>
  <header class="border-b border-slate-800 bg-slate-900 px-4 flex items-center justify-between">
    <h1 class="text-lg font-semibold tracking-wide">SCANN Native Annotation</h1>
    <div class="flex items-center gap-3">
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

const props = defineProps({
  username: {
    type: String,
    default: '',
  },
  token: {
    type: String,
    default: '',
  },
})

defineEmits(['logout'])

const usernameText = computed(() => (props.username ? `用户: ${props.username}` : '用户: 访客'))
const nowMs = ref(Date.now())
let sessionTimerId = null

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

onBeforeUnmount(() => {
  stopSessionTimer()
})
</script>
