<template>
  <div class="h-full grid grid-rows-[56px_1fr]">
    <HeaderBar
      :username="authState.username"
      :token="authState.token"
      :role="authState.role"
      @logout="onLogout"
    />
    <main
      ref="mainRef"
      class="flex flex-col lg:flex-row gap-3 p-3 min-h-0"
      :class="isResizing ? 'select-none' : ''"
    >
      <aside
        class="hidden lg:flex rounded-lg border border-slate-800 bg-slate-900 p-3 overflow-y-auto flex-col min-h-0"
        :style="leftPaneStyle"
      >
        <div class="flex items-center justify-between mb-2">
          <p v-if="!leftCollapsed" class="text-sm font-semibold text-slate-200">快捷键</p>
          <button
            class="text-xs px-2 py-1 rounded border border-slate-700 text-slate-300"
            :title="leftCollapsed ? '展开左侧面板' : '折叠左侧面板'"
            @click="toggleLeftCollapsed"
          >
            {{ leftCollapsed ? '»' : '«' }}
          </button>
        </div>

        <div v-show="!leftCollapsed" class="min-h-0 flex flex-col">
          <div class="mb-2 flex justify-end">
            <button
              class="text-[11px] px-2 py-1 rounded border border-slate-700 text-slate-300"
              :title="hotkeysCollapsed ? '展开快捷键' : '折叠快捷键'"
              @click="hotkeysCollapsed = !hotkeysCollapsed"
            >
              {{ hotkeysCollapsed ? '展开快捷键' : '折叠快捷键' }}
            </button>
          </div>

          <div
            class="text-xs text-slate-400 overflow-hidden transition-all duration-200"
            :style="hotkeysBodyStyle"
          >
            <div class="flex gap-3 min-w-0">
              <div class="flex-1 min-w-0">
              <p class="text-emerald-400 font-medium mb-1">真实目标 (Real)</p>
              <ul class="space-y-1">
                <li class="text-emerald-300"><kbd class="bg-emerald-900/60 border border-emerald-500 px-1 rounded text-emerald-200">1</kbd> - 小行星</li>
                <li class="text-green-300"><kbd class="bg-green-900/60 border border-green-500 px-1 rounded text-green-200">2</kbd> - 超新星</li>
                <li class="text-lime-300"><kbd class="bg-lime-900/60 border border-lime-500 px-1 rounded text-lime-200">3</kbd> - 变星</li>
              </ul>
              </div>
              <div class="flex-1 min-w-0">
              <p class="text-rose-400 font-medium mb-1">伪目标 (Bogus)</p>
              <ul class="space-y-1">
                <li class="text-rose-300"><kbd class="bg-rose-900/50 border border-rose-500 px-1 rounded text-rose-200">4</kbd> - 卫星轨迹</li>
                <li class="text-rose-300"><kbd class="bg-rose-900/50 border border-rose-500 px-1 rounded text-rose-200">5</kbd> - 噪声</li>
                <li class="text-red-300"><kbd class="bg-red-900/50 border border-red-500 px-1 rounded text-red-200">6</kbd> - 衍射芒</li>
                <li class="text-pink-300"><kbd class="bg-pink-900/50 border border-pink-500 px-1 rounded text-pink-200">7</kbd> - CMOS结露</li>
                <li class="text-rose-400"><kbd class="bg-rose-950/60 border border-rose-600 px-1 rounded text-rose-300">8</kbd> - 对应体</li>
                <li class="text-fuchsia-200"><kbd class="bg-fuchsia-950/60 border border-fuchsia-600 px-1 rounded text-fuchsia-200">9</kbd> - 消失小行星</li>
                <li class="text-fuchsia-300"><kbd class="bg-fuchsia-950/60 border border-fuchsia-600 px-1 rounded text-fuchsia-300">0</kbd> - 消失恒星</li>
                <li class="text-amber-200"><kbd class="bg-amber-950/60 border border-amber-700 px-1 rounded text-amber-200">-</kbd> - 消失星系</li>
              </ul>
              </div>
            </div>
          </div>

        </div>
        <div id="hotkeys-extra" class="mt-3" :class="leftCollapsed ? 'hidden' : ''" />
      </aside>

      <div
        class="hidden lg:block w-1 rounded bg-slate-800/70 hover:bg-sky-700/80 cursor-col-resize"
        title="拖拽调整左侧宽度"
        @mousedown.prevent="startResize('left', $event)"
      />

      <div class="flex-1 min-w-0 min-h-0 flex">
        <CanvasPanel
          :revision-overlay="activeRevisionOverlay"
          :task-refresh-key="taskRefreshKey"
          @task-changed="onTaskChanged"
          @annotations-saved="onAnnotationsSaved"
        />
      </div>

      <div
        class="hidden lg:block w-1 rounded bg-slate-800/70 hover:bg-sky-700/80 cursor-col-resize"
        title="拖拽调整右侧宽度"
        @mousedown.prevent="startResize('right', $event)"
      />

      <div class="overflow-auto min-h-0" :style="rightPaneStyle">
        <aside class="rounded-lg border border-slate-800 bg-slate-900 p-3 space-y-3 min-h-0">
          <div class="flex items-center justify-end">
            <button
              class="text-xs px-2 py-1 rounded border border-slate-700 text-slate-300"
              :title="rightCollapsed ? '展开右侧面板' : '折叠右侧面板'"
              @click="toggleRightCollapsed"
            >
              {{ rightCollapsed ? '«' : '»' }}
            </button>
          </div>
          <div v-show="!rightCollapsed">
            <InspectorPanel
              :task-id="activeTaskId"
              :refresh-key="historyRefreshKey"
              :user-role="authState.role"
              @revision-selected="onRevisionSelected"
              @revision-cleared="onRevisionCleared"
              @history-mutated="onHistoryMutated"
            />
          </div>
          <div id="inspector-extra" class="mt-3" :class="rightCollapsed ? 'hidden' : ''" />
        </aside>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { useRouter } from 'vue-router'

import CanvasPanel from '../components/CanvasPanel.vue'
import HeaderBar from '../components/HeaderBar.vue'
import InspectorPanel from '../components/InspectorPanel.vue'
import { logout } from '../services/authApi'
import { authState } from '../services/authStore'

const router = useRouter()
const activeTaskId = ref('')
const historyRefreshKey = ref(0)
const taskRefreshKey = ref(0)
const activeRevisionOverlay = ref(null)
const mainRef = ref(null)

const LEFT_MIN_WIDTH = 180
const LEFT_MAX_WIDTH = 560
const LEFT_COLLAPSED_WIDTH = 52
const RIGHT_MIN_WIDTH = 240
const RIGHT_MAX_WIDTH = 560
const RIGHT_COLLAPSED_WIDTH = 52

const leftPaneWidth = ref(220)
const rightPaneWidth = ref(320)
const leftCollapsed = ref(false)
const rightCollapsed = ref(false)
const hotkeysCollapsed = ref(false)
const resizeMode = ref('')
const isResizing = computed(() => resizeMode.value === 'left' || resizeMode.value === 'right')

const leftPaneStyle = computed(() => ({
  width: `${leftCollapsed.value ? LEFT_COLLAPSED_WIDTH : leftPaneWidth.value}px`,
  minWidth: `${leftCollapsed.value ? LEFT_COLLAPSED_WIDTH : LEFT_MIN_WIDTH}px`,
  maxWidth: `${leftCollapsed.value ? LEFT_COLLAPSED_WIDTH : LEFT_MAX_WIDTH}px`,
}))

const rightPaneStyle = computed(() => ({
  width: `${rightCollapsed.value ? RIGHT_COLLAPSED_WIDTH : rightPaneWidth.value}px`,
  minWidth: `${rightCollapsed.value ? RIGHT_COLLAPSED_WIDTH : RIGHT_MIN_WIDTH}px`,
  maxWidth: `${rightCollapsed.value ? RIGHT_COLLAPSED_WIDTH : RIGHT_MAX_WIDTH}px`,
}))

const hotkeysBodyStyle = computed(() => ({
  maxHeight: hotkeysCollapsed.value ? '0px' : '2000px',
  opacity: hotkeysCollapsed.value ? '0' : '1',
}))

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function startResize(mode) {
  resizeMode.value = mode
  window.addEventListener('mousemove', onResizeMove)
  window.addEventListener('mouseup', stopResize)
}

function onResizeMove(event) {
  const root = mainRef.value
  if (!root) {
    return
  }
  const rect = root.getBoundingClientRect()
  if (resizeMode.value === 'left' && !leftCollapsed.value) {
    const next = clamp(event.clientX - rect.left, LEFT_MIN_WIDTH, LEFT_MAX_WIDTH)
    leftPaneWidth.value = Math.round(next)
  }

  if (resizeMode.value === 'right' && !rightCollapsed.value) {
    const next = clamp(rect.right - event.clientX, RIGHT_MIN_WIDTH, RIGHT_MAX_WIDTH)
    rightPaneWidth.value = Math.round(next)
  }
}

function stopResize() {
  resizeMode.value = ''
  window.removeEventListener('mousemove', onResizeMove)
  window.removeEventListener('mouseup', stopResize)
}

function toggleLeftCollapsed() {
  leftCollapsed.value = !leftCollapsed.value
}

function toggleRightCollapsed() {
  rightCollapsed.value = !rightCollapsed.value
}

function onTaskChanged(taskId) {
  activeTaskId.value = taskId || ''
  activeRevisionOverlay.value = null
}

function onAnnotationsSaved() {
  historyRefreshKey.value += 1
  activeRevisionOverlay.value = null
}

function onRevisionSelected(revisionDetail) {
  activeRevisionOverlay.value = revisionDetail || null
}

function onRevisionCleared() {
  activeRevisionOverlay.value = null
}

function onHistoryMutated() {
  historyRefreshKey.value += 1
  taskRefreshKey.value += 1
  activeRevisionOverlay.value = null
}

function onLogout() {
  logout()
  router.push({ name: 'login' })
}

onBeforeUnmount(() => {
  stopResize()
})
</script>
