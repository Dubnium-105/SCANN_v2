<template>
  <div class="h-full grid grid-rows-[56px_1fr]">
    <HeaderBar :username="authState.username" @logout="onLogout" />
    <main class="flex flex-col lg:flex-row gap-3 p-3 min-h-0">
      <aside class="hidden lg:block rounded-lg border border-slate-800 bg-slate-900 p-3 overflow-y-auto lg:w-[220px] lg:min-w-[180px] lg:max-w-[420px] lg:resize-x">
        <p class="text-sm font-semibold text-slate-200 mb-2">快捷键 (Hotkeys)</p>
        <div class="text-xs text-slate-400 space-y-3">
          <div>
            <p class="text-emerald-400 font-medium mb-1">真实目标 (Real)</p>
            <ul class="space-y-1">
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">1</kbd> - 小行星</li>
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">2</kbd> - 超新星</li>
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">3</kbd> - 变星</li>
            </ul>
          </div>
          <div>
            <p class="text-rose-400 font-medium mb-1">伪目标 (Bogus)</p>
            <ul class="space-y-1">
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">4</kbd> - 卫星轨迹</li>
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">5</kbd> - 噪声</li>
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">6</kbd> - 衍射芒</li>
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">7</kbd> - CMOS结露</li>
              <li><kbd class="bg-slate-700 px-1 rounded text-slate-200">8</kbd> - 对应体</li>
            </ul>
          </div>
        </div>
      </aside>

      <div class="flex-1 min-w-0 min-h-0">
        <CanvasPanel @task-changed="onTaskChanged" @annotations-saved="onAnnotationsSaved" />
      </div>
      <div class="lg:w-[320px] lg:min-w-[240px] lg:max-w-[560px] lg:resize-x overflow-auto">
        <InspectorPanel :task-id="activeTaskId" :refresh-key="historyRefreshKey" />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import CanvasPanel from '../components/CanvasPanel.vue'
import HeaderBar from '../components/HeaderBar.vue'
import InspectorPanel from '../components/InspectorPanel.vue'
import { logout } from '../services/authApi'
import { authState } from '../services/authStore'

const router = useRouter()
const activeTaskId = ref('')
const historyRefreshKey = ref(0)

function onTaskChanged(taskId) {
  activeTaskId.value = taskId || ''
}

function onAnnotationsSaved(taskId) {
  activeTaskId.value = taskId || activeTaskId.value
  historyRefreshKey.value += 1
}

function onLogout() {
  logout()
  router.push({ name: 'login' })
}
</script>
