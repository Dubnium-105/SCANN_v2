<template>
  <div class="h-full grid grid-rows-[56px_1fr]">
    <HeaderBar :username="authState.username" @logout="onLogout" />
    <main class="grid grid-cols-[220px_1fr_300px] gap-3 p-3 min-h-0">
      <aside class="rounded-lg border border-slate-800 bg-slate-900 p-3">
        <p class="text-sm text-slate-300">Tools</p>
      </aside>

      <CanvasPanel />
      <InspectorPanel />
    </main>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

import CanvasPanel from '../components/CanvasPanel.vue'
import HeaderBar from '../components/HeaderBar.vue'
import InspectorPanel from '../components/InspectorPanel.vue'
import { logout } from '../services/authApi'
import { authState } from '../services/authStore'

const router = useRouter()

function onLogout() {
  logout()
  router.push({ name: 'login' })
}
</script>
