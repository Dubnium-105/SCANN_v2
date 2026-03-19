<template>
  <div class="h-full grid grid-rows-[56px_1fr]">
    <HeaderBar />
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
import HeaderBar from './components/HeaderBar.vue'
import CanvasPanel from './components/CanvasPanel.vue'
import InspectorPanel from './components/InspectorPanel.vue'
</script>
