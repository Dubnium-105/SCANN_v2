<template>
  <section class="rounded-lg border border-slate-800 bg-slate-900 p-3 min-h-0">
    <div class="h-full rounded border border-slate-700 overflow-hidden relative">
      <v-stage :config="stageConfig" class="h-full w-full bg-black">
        <v-layer
          v-for="node in imageNodes"
          :key="node.view"
          :config="{ visible: node.visible }"
        >
          <v-image :config="{ image: node.image, width: stageConfig.width, height: stageConfig.height }" />
        </v-layer>
      </v-stage>

      <div
        v-if="isLoading"
        class="absolute inset-0 flex items-center justify-center text-sm text-slate-300 bg-slate-950/65"
      >
        Loading triplet images...
      </div>

      <div
        v-else-if="error"
        class="absolute inset-0 flex items-center justify-center text-sm text-rose-300 bg-slate-950/65"
      >
        {{ error }}
      </div>

      <div
        v-else-if="!activeTask"
        class="absolute inset-0 flex items-center justify-center text-sm text-slate-400 bg-slate-950/65"
      >
        Waiting for tasks...
      </div>

      <ul class="hidden" data-testid="image-state-list">
        <li
          v-for="node in imageNodes"
          :key="`debug-${node.view}`"
          data-testid="image-state-item"
          :data-view="node.view"
          :data-visible="String(node.visible)"
        />
      </ul>
    </div>
  </section>
</template>

<script setup>
import { onBeforeUnmount, onMounted } from 'vue'

import { useImageLoader } from '../composables/useImageLoader'
import { fetchTasks } from '../services/taskApi'

const stageConfig = {
  width: 1024,
  height: 768,
}

const {
  activeTask,
  error,
  imageNodes,
  isLoading,
  preloadTaskImages,
  releaseObjectUrls,
  setError,
} = useImageLoader()

async function loadInitialTask() {
  try {
    const tasks = await fetchTasks()
    const firstTask = tasks[0]
    if (!firstTask) {
      return
    }
    await preloadTaskImages(firstTask)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to load initial task'
    setError(message)
  }
}

onMounted(async () => {
  await loadInitialTask()
})

onBeforeUnmount(() => {
  releaseObjectUrls()
})
</script>
