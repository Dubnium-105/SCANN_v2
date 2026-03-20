<template>
  <aside class="rounded-lg border border-slate-800 bg-slate-900 p-3 space-y-3">
    <p class="text-sm text-slate-300">Inspector</p>

    <div class="rounded border border-slate-800 bg-slate-950/30 p-2">
      <p class="text-xs text-slate-300">Version History</p>
      <p v-if="!taskId" class="text-xs text-slate-500 mt-2" data-testid="history-empty-task">No task selected</p>
      <p v-else-if="isLoading" class="text-xs text-slate-400 mt-2" data-testid="history-loading">Loading history...</p>
      <p v-else-if="errorMessage" class="text-xs text-rose-300 mt-2" data-testid="history-error">{{ errorMessage }}</p>
      <ul v-else data-testid="history-list" class="mt-2 space-y-1 max-h-52 overflow-auto">
        <li
          v-for="revision in revisions"
          :key="revision.revision_id"
          class="rounded border border-slate-800 px-2 py-1 text-xs text-slate-300"
        >
          <p data-testid="history-item-user">{{ revision.submitted_by }}</p>
          <p class="text-slate-500">{{ revision.saved_at }}</p>
          <p class="text-slate-400">{{ revision.format_version || 'v2' }} · {{ revision.annotation_count }} items</p>
        </li>
      </ul>
    </div>
  </aside>
</template>

<script setup>
import { ref, watch } from 'vue'

import { fetchAnnotationHistory } from '../services/annotationHistoryApi'

const props = defineProps({
  taskId: {
    type: String,
    default: '',
  },
  refreshKey: {
    type: Number,
    default: 0,
  },
})

const revisions = ref([])
const isLoading = ref(false)
const errorMessage = ref('')

async function loadHistory() {
  revisions.value = []
  errorMessage.value = ''

  if (!props.taskId) {
    return
  }

  isLoading.value = true
  try {
    const response = await fetchAnnotationHistory(props.taskId)
    revisions.value = response.revisions || []
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : 'Failed to load history'
  } finally {
    isLoading.value = false
  }
}

watch(
  () => [props.taskId, props.refreshKey],
  () => {
    loadHistory()
  },
  { immediate: true },
)
</script>
