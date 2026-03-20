<template>
  <aside class="rounded-lg border border-slate-800 bg-slate-900 p-3 space-y-3">
    <p class="text-sm text-slate-300">Inspector</p>

    <div class="rounded border border-slate-800 bg-slate-950/30 p-2">
      <div class="flex items-center justify-between gap-2">
        <p class="text-xs text-slate-300">Version History</p>
        <button
          data-testid="history-clear-selection"
          class="text-[10px] px-2 py-0.5 rounded border border-slate-700 text-slate-400"
          :disabled="!selectedRevisionId"
          @click="clearSelection"
        >
          清除对比
        </button>
      </div>
      <p v-if="!taskId" class="text-xs text-slate-500 mt-2" data-testid="history-empty-task">No task selected</p>
      <p v-else-if="isLoading" class="text-xs text-slate-400 mt-2" data-testid="history-loading">Loading history...</p>
      <p v-else-if="errorMessage" class="text-xs text-rose-300 mt-2" data-testid="history-error">{{ errorMessage }}</p>
      <ul v-else data-testid="history-list" class="mt-2 space-y-1 max-h-52 overflow-auto">
        <li
          v-for="revision in revisions"
          :key="revision.revision_id"
          data-testid="history-item"
          class="rounded border px-2 py-1 text-xs"
          :class="selectedRevisionId === revision.revision_id ? 'border-sky-500 text-sky-100 bg-sky-950/20' : 'border-slate-800 text-slate-300'"
        >
          <div class="flex items-start gap-2">
            <button class="flex-1 text-left" @click="selectRevision(revision.revision_id)">
              <p data-testid="history-item-user">{{ revision.submitted_by }}</p>
              <p class="text-slate-500">{{ revision.saved_at }}</p>
              <p class="text-slate-400">{{ revision.format_version || 'v2' }} · {{ revision.annotation_count }} items</p>
              <p class="text-[10px] mt-1 text-slate-500">
                +{{ revision.change_summary?.added || 0 }}
                / ~{{ revision.change_summary?.modified || 0 }}
                / -{{ revision.change_summary?.removed || 0 }}
              </p>
              <p v-if="revision.rollback_of_revision_id" class="text-[10px] text-amber-300 mt-1">
                rollback of {{ revision.rollback_of_revision_id.slice(0, 8) }}
              </p>
            </button>
            <button
              v-if="userRole === 'admin'"
              data-testid="history-row-rollback"
              class="shrink-0 text-[10px] px-2 py-1 rounded border border-rose-700 text-rose-300 disabled:opacity-50"
              :disabled="isRollbackLoading"
              @click="rollbackRevision(revision.revision_id)"
            >
              回到这个状态
            </button>
          </div>
        </li>
      </ul>

      <div v-if="selectedRevision" data-testid="history-detail" class="mt-3 rounded border border-slate-800 bg-slate-950/40 p-2">
        <p class="text-xs text-slate-200">提交详情</p>
        <p class="text-[11px] text-slate-400 mt-1">提交人：{{ selectedRevision.submitted_by }}</p>
        <p class="text-[11px] text-slate-400">时间：{{ selectedRevision.saved_at }}</p>
        <p class="text-[11px] text-slate-400">对比：{{ selectedRevision.parent_revision_id ? selectedRevision.parent_revision_id.slice(0, 8) : '初始提交' }}</p>
        <p class="text-[11px] text-slate-300 mt-1">
          新增 {{ selectedRevision.change_summary?.added || 0 }} ·
          修改 {{ selectedRevision.change_summary?.modified || 0 }} ·
          删除 {{ selectedRevision.change_summary?.removed || 0 }}
        </p>

        <ul class="mt-2 max-h-36 overflow-auto space-y-1">
          <li
            v-for="(item, index) in selectedRevision.changed_items || []"
            :key="`${item.change_type}-${index}`"
            class="text-[10px] rounded border border-slate-800 px-2 py-1 text-slate-400"
          >
            <span
              :class="item.change_type === 'added' ? 'text-emerald-300' : item.change_type === 'removed' ? 'text-rose-300' : 'text-amber-300'"
            >
              {{ item.change_type }}
            </span>
            <span v-if="item.changed_fields?.length"> · {{ item.changed_fields.join(',') }}</span>
          </li>
        </ul>

        <button
          v-if="userRole === 'admin'"
          data-testid="history-rollback"
          class="mt-2 w-full text-xs px-2 py-1 rounded border border-rose-700 text-rose-300 disabled:opacity-50"
          :disabled="isRollbackLoading"
          @click="rollbackRevision(selectedRevisionId)"
        >
          {{ isRollbackLoading ? '回退中...' : '回退到此提交' }}
        </button>
        <p v-if="rollbackMessage" class="text-[10px] mt-1 text-emerald-300">{{ rollbackMessage }}</p>
        <p v-if="rollbackError" class="text-[10px] mt-1 text-rose-300">{{ rollbackError }}</p>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, watch } from 'vue'

import {
  fetchAnnotationHistory,
  fetchAnnotationRevision,
  rollbackAnnotationRevision,
} from '../services/annotationHistoryApi'

const emit = defineEmits(['revision-selected', 'revision-cleared', 'history-mutated'])

const props = defineProps({
  taskId: {
    type: String,
    default: '',
  },
  refreshKey: {
    type: Number,
    default: 0,
  },
  userRole: {
    type: String,
    default: '',
  },
})

const revisions = ref([])
const isLoading = ref(false)
const errorMessage = ref('')
const selectedRevisionId = ref('')
const selectedRevision = ref(null)
const isRollbackLoading = ref(false)
const rollbackMessage = ref('')
const rollbackError = ref('')
const lastTaskId = ref('')

function clearSelection() {
  selectedRevisionId.value = ''
  selectedRevision.value = null
  rollbackMessage.value = ''
  rollbackError.value = ''
  emit('revision-cleared')
}

async function selectRevision(revisionId) {
  if (!props.taskId || !revisionId) {
    return
  }
  rollbackMessage.value = ''
  rollbackError.value = ''
  selectedRevisionId.value = revisionId

  try {
    const detail = await fetchAnnotationRevision(props.taskId, revisionId)
    selectedRevision.value = detail
    emit('revision-selected', detail)
  } catch (err) {
    selectedRevision.value = null
    emit('revision-cleared')
    errorMessage.value = err instanceof Error ? err.message : 'Failed to load revision detail'
  }
}

async function rollbackRevision(revisionId) {
  if (!props.taskId || !revisionId || props.userRole !== 'admin') {
    return
  }
  isRollbackLoading.value = true
  rollbackMessage.value = ''
  rollbackError.value = ''
  try {
    const result = await rollbackAnnotationRevision(props.taskId, revisionId)
    rollbackMessage.value = `已回退，创建提交 ${String(result.new_revision_id || '').slice(0, 8)}`
    emit('history-mutated')
    await loadHistory()
    await selectRevision(result.new_revision_id)
  } catch (err) {
    rollbackError.value = err instanceof Error ? err.message : 'Failed to rollback revision'
  } finally {
    isRollbackLoading.value = false
  }
}

async function loadHistory() {
  revisions.value = []
  errorMessage.value = ''
  rollbackMessage.value = ''
  rollbackError.value = ''

  if (!props.taskId) {
    clearSelection()
    return
  }

  isLoading.value = true
  try {
    const response = await fetchAnnotationHistory(props.taskId)
    revisions.value = response.revisions || []

    if (!selectedRevisionId.value) {
      emit('revision-cleared')
      return
    }
    const matched = revisions.value.find((item) => item.revision_id === selectedRevisionId.value)
    if (!matched) {
      clearSelection()
      return
    }
    await selectRevision(selectedRevisionId.value)
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : 'Failed to load history'
    emit('revision-cleared')
  } finally {
    isLoading.value = false
  }
}

watch(
  () => [props.taskId, props.refreshKey],
  () => {
    if (props.taskId !== lastTaskId.value) {
      clearSelection()
      lastTaskId.value = props.taskId
    }
    loadHistory()
  },
  { immediate: true },
)
</script>
