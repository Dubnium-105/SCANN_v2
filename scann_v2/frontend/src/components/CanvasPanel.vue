<template>
  <section class="rounded-lg border border-slate-800 bg-slate-900 p-3 min-h-0">
    <div class="h-full rounded border border-slate-700 overflow-hidden relative">
      <v-stage
        :config="stageConfig"
        class="h-full w-full bg-black"
        @dragend="onDragEnd"
        @wheel="onWheel"
        @mousedown="onStageMouseDown"
        @mousemove="onStageMouseMove"
        @mouseup="onStageMouseUp"
      >
        <v-layer
          v-for="node in imageNodes"
          :key="node.view"
          :config="{ visible: node.visible }"
        >
          <v-image :config="{ image: node.image, width: stageConfig.width, height: stageConfig.height }" />
        </v-layer>

        <v-layer>
          <v-rect
            v-for="ann in annotations"
            :key="ann.id"
            :config="{
              x: ann.x,
              y: ann.y,
              width: ann.width,
              height: ann.height,
              stroke: '#22c55e',
              strokeWidth: 2,
            }"
          />
          <v-rect
            v-if="draftRect"
            :config="{
              x: draftRect.x,
              y: draftRect.y,
              width: draftRect.width,
              height: draftRect.height,
              stroke: '#38bdf8',
              strokeWidth: 2,
              dash: [6, 4],
            }"
          />
        </v-layer>
      </v-stage>

      <canvas
        ref="fitsCanvasRef"
        class="absolute inset-0 w-full h-full pointer-events-none"
        data-testid="fits-render-canvas"
      />

      <div class="absolute top-2 left-2 flex items-center gap-2 bg-slate-950/70 rounded px-2 py-1">
        <button
          data-testid="tool-move"
          class="text-xs px-2 py-1 rounded border"
          :class="toolMode === 'move' ? 'border-sky-400 text-sky-300' : 'border-slate-700 text-slate-300'"
          @click="setToolMode('move')"
        >
          Move
        </button>
        <button
          data-testid="tool-bbox"
          class="text-xs px-2 py-1 rounded border"
          :class="toolMode === 'bbox' ? 'border-emerald-400 text-emerald-300' : 'border-slate-700 text-slate-300'"
          @click="setToolMode('bbox')"
        >
          BBox
        </button>
      </div>

      <div class="absolute top-2 right-2 flex items-center gap-2 bg-slate-950/70 rounded px-2 py-1">
        <select
          v-model="selectedBucket"
          data-testid="bucket-select"
          class="text-xs bg-slate-800 text-slate-200 border border-slate-700 rounded px-2 py-1"
        >
          <option value="positive">positive</option>
          <option value="negative">negative</option>
        </select>
        <button
          data-testid="submit-annotations"
          class="text-xs px-2 py-1 rounded border border-emerald-600 text-emerald-300 disabled:opacity-50"
          :disabled="isSubmitting || !activeTask"
          @click="submitCurrentAnnotations"
        >
          {{ isSubmitting ? 'Submitting...' : 'Submit' }}
        </button>
      </div>

      <p
        v-if="saveMessage"
        data-testid="save-message"
        class="absolute bottom-2 left-2 text-xs px-2 py-1 rounded bg-slate-950/70 text-emerald-300"
      >
        {{ saveMessage }}
      </p>

      <div class="absolute bottom-2 right-2 bg-slate-950/70 rounded px-2 py-2 w-72 space-y-2">
        <p class="text-[11px] text-slate-200">Stretch</p>
        <div class="space-y-1">
          <label class="text-[10px] text-slate-400">Min: {{ stretchMin.toFixed(2) }}</label>
          <input
            data-testid="stretch-min-slider"
            type="range"
            class="w-full"
            :min="stretchRangeMin"
            :max="stretchRangeMax"
            step="0.01"
            :value="stretchMin"
            @input="onStretchMinInput"
          >
        </div>
        <div class="space-y-1">
          <label class="text-[10px] text-slate-400">Max: {{ stretchMax.toFixed(2) }}</label>
          <input
            data-testid="stretch-max-slider"
            type="range"
            class="w-full"
            :min="stretchRangeMin"
            :max="stretchRangeMax"
            step="0.01"
            :value="stretchMax"
            @input="onStretchMaxInput"
          >
        </div>
        <label class="text-[11px] text-slate-300 inline-flex items-center gap-2">
          <input
            data-testid="invert-toggle"
            type="checkbox"
            :checked="invertDisplay"
            @change="onInvertChange"
          >
          Invert
        </label>
      </div>

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
        v-else-if="fitsError"
        class="absolute inset-0 flex items-center justify-center text-sm text-rose-300 bg-slate-950/65"
      >
        {{ fitsError }}
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

      <span
        class="hidden"
        data-testid="stretch-debug"
        :data-rgba="stretchDebug"
      />
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import { useBlinkControl } from '../composables/useBlinkControl'
import { useFitsImagePool } from '../composables/useFitsImagePool'
import { useImageLoader } from '../composables/useImageLoader'
import { calculatePixelRange, renderStretchToRgba } from '../fits/stretchRenderer'
import { submitAnnotations } from '../services/annotationApi'
import { fetchTasks } from '../services/taskApi'

const stageWidth = 1024
const stageHeight = 768
const stageX = ref(0)
const stageY = ref(0)
const stageScale = ref(1)
const toolMode = ref('move')
const annotations = ref([])
const draftRect = ref(null)
const drawStart = ref(null)
const selectedBucket = ref('positive')
const isSubmitting = ref(false)
const saveMessage = ref('')
const fitsCanvasRef = ref(null)
const stretchRangeMin = ref(0)
const stretchRangeMax = ref(1)
const stretchMin = ref(0)
const stretchMax = ref(1)
const invertDisplay = ref(false)

const stageConfig = computed(() => ({
  width: stageWidth,
  height: stageHeight,
  draggable: toolMode.value === 'move',
  x: stageX.value,
  y: stageY.value,
  scaleX: stageScale.value,
  scaleY: stageScale.value,
}))

const {
  activeTask,
  currentView,
  error,
  imageNodes,
  isLoading,
  preloadTaskImages,
  releaseObjectUrls,
  setError,
  setCurrentView,
} = useImageLoader()

const {
  fitsError,
  fitsNodes,
  preloadTaskFits,
} = useFitsImagePool()

const activeFitsNode = computed(
  () => fitsNodes.value.find((node) => node.view === currentView.value) ?? null,
)

const stretchedRgba = computed(() => {
  const pixels = activeFitsNode.value?.pixels
  return renderStretchToRgba(pixels, stretchMin.value, stretchMax.value, invertDisplay.value)
})

const stretchDebug = computed(() => Array.from(stretchedRgba.value.slice(0, 16)).join(','))

useBlinkControl({
  currentView,
  setCurrentView,
})

function onDragEnd(event) {
  const position = event?.target?.position?.()
  if (!position) {
    return
  }

  stageX.value = position.x
  stageY.value = position.y
}

function onWheel(event) {
  const deltaY = event?.evt?.deltaY
  if (typeof deltaY !== 'number') {
    return
  }

  event.evt.preventDefault()
  const direction = deltaY > 0 ? -1 : 1
  const nextScale = stageScale.value * (direction > 0 ? 1.05 : 0.95)
  stageScale.value = Math.max(0.1, Math.min(10, nextScale))
}

function onStretchMinInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  stretchMin.value = Math.min(value, stretchMax.value)
}

function onStretchMaxInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  stretchMax.value = Math.max(value, stretchMin.value)
}

function onInvertChange(event) {
  invertDisplay.value = Boolean(event?.target?.checked)
}

function redrawFitsCanvas() {
  const canvas = fitsCanvasRef.value
  const node = activeFitsNode.value
  if (!canvas || !node || !node.width || !node.height) {
    return
  }

  canvas.width = node.width
  canvas.height = node.height

  let context = null
  try {
    context = canvas.getContext('2d')
  } catch {
    context = null
  }
  if (!context) {
    return
  }

  if (typeof ImageData !== 'undefined') {
    const imageData = new ImageData(stretchedRgba.value, node.width, node.height)
    context.putImageData(imageData, 0, 0)
  }
}

function setToolMode(mode) {
  toolMode.value = mode
  if (mode !== 'bbox') {
    draftRect.value = null
    drawStart.value = null
  }
}

function getPointer(event) {
  const stage = event?.target?.getStage?.()
  const stagePoint = stage?.getPointerPosition?.()
  if (stagePoint && typeof stagePoint.x === 'number' && typeof stagePoint.y === 'number') {
    return { x: stagePoint.x, y: stagePoint.y }
  }

  const raw = event?.evt ?? event
  if (typeof raw?.offsetX === 'number' && typeof raw?.offsetY === 'number') {
    return { x: raw.offsetX, y: raw.offsetY }
  }
  if (typeof raw?.clientX === 'number' && typeof raw?.clientY === 'number') {
    return { x: raw.clientX, y: raw.clientY }
  }
  return null
}

function normalizeRect(start, current) {
  return {
    x: Math.min(start.x, current.x),
    y: Math.min(start.y, current.y),
    width: Math.abs(current.x - start.x),
    height: Math.abs(current.y - start.y),
  }
}

function onStageMouseDown(event) {
  if (toolMode.value !== 'bbox') {
    return
  }

  const pointer = getPointer(event)
  if (!pointer) {
    return
  }

  drawStart.value = pointer
  draftRect.value = {
    x: pointer.x,
    y: pointer.y,
    width: 0,
    height: 0,
  }
}

function onStageMouseMove(event) {
  if (toolMode.value !== 'bbox' || !drawStart.value) {
    return
  }

  const pointer = getPointer(event)
  if (!pointer) {
    return
  }

  draftRect.value = normalizeRect(drawStart.value, pointer)
}

function onStageMouseUp(event) {
  if (toolMode.value !== 'bbox' || !drawStart.value) {
    return
  }

  const pointer = getPointer(event)
  if (!pointer) {
    draftRect.value = null
    drawStart.value = null
    return
  }

  const rect = normalizeRect(drawStart.value, pointer)
  if (rect.width > 0 && rect.height > 0) {
    annotations.value.push({
      id: `ann-${Date.now()}-${annotations.value.length}`,
      ...rect,
    })
  }

  draftRect.value = null
  drawStart.value = null
}

async function submitCurrentAnnotations() {
  saveMessage.value = ''
  if (!activeTask.value || annotations.value.length === 0) {
    saveMessage.value = 'No annotations to submit'
    return
  }

  isSubmitting.value = true
  try {
    const payload = {
      bucket: selectedBucket.value,
      source_view: currentView.value,
      metadata: {
        tool: 'bbox',
      },
      annotations: annotations.value.map((ann) => ({
        x: ann.x,
        y: ann.y,
        width: ann.width,
        height: ann.height,
        label: 'BBox',
      })),
    }
    const response = await submitAnnotations(activeTask.value.task_id, payload)
    saveMessage.value = `Saved ${response.saved_count} annotations`
    annotations.value = []
  } catch (err) {
    saveMessage.value = err instanceof Error ? err.message : 'Failed to submit annotations'
  } finally {
    isSubmitting.value = false
  }
}

async function loadInitialTask() {
  try {
    const tasks = await fetchTasks()
    const firstTask = tasks[0]
    if (!firstTask) {
      return
    }
    await Promise.all([
      preloadTaskImages(firstTask),
      preloadTaskFits(firstTask),
    ])
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to load initial task'
    setError(message)
  }
}

watch(activeFitsNode, (node) => {
  if (!node?.pixels || node.pixels.length === 0) {
    return
  }

  const range = calculatePixelRange(node.pixels)
  stretchRangeMin.value = range.min
  stretchRangeMax.value = range.max
  stretchMin.value = range.min
  stretchMax.value = range.max
})

watch(
  [stretchedRgba, activeFitsNode],
  () => {
    redrawFitsCanvas()
  },
  { immediate: true },
)

onMounted(async () => {
  await loadInitialTask()
})

onBeforeUnmount(() => {
  releaseObjectUrls()
})
</script>
