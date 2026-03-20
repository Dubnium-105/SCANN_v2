<template>
  <section class="rounded-lg border border-slate-800 bg-slate-900 p-3 min-h-0">
    <div class="h-full flex flex-col xl:flex-row gap-3 min-h-0">
      <aside class="rounded border border-slate-700 bg-slate-950/70 p-3 space-y-3 overflow-y-auto xl:w-[320px] xl:min-w-[260px] xl:max-w-[560px] xl:resize-x">
        <div class="space-y-3">
          <div class="space-y-2">
            <p class="text-xs text-slate-200">任务切换</p>
            <p class="text-[11px] text-slate-400">{{ activeTask?.task_id || '暂无任务' }}</p>
            <p class="text-[10px] text-slate-500">进度 {{ taskProgressText }}</p>
            <div class="grid grid-cols-2 gap-2">
              <button
                data-testid="task-prev"
                class="text-xs px-2 py-1 rounded border border-slate-700 text-slate-300 disabled:opacity-50"
                :disabled="!hasPrevTask || isLoading || isSubmitting"
                @click="goToPreviousTask"
              >
                上一任务
              </button>
              <button
                data-testid="task-next"
                class="text-xs px-2 py-1 rounded border border-slate-700 text-slate-300 disabled:opacity-50"
                :disabled="!hasNextTask || isLoading || isSubmitting"
                @click="goToNextTask"
              >
                下一任务
              </button>
            </div>
            <p class="text-[10px] text-slate-500">快捷键：Q / E 切换任务</p>
          </div>

          <div class="space-y-2">
            <p class="text-xs text-slate-200">任务视图切换</p>
            <div class="grid grid-cols-3 gap-2">
              <button
                data-testid="switch-view-new"
                class="text-xs px-2 py-1 rounded border"
                :class="currentView === 'new' ? 'border-sky-400 text-sky-300' : 'border-slate-700 text-slate-300'"
                @click="switchToView('new')"
              >
                新图
              </button>
              <button
                data-testid="switch-view-new-marked"
                class="text-xs px-2 py-1 rounded border"
                :class="currentView === 'new_marked' ? 'border-sky-400 text-sky-300' : 'border-slate-700 text-slate-300'"
                @click="switchToView('new_marked')"
              >
                新图标记
              </button>
              <button
                data-testid="switch-view-old"
                class="text-xs px-2 py-1 rounded border"
                :class="currentView === 'old' ? 'border-sky-400 text-sky-300' : 'border-slate-700 text-slate-300'"
                @click="switchToView('old')"
              >
                旧图
              </button>
            </div>
            <p class="text-[10px] text-slate-400">快捷键：Tab / Space 循环切换</p>
          </div>

          <div class="space-y-2">
            <p class="text-xs text-slate-200">Tools</p>
            <div class="grid grid-cols-2 gap-2">
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
              <button
                data-testid="tool-point"
                class="text-xs px-2 py-1 rounded border"
                :class="toolMode === 'point' ? 'border-amber-400 text-amber-300' : 'border-slate-700 text-slate-300'"
                @click="setToolMode('point')"
              >
                Point
              </button>
              <button
                data-testid="tool-polygon"
                class="text-xs px-2 py-1 rounded border"
                :class="toolMode === 'polygon' ? 'border-violet-400 text-violet-300' : 'border-slate-700 text-slate-300'"
                @click="setToolMode('polygon')"
              >
                Polygon
              </button>
            </div>
            <button
              v-if="toolMode === 'polygon'"
              data-testid="finish-polygon"
              class="w-full text-xs px-2 py-1 rounded border border-violet-700 text-violet-200"
              @click="finishPolygon"
            >
              Finish Polygon
            </button>
          </div>

          <div class="space-y-2">
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
        </div>
      </aside>

      <div
        ref="canvasHostRef"
        class="rounded border border-slate-700 overflow-hidden relative min-h-[480px]"
        :class="['flex-1 min-w-0', toolMode === 'move' ? 'cursor-grab' : 'cursor-crosshair']"
        data-testid="canvas-host"
        @wheel.prevent="onContainerWheel"
      >
        <canvas
          ref="fitsCanvasRef"
          class="absolute inset-0 w-full h-full pointer-events-none z-0"
          data-testid="fits-render-canvas"
        />

        <v-stage
          ref="stageRef"
          :config="stageConfig"
          class="absolute inset-0 z-10"
          @dragend="onDragEnd"
          @wheel="onWheel"
          @mousedown="onStageMouseDown"
          @mousemove="onStageMouseMove"
          @mouseup="onStageMouseUp"
        >
          <v-layer>
            <v-rect
              v-for="ann in annotations.filter((item) => item.type === 'bbox')"
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
            <v-circle
              v-for="ann in annotations.filter((item) => item.type === 'point')"
              :key="ann.id"
              :config="{
                x: ann.x,
                y: ann.y,
                radius: 4,
                fill: '#f59e0b',
                stroke: '#fde68a',
                strokeWidth: 1,
              }"
            />
            <v-line
              v-for="ann in annotations.filter((item) => item.type === 'polygon')"
              :key="ann.id"
              :config="{
                points: toFlatPoints(ann.points),
                closed: true,
                stroke: '#a78bfa',
                strokeWidth: 2,
              }"
            />
            <v-line
              v-if="toolMode === 'polygon' && currentPolygonPoints.length > 0"
              :config="{
                points: toFlatPoints(currentPolygonPoints),
                closed: false,
                stroke: '#60a5fa',
                strokeWidth: 2,
                dash: [4, 4],
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
      </div>

      <aside class="rounded border border-slate-700 bg-slate-950/70 p-3 space-y-2 overflow-y-auto xl:w-[320px] xl:min-w-[260px] xl:max-w-[560px] xl:resize-x">
        <div class="space-y-2">
          <button
            data-testid="submit-annotations"
            class="w-full text-xs px-2 py-2 rounded border border-emerald-600 text-emerald-300 disabled:opacity-50"
            :disabled="isSubmitting || !activeTask"
            @click="submitCurrentAnnotations"
          >
            {{ isSubmitting ? 'Submitting...' : 'Submit' }}
          </button>
          <p
            v-if="saveMessage"
            data-testid="save-message"
            class="text-xs px-2 py-1 rounded bg-slate-900 text-emerald-300"
          >
            {{ saveMessage }}
          </p>

          <p class="text-[11px] text-slate-200">Annotations</p>
          <ul data-testid="annotation-list" class="max-h-28 overflow-auto space-y-1">
            <li v-for="ann in annotations" :key="`list-${ann.id}`">
              <button
                data-testid="annotation-item"
                class="w-full text-left text-[11px] px-2 py-1 rounded border"
                :class="selectedAnnotationId === ann.id ? 'border-sky-500 text-sky-200' : 'border-slate-700 text-slate-300'"
                :data-ann-id="ann.id"
                :data-ann-type="ann.type"
                :data-ann-label="ann.label"
                @click="selectAnnotation(ann.id)"
              >
                {{ ann.type }} · {{ ann.detail_type ? ann.detail_type : ann.label }}
              </button>
            </li>
          </ul>

          <label class="text-[11px] text-slate-300 block">
            Target Type (目标类型)
            <select
              data-testid="annotation-label-select"
              class="mt-1 w-full text-xs bg-slate-800 text-slate-200 border border-slate-700 rounded px-2 py-1"
              :disabled="!selectedAnnotationId"
              :value="selectedLabel"
              @change="onSelectedLabelChange"
            >
              <option value="Unlabeled">Unlabeled (未标记)</option>
              <optgroup label="Real (真实目标)">
                <option value="real:asteroid">Asteroid (小行星)</option>
                <option value="real:supernova">Supernova (超新星)</option>
                <option value="real:variable_star">Variable Star (变星)</option>
              </optgroup>
              <optgroup label="Bogus (伪目标)">
                <option value="bogus:satellite_trail">Satellite Trail (卫星轨迹)</option>
                <option value="bogus:noise">Noise (噪声)</option>
                <option value="bogus:diffraction_spike">Diffraction Spike (衍射芒)</option>
                <option value="bogus:cmos_condensation">CMOS Condensation (CMOS结露)</option>
                <option value="bogus:corresponding">Corresponding (对应体)</option>
              </optgroup>
            </select>
          </label>
        </div>
      </aside>

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

const emit = defineEmits(['task-changed', 'annotations-saved'])

const stageWidth = ref(1024)
const stageHeight = ref(768)
const stageX = ref(0)
const stageY = ref(0)
const stageScale = ref(1)
const toolMode = ref('move')
const annotations = ref([])
const draftRect = ref(null)
const drawStart = ref(null)
const currentPolygonPoints = ref([])
const taskList = ref([])
const currentTaskIndex = ref(-1)
const isSubmitting = ref(false)
const saveMessage = ref('')
const selectedAnnotationId = ref('')
const selectedLabel = ref('Unlabeled')
const fitsCanvasRef = ref(null)
const canvasHostRef = ref(null)
const stageRef = ref(null)
const stretchRangeMin = ref(0)
const stretchRangeMax = ref(1)
const stretchMin = ref(0)
const stretchMax = ref(1)
const invertDisplay = ref(false)
let hostResizeObserver = null

const taskProgressText = computed(() => {
  if (taskList.value.length === 0 || currentTaskIndex.value < 0) {
    return '0 / 0'
  }
  return `${currentTaskIndex.value + 1} / ${taskList.value.length}`
})

const hasPrevTask = computed(() => currentTaskIndex.value > 0)
const hasNextTask = computed(() => (
  currentTaskIndex.value >= 0 && currentTaskIndex.value < taskList.value.length - 1
))

const stageConfig = computed(() => ({
  width: stageWidth.value,
  height: stageHeight.value,
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

function switchToView(view) {
  if (!['new', 'new_marked', 'old'].includes(view)) {
    return
  }
  setCurrentView(view)
}

function resetAnnotationStates() {
  annotations.value = []
  selectedAnnotationId.value = ''
  selectedLabel.value = 'Unlabeled'
  draftRect.value = null
  drawStart.value = null
  currentPolygonPoints.value = []
}

async function loadTaskAtIndex(index) {
  if (index < 0 || index >= taskList.value.length) {
    return false
  }

  const task = taskList.value[index]
  await Promise.all([
    preloadTaskImages(task),
    preloadTaskFits(task),
  ])

  currentTaskIndex.value = index
  resetAnnotationStates()
  emit('task-changed', task.task_id)
  return true
}

async function goToTaskByOffset(offset) {
  if (taskList.value.length === 0 || currentTaskIndex.value < 0) {
    return
  }

  const target = currentTaskIndex.value + offset
  if (target < 0 || target >= taskList.value.length) {
    return
  }

  await loadTaskAtIndex(target)
}

async function goToPreviousTask() {
  await goToTaskByOffset(-1)
}

async function goToNextTask() {
  await goToTaskByOffset(1)
}

function onContainerWheel(event) {
  const deltaY = event?.deltaY
  if (typeof deltaY !== 'number') {
    return
  }

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

function syncStageSizeToHost() {
  const host = canvasHostRef.value
  if (!host) {
    return
  }
  const nextWidth = Math.max(320, Math.floor(host.clientWidth))
  const nextHeight = Math.max(240, Math.floor(host.clientHeight))
  stageWidth.value = nextWidth
  stageHeight.value = nextHeight
}

function setToolMode(mode) {
  toolMode.value = mode
  if (mode !== 'bbox') {
    draftRect.value = null
    drawStart.value = null
  }
  if (mode !== 'polygon') {
    currentPolygonPoints.value = []
  }
}

function toFlatPoints(points) {
  const flat = []
  for (const point of points || []) {
    flat.push(point.x, point.y)
  }
  return flat
}

function createAnnotation(base) {
  let initialLabel = 'Unlabeled'
  let initialDetail = undefined

  if (selectedLabel.value && selectedLabel.value !== 'Unlabeled') {
    const parts = selectedLabel.value.split(':')
    initialLabel = parts[0]
    if (parts.length > 1) {
      initialDetail = parts[1]
    }
  }

  return {
    id: `ann-${Date.now()}-${annotations.value.length}`,
    label: initialLabel,
    detail_type: initialDetail,
    ...base,
  }
}

function addAnnotation(annotation) {
  annotations.value.push(annotation)
  selectAnnotation(annotation.id)
}

function selectAnnotation(annotationId) {
  selectedAnnotationId.value = annotationId
  const selected = annotations.value.find((item) => item.id === annotationId)
  if (selected && selected.label && selected.label !== 'Unlabeled') {
    selectedLabel.value = selected.detail_type ? `${selected.label}:${selected.detail_type}` : selected.label
  } else {
    selectedLabel.value = 'Unlabeled'
  }
}

function onSelectedLabelChange(event) {
  const value = String(event?.target?.value ?? 'Unlabeled')
  selectedLabel.value = value
  if (!selectedAnnotationId.value) {
    return
  }

  let newLabel = 'Unlabeled'
  let newDetailType = undefined

  if (value !== 'Unlabeled') {
    const parts = value.split(':')
    newLabel = parts[0]
    if (parts.length > 1) {
      newDetailType = parts[1]
    }
  }

  annotations.value = annotations.value.map((item) =>
    item.id === selectedAnnotationId.value
      ? {
          ...item,
          label: newLabel,
          detail_type: newDetailType,
        }
      : item,
  )
}

function finishPolygon() {
  if (currentPolygonPoints.value.length < 3) {
    return
  }

  addAnnotation(
    createAnnotation({
      type: 'polygon',
      points: [...currentPolygonPoints.value],
    }),
  )
  currentPolygonPoints.value = []
}

function getPointer(event) {
  const stage = event?.target?.getStage?.()
  const stagePoint = stage?.getPointerPosition?.()
  if (stagePoint && typeof stagePoint.x === 'number' && typeof stagePoint.y === 'number') {
    try {
      const transform = stage.getAbsoluteTransform().copy()
      transform.invert()
      const converted = transform.point(stagePoint)
      return { x: converted.x, y: converted.y }
    } catch {
      return {
        x: (stagePoint.x - stageX.value) / stageScale.value,
        y: (stagePoint.y - stageY.value) / stageScale.value,
      }
    }
  }

  const raw = event?.evt ?? event
  const host = canvasHostRef.value
  if (host && typeof raw?.clientX === 'number' && typeof raw?.clientY === 'number') {
    const rect = host.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0) {
      const localX = ((raw.clientX - rect.left) / rect.width) * stageWidth.value
      const localY = ((raw.clientY - rect.top) / rect.height) * stageHeight.value
      return {
        x: (localX - stageX.value) / stageScale.value,
        y: (localY - stageY.value) / stageScale.value,
      }
    }
  }

  if (typeof raw?.offsetX === 'number' && typeof raw?.offsetY === 'number') {
    return {
      x: (raw.offsetX - stageX.value) / stageScale.value,
      y: (raw.offsetY - stageY.value) / stageScale.value,
    }
  }
  if (typeof raw?.clientX === 'number' && typeof raw?.clientY === 'number') {
    return {
      x: (raw.clientX - stageX.value) / stageScale.value,
      y: (raw.clientY - stageY.value) / stageScale.value,
    }
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
  const pointer = getPointer(event)
  if (!pointer) {
    if (toolMode.value === 'bbox') {
      draftRect.value = null
      drawStart.value = null
    }
    return
  }

  if (toolMode.value === 'point') {
    addAnnotation(
      createAnnotation({
        type: 'point',
        x: pointer.x,
        y: pointer.y,
      }),
    )
    return
  }

  if (toolMode.value === 'polygon') {
    currentPolygonPoints.value = [...currentPolygonPoints.value, { x: pointer.x, y: pointer.y }]
    return
  }

  if (toolMode.value === 'bbox' && drawStart.value) {
    const rect = normalizeRect(drawStart.value, pointer)
    if (rect.width > 0 && rect.height > 0) {
      addAnnotation(
        createAnnotation({
          type: 'bbox',
          ...rect,
        }),
      )
    }

    draftRect.value = null
    drawStart.value = null
  }
}

async function submitCurrentAnnotations() {
  saveMessage.value = ''
  const bboxAnnotations = annotations.value.filter(
    (ann) => ann.type === 'bbox' && ann.label !== 'Unlabeled'
  )

  if (!activeTask.value || bboxAnnotations.length === 0) {
    saveMessage.value = 'No valid bbox annotations to submit'
    return
  }

  isSubmitting.value = true
  try {
    const savedTaskId = activeTask.value.task_id
    const payload = {
      source_view: currentView.value,
      metadata: {
        tool: 'bbox',
        format_version: 'v2',
      },
      annotations: bboxAnnotations.map((ann) => ({
        x: ann.x,
        y: ann.y,
        width: ann.width,
        height: ann.height,
        label: ann.label,
        detail_type: ann.detail_type,
      })),
    }
    const response = await submitAnnotations(activeTask.value.task_id, payload)

    let movedToNextTask = false
    if (hasNextTask.value) {
      movedToNextTask = await loadTaskAtIndex(currentTaskIndex.value + 1)
    }

    saveMessage.value = movedToNextTask
      ? `Saved ${response.saved_count} annotations · switched to next task`
      : `Saved ${response.saved_count} annotations`

    if (!movedToNextTask) {
      resetAnnotationStates()
    }

    emit('annotations-saved', savedTaskId)
  } catch (err) {
    saveMessage.value = err instanceof Error ? err.message : 'Failed to submit annotations'
  } finally {
    isSubmitting.value = false
  }
}

async function loadInitialTask() {
  try {
    const tasks = await fetchTasks()
    taskList.value = tasks
    if (!tasks[0]) {
      return
    }
    await loadTaskAtIndex(0)
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

function onKeyDown(event) {
  if (event.key === 'q' || event.key === 'Q') {
    event.preventDefault()
    goToPreviousTask()
    return
  }

  if (event.key === 'e' || event.key === 'E') {
    event.preventDefault()
    goToNextTask()
    return
  }

  if (!selectedAnnotationId.value) return

  const keyMap = {
    '1': 'real:asteroid',
    '2': 'real:supernova',
    '3': 'real:variable_star',
    '4': 'bogus:satellite_trail',
    '5': 'bogus:noise',
    '6': 'bogus:diffraction_spike',
    '7': 'bogus:cmos_condensation',
    '8': 'bogus:corresponding',
  }

  const newLabelStr = keyMap[event.key]
  if (newLabelStr) {
    onSelectedLabelChange({ target: { value: newLabelStr } })
  } else if (event.key === 'Backspace' || event.key === 'Delete') {
    annotations.value = annotations.value.filter((ann) => ann.id !== selectedAnnotationId.value)
    selectedAnnotationId.value = ''
    selectedLabel.value = 'Unlabeled'
  }
}

onMounted(async () => {
  syncStageSizeToHost()
  if (typeof ResizeObserver !== 'undefined') {
    hostResizeObserver = new ResizeObserver(() => {
      syncStageSizeToHost()
    })
    if (canvasHostRef.value) {
      hostResizeObserver.observe(canvasHostRef.value)
    }
  }
  window.addEventListener('keydown', onKeyDown)
  await loadInitialTask()
})

onBeforeUnmount(() => {
  if (hostResizeObserver) {
    hostResizeObserver.disconnect()
    hostResizeObserver = null
  }
  window.removeEventListener('keydown', onKeyDown)
  releaseObjectUrls()
})
</script>
