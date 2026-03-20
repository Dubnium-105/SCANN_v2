<template>
  <section class="rounded-lg border border-slate-800 bg-slate-900 p-3 h-full w-full min-h-0">
    <div class="h-full flex flex-col xl:flex-row gap-3 min-h-0">
      <Teleport v-if="hotkeysTeleportTarget" :to="hotkeysTeleportTarget">
        <aside class="w-full rounded border border-slate-700 bg-slate-950/70 p-3 space-y-3">
          <div class="space-y-3">
            <div class="space-y-2">
              <div class="flex items-center justify-between gap-2">
                <p class="text-xs text-slate-200">任务切换</p>
                <button
                  data-testid="task-catalog-open"
                  class="text-[10px] px-2 py-0.5 rounded border border-slate-700 text-slate-300"
                  @click="openTaskCatalog"
                >
                  总任务目录
                </button>
              </div>
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
              <div class="flex items-center justify-between gap-2">
                <p class="text-[11px] text-slate-200">Stretch</p>
                <label class="text-[10px] text-slate-300 inline-flex items-center gap-1">
                  <input
                    data-testid="auto-stretch-toggle"
                    type="checkbox"
                    :checked="autoStretchEnabled"
                    @change="onAutoStretchToggle"
                  >
                  自动拉伸
                </label>
              </div>
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

            <div class="space-y-2">
              <p class="text-[11px] text-slate-200">BBox 线宽</p>
              <div class="space-y-1">
                <label class="text-[10px] text-slate-400">线宽: {{ bboxStrokeWidth.toFixed(1) }}</label>
                <input
                  data-testid="bbox-stroke-width-slider"
                  type="range"
                  class="w-full"
                  min="1"
                  max="8"
                  step="0.5"
                  :value="bboxStrokeWidth"
                  @input="onBboxStrokeWidthInput"
                >
              </div>
            </div>
          </div>
        </aside>
      </Teleport>

      <div
        v-if="taskCatalogVisible"
        class="fixed inset-0 z-50 bg-black/50 backdrop-blur-[1px] flex items-center justify-center p-4"
        data-testid="task-catalog-modal"
        @click.self="closeTaskCatalog"
      >
        <aside class="w-full max-w-xl rounded-lg border border-slate-700 bg-slate-950 p-3 space-y-3 max-h-[80vh] flex flex-col">
          <div class="flex items-center justify-between gap-2">
            <p class="text-sm text-slate-200">总任务目录</p>
            <button
              data-testid="task-catalog-close"
              class="text-xs px-2 py-1 rounded border border-slate-700 text-slate-300"
              @click="closeTaskCatalog"
            >
              关闭
            </button>
          </div>
          <input
            v-model="taskCatalogQuery"
            data-testid="task-catalog-search"
            type="text"
            class="w-full text-xs bg-slate-900 text-slate-200 border border-slate-700 rounded px-2 py-1"
            placeholder="搜索任务ID..."
          >
          <p class="text-[11px] text-slate-500">共 {{ filteredTaskCatalog.length }} / {{ taskList.length }} 个任务</p>
          <ul class="flex-1 min-h-0 overflow-auto space-y-1" data-testid="task-catalog-list">
            <li
              v-for="item in filteredTaskCatalog"
              :key="item.task.task_id"
            >
              <button
                class="w-full text-left text-xs px-2 py-1 rounded border"
                :class="currentTaskIndex === item.index ? 'border-sky-500 text-sky-200 bg-sky-950/20' : 'border-slate-700 text-slate-300'"
                @click="jumpToTaskIndex(item.index)"
              >
                {{ item.task.task_id }}
              </button>
            </li>
          </ul>
        </aside>
      </div>

      <div
        ref="canvasHostRef"
        class="rounded border border-slate-700 overflow-hidden relative min-h-[480px]"
        :class="[
          'flex-1 min-w-0',
          middlePanActive ? 'cursor-grabbing' : (toolMode === 'move' ? 'cursor-grab' : 'cursor-crosshair'),
        ]"
        data-testid="canvas-host"
        @wheel.prevent="onContainerWheel"
      >
        <canvas
          ref="fitsCanvasRef"
          class="absolute inset-0 pointer-events-none z-0"
          :style="fitsCanvasStyle"
          data-testid="fits-render-canvas"
        />

        <v-stage
          ref="stageRef"
          :config="stageConfig"
          class="absolute inset-0 z-10"
          @dragmove="onDragMove"
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
                stroke: getAnnotationColor(ann),
                strokeWidth: bboxStrokeWidth,
              }"
            />
            <v-rect
              v-for="(overlay, index) in revisionOverlayRemovedRects"
              :key="`overlay-removed-${index}`"
              :config="{
                x: overlay.x,
                y: overlay.y,
                width: overlay.width,
                height: overlay.height,
                stroke: '#fb7185',
                strokeWidth: Math.max(1.5, bboxStrokeWidth),
                dash: [8, 4],
              }"
            />
            <v-rect
              v-for="(overlay, index) in revisionOverlayModifiedBeforeRects"
              :key="`overlay-modified-before-${index}`"
              :config="{
                x: overlay.x,
                y: overlay.y,
                width: overlay.width,
                height: overlay.height,
                stroke: '#f59e0b',
                strokeWidth: Math.max(1.5, bboxStrokeWidth),
                dash: [5, 3],
              }"
            />
            <v-rect
              v-for="(overlay, index) in revisionOverlayModifiedAfterRects"
              :key="`overlay-modified-after-${index}`"
              :config="{
                x: overlay.x,
                y: overlay.y,
                width: overlay.width,
                height: overlay.height,
                stroke: '#fde047',
                strokeWidth: Math.max(1.5, bboxStrokeWidth),
              }"
            />
            <v-rect
              v-for="(overlay, index) in revisionOverlayAddedRects"
              :key="`overlay-added-${index}`"
              :config="{
                x: overlay.x,
                y: overlay.y,
                width: overlay.width,
                height: overlay.height,
                stroke: '#2dd4bf',
                strokeWidth: Math.max(1.5, bboxStrokeWidth),
                dash: [2, 2],
              }"
            />
            <v-circle
              v-for="ann in annotations.filter((item) => item.type === 'point')"
              :key="ann.id"
              :config="{
                x: ann.x,
                y: ann.y,
                radius: 4,
                fill: getAnnotationColor(ann),
                stroke: getAnnotationColor(ann),
                strokeWidth: 1,
              }"
            />
            <v-line
              v-for="ann in annotations.filter((item) => item.type === 'polygon')"
              :key="ann.id"
              :config="{
                points: toFlatPoints(ann.points),
                closed: true,
                stroke: getAnnotationColor(ann),
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
                strokeWidth: bboxStrokeWidth,
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

      <Teleport v-if="inspectorTeleportTarget" :to="inspectorTeleportTarget">
        <aside class="w-full rounded border border-slate-700 bg-slate-950/70 p-3 space-y-2 overflow-y-auto">
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
            <div
              v-if="undoDeleteVisible"
              data-testid="undo-delete-banner"
              class="text-xs px-2 py-1 rounded bg-amber-950/40 border border-amber-700/50 text-amber-200 flex items-center justify-between gap-2"
            >
              <span>{{ undoDeleteMessage }}</span>
              <button
                data-testid="undo-delete"
                class="px-2 py-0.5 rounded border border-amber-500 text-amber-100"
                @click="undoRemoveAnnotation"
              >
                Undo
              </button>
            </div>

            <p class="text-[11px] text-slate-200">Annotations</p>
            <ul data-testid="annotation-list" class="max-h-28 overflow-auto space-y-1">
              <li v-for="ann in annotations" :key="`list-${ann.id}`">
                <div class="flex items-center gap-1">
                  <button
                    data-testid="annotation-item"
                    class="flex-1 text-left text-[11px] px-2 py-1 rounded border"
                    :class="selectedAnnotationId === ann.id ? 'border-sky-500 text-sky-200 ring-1 ring-sky-500/30' : 'border-slate-700 text-slate-300'"
                    :style="getAnnotationItemStyle(ann, selectedAnnotationId === ann.id)"
                    :data-ann-id="ann.id"
                    :data-ann-display-id="ann.display_id"
                    :data-ann-type="ann.type"
                    :data-ann-label="ann.label"
                    @click="selectAnnotation(ann.id)"
                  >
                      <span class="inline-flex items-center gap-2">
                        <span
                          class="inline-block w-2 h-2 rounded-full"
                          :style="{ backgroundColor: getAnnotationColor(ann) }"
                        />
                        <span class="font-semibold">{{ ann.display_id }}</span>
                        <span>{{ ann.type }} · {{ ann.detail_type ? ann.detail_type : ann.label }}</span>
                      </span>
                  </button>
                  <button
                    data-testid="annotation-remove"
                    class="text-[10px] px-2 py-1 rounded border hover:bg-rose-900/20"
                    :class="pendingDeleteAnnotationId === ann.id ? 'border-rose-500 text-rose-100 bg-rose-900/30' : 'border-rose-800 text-rose-300'"
                    title="删除该标注"
                    @click.stop="removeAnnotation(ann.id)"
                  >
                    {{ pendingDeleteAnnotationId === ann.id ? '确认删除' : '删除' }}
                  </button>
                </div>
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
      </Teleport>

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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

import { useBlinkControl } from '../composables/useBlinkControl'
import { useFitsImagePool } from '../composables/useFitsImagePool'
import { useImageLoader } from '../composables/useImageLoader'
import { calculatePixelRange, renderStretchToRgba } from '../fits/stretchRenderer'
import { fetchAnnotationHistory, fetchAnnotationRevision } from '../services/annotationHistoryApi'
import { submitAnnotations } from '../services/annotationApi'
import { fetchTasks } from '../services/taskApi'

const emit = defineEmits(['task-changed', 'annotations-saved'])
const props = defineProps({
  revisionOverlay: {
    type: Object,
    default: null,
  },
})

const stageWidth = ref(1024)
const stageHeight = ref(768)
const stageX = ref(0)
const stageY = ref(0)
const stageScale = ref(1)
const toolMode = ref('move')
const middlePanActive = ref(false)
const annotations = ref([])
const annotationDisplayCounter = ref(1)
const draftRect = ref(null)
const drawStart = ref(null)
const currentPolygonPoints = ref([])
const taskList = ref([])
const currentTaskIndex = ref(-1)
const isSubmitting = ref(false)
const saveMessage = ref('')
const selectedAnnotationId = ref('')
const selectedLabel = ref('Unlabeled')
const taskCatalogVisible = ref(false)
const taskCatalogQuery = ref('')
const undoDeleteVisible = ref(false)
const undoDeleteMessage = ref('')
const lastRemovedAnnotation = ref(null)
const lastRemovedIndex = ref(-1)
const pendingDeleteAnnotationId = ref('')
let pendingDeleteTimerId = null
const fitsCanvasRef = ref(null)
const canvasHostRef = ref(null)
const stageRef = ref(null)
const hotkeysTeleportTarget = ref(null)
const inspectorTeleportTarget = ref(null)
const stretchRangeMin = ref(0)
const stretchRangeMax = ref(1)
const stretchMin = ref(0)
const stretchMax = ref(1)
const autoStretchEnabled = ref(true)
const taskAutoStretchById = ref({})
const invertDisplay = ref(false)
const bboxStrokeWidth = ref(2)
let hostResizeObserver = null

const taskProgressText = computed(() => {
  if (taskList.value.length === 0 || currentTaskIndex.value < 0) {
    return '0 / 0'
  }
  return `${currentTaskIndex.value + 1} / ${taskList.value.length}`
})

const filteredTaskCatalog = computed(() => {
  const keyword = String(taskCatalogQuery.value || '').trim().toLowerCase()
  const mapped = taskList.value.map((task, index) => ({ task, index }))
  if (!keyword) {
    return mapped
  }
  return mapped.filter((item) => String(item.task.task_id || '').toLowerCase().includes(keyword))
})

const revisionOverlayItems = computed(() => props.revisionOverlay?.changed_items || [])

function toOverlayRect(annotation) {
  if (!annotation) {
    return null
  }
  const x = Number(annotation.x)
  const y = Number(annotation.y)
  const width = Number(annotation.width)
  const height = Number(annotation.height)
  if (![x, y, width, height].every(Number.isFinite)) {
    return null
  }
  return { x, y, width, height }
}

const revisionOverlayAddedRects = computed(() => revisionOverlayItems.value
  .filter((item) => item?.change_type === 'added')
  .map((item) => toOverlayRect(item.after))
  .filter(Boolean))

const revisionOverlayRemovedRects = computed(() => revisionOverlayItems.value
  .filter((item) => item?.change_type === 'removed')
  .map((item) => toOverlayRect(item.before))
  .filter(Boolean))

const revisionOverlayModifiedBeforeRects = computed(() => revisionOverlayItems.value
  .filter((item) => item?.change_type === 'modified')
  .map((item) => toOverlayRect(item.before))
  .filter(Boolean))

const revisionOverlayModifiedAfterRects = computed(() => revisionOverlayItems.value
  .filter((item) => item?.change_type === 'modified')
  .map((item) => toOverlayRect(item.after))
  .filter(Boolean))

const hasPrevTask = computed(() => currentTaskIndex.value > 0)
const hasNextTask = computed(() => (
  currentTaskIndex.value >= 0 && currentTaskIndex.value < taskList.value.length - 1
))

const stageConfig = computed(() => ({
  width: stageWidth.value,
  height: stageHeight.value,
  draggable: toolMode.value === 'move' && !middlePanActive.value,
  x: stageX.value,
  y: stageY.value,
  scaleX: stageScale.value,
  scaleY: stageScale.value,
}))

const fitsCanvasStyle = computed(() => ({
  width: `${stageWidth.value}px`,
  height: `${stageHeight.value}px`,
  transform: `translate(${stageX.value}px, ${stageY.value}px) scale(${stageScale.value})`,
  transformOrigin: 'top left',
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

function onDragMove(event) {
  const position = event?.target?.position?.()
  if (!position) {
    return
  }

  stageX.value = position.x
  stageY.value = position.y
}

function zoomAtPointer(deltaY, pointerX, pointerY) {
  if (typeof deltaY !== 'number' || typeof pointerX !== 'number' || typeof pointerY !== 'number') {
    return
  }

  const oldScale = stageScale.value
  const direction = deltaY > 0 ? -1 : 1
  const nextScaleRaw = oldScale * (direction > 0 ? 1.05 : 0.95)
  const nextScale = Math.max(0.1, Math.min(10, nextScaleRaw))
  if (nextScale === oldScale) {
    return
  }

  const worldX = (pointerX - stageX.value) / oldScale
  const worldY = (pointerY - stageY.value) / oldScale

  stageScale.value = nextScale
  stageX.value = pointerX - worldX * nextScale
  stageY.value = pointerY - worldY * nextScale
}

function onWheel(event) {
  const rawEvent = event?.evt
  const deltaY = rawEvent?.deltaY
  const stage = event?.target?.getStage?.()
  const pointer = stage?.getPointerPosition?.()
  if (typeof deltaY !== 'number' || !pointer) {
    return
  }

  rawEvent.preventDefault?.()
  rawEvent.stopPropagation?.()
  zoomAtPointer(deltaY, pointer.x, pointer.y)
}

function switchToView(view) {
  if (!['new', 'new_marked', 'old'].includes(view)) {
    return
  }
  setCurrentView(view)
}

function resetAnnotationStates() {
  annotations.value = []
  annotationDisplayCounter.value = 1
  selectedAnnotationId.value = ''
  selectedLabel.value = 'Unlabeled'
  clearPendingDeleteState()
  clearUndoState()
  draftRect.value = null
  drawStart.value = null
  currentPolygonPoints.value = []
}

function clearUndoState() {
  undoDeleteVisible.value = false
  undoDeleteMessage.value = ''
  lastRemovedAnnotation.value = null
  lastRemovedIndex.value = -1
}

function clearPendingDeleteState() {
  pendingDeleteAnnotationId.value = ''
  if (pendingDeleteTimerId) {
    clearTimeout(pendingDeleteTimerId)
    pendingDeleteTimerId = null
  }
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
  await loadLatestRevisionAnnotations(task.task_id)
  emit('task-changed', task.task_id)
  return true
}

async function loadLatestRevisionAnnotations(taskId) {
  if (!taskId) {
    return
  }

  try {
    const history = await fetchAnnotationHistory(taskId)
    const latest = history?.revisions?.[0]
    if (!latest?.revision_id) {
      return
    }

    const detail = await fetchAnnotationRevision(taskId, latest.revision_id)
    const revisionAnnotations = Array.isArray(detail?.annotations) ? detail.annotations : []
    const restored = revisionAnnotations.map((ann, index) => ({
      id: `hist-${taskId}-${index}-${Date.now()}`,
      display_id: `A${String(index + 1).padStart(4, '0')}`,
      type: 'bbox',
      x: Number(ann.x) || 0,
      y: Number(ann.y) || 0,
      width: Number(ann.width) || 0,
      height: Number(ann.height) || 0,
      label: ann.label || 'Unlabeled',
      detail_type: ann.detail_type,
    }))

    annotations.value = restored
    annotationDisplayCounter.value = restored.length + 1

    if (detail?.source_view && ['new', 'new_marked', 'old'].includes(detail.source_view)) {
      setCurrentView(detail.source_view)
    }
  } catch {
    // 历史读取失败时保持空状态，避免阻塞任务切换。
  }
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

function openTaskCatalog() {
  taskCatalogQuery.value = ''
  taskCatalogVisible.value = true
}

function closeTaskCatalog() {
  taskCatalogVisible.value = false
}

async function jumpToTaskIndex(index) {
  const loaded = await loadTaskAtIndex(index)
  if (loaded) {
    closeTaskCatalog()
  }
}

function onContainerWheel(event) {
  const deltaY = event?.deltaY
  const host = canvasHostRef.value
  if (typeof deltaY !== 'number' || !host) {
    return
  }

  const rect = host.getBoundingClientRect()
  const pointerX = event.clientX - rect.left
  const pointerY = event.clientY - rect.top
  zoomAtPointer(deltaY, pointerX, pointerY)
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

function onAutoStretchToggle(event) {
  autoStretchEnabled.value = Boolean(event?.target?.checked)
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function quantile(sortedValues, ratio) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
    return 0
  }
  const index = clamp(Math.floor((sortedValues.length - 1) * ratio), 0, sortedValues.length - 1)
  return sortedValues[index]
}

function buildHistogramAutoStretch(pixels) {
  const range = calculatePixelRange(pixels)
  const allFinite = []
  const maxSamples = 50000
  const step = Math.max(1, Math.floor(pixels.length / maxSamples))
  for (let i = 0; i < pixels.length; i += step) {
    const value = Number(pixels[i])
    if (Number.isFinite(value)) {
      allFinite.push(value)
    }
  }

  if (allFinite.length < 8) {
    return {
      rangeMin: range.min,
      rangeMax: range.max,
      stretchMin: range.min,
      stretchMax: range.max,
    }
  }

  allFinite.sort((a, b) => a - b)
  const low = quantile(allFinite, 0.003)
  const high = quantile(allFinite, 0.997)

  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) {
    return {
      rangeMin: range.min,
      rangeMax: range.max,
      stretchMin: range.min,
      stretchMax: range.max,
    }
  }

  return {
    rangeMin: range.min,
    rangeMax: range.max,
    stretchMin: clamp(low, range.min, range.max),
    stretchMax: clamp(high, range.min, range.max),
  }
}

function applyStretchForNode(node) {
  if (!node?.pixels || node.pixels.length === 0) {
    return
  }

  if (!autoStretchEnabled.value) {
    const range = calculatePixelRange(node.pixels)
    stretchRangeMin.value = range.min
    stretchRangeMax.value = range.max
    stretchMin.value = range.min
    stretchMax.value = range.max
    syncStageSizeToHost()
    return
  }

  const taskId = String(activeTask.value?.task_id || '')
  if (!taskId) {
    const range = calculatePixelRange(node.pixels)
    stretchRangeMin.value = range.min
    stretchRangeMax.value = range.max
    stretchMin.value = range.min
    stretchMax.value = range.max
    syncStageSizeToHost()
    return
  }

  let preset = taskAutoStretchById.value[taskId]
  if (!preset) {
    preset = buildHistogramAutoStretch(node.pixels)
    taskAutoStretchById.value = {
      ...taskAutoStretchById.value,
      [taskId]: preset,
    }
  }

  stretchRangeMin.value = preset.rangeMin
  stretchRangeMax.value = preset.rangeMax
  stretchMin.value = clamp(preset.stretchMin, preset.rangeMin, preset.rangeMax)
  stretchMax.value = clamp(preset.stretchMax, preset.rangeMin, preset.rangeMax)
  syncStageSizeToHost()
}

function onInvertChange(event) {
  invertDisplay.value = Boolean(event?.target?.checked)
}

function onBboxStrokeWidthInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  bboxStrokeWidth.value = Math.max(1, Math.min(8, value))
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

  const node = activeFitsNode.value
  if (node?.width && node?.height) {
    stageWidth.value = Math.max(1, Math.floor(node.width))
    stageHeight.value = Math.max(1, Math.floor(node.height))
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
    display_id: `A${String(annotationDisplayCounter.value).padStart(4, '0')}`,
    label: initialLabel,
    detail_type: initialDetail,
    ...base,
  }
}

function addAnnotation(annotation) {
  annotations.value.push(annotation)
  annotationDisplayCounter.value += 1
  selectAnnotation(annotation.id)
}

function getAnnotationLabelKey(annotation) {
  if (!annotation || annotation.label === 'Unlabeled' || !annotation.label) {
    return 'Unlabeled'
  }
  if (annotation.detail_type) {
    return `${annotation.label}:${annotation.detail_type}`
  }
  return annotation.label
}

function getAnnotationColor(annotation) {
  const key = getAnnotationLabelKey(annotation)
  const colorMap = {
    Unlabeled: '#94a3b8',
    'real:asteroid': '#22c55e',
    'real:supernova': '#16a34a',
    'real:variable_star': '#4ade80',
    'bogus:satellite_trail': '#f43f5e',
    'bogus:noise': '#ef4444',
    'bogus:diffraction_spike': '#e11d48',
    'bogus:cmos_condensation': '#fb7185',
    'bogus:corresponding': '#be123c',
    real: '#22c55e',
    bogus: '#ef4444',
  }
  return colorMap[key] || '#94a3b8'
}

function hexToRgba(hexColor, alpha) {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hexColor)
  if (!match) {
    return `rgba(148, 163, 184, ${alpha})`
  }
  const r = parseInt(match[1], 16)
  const g = parseInt(match[2], 16)
  const b = parseInt(match[3], 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function getAnnotationItemStyle(annotation, isSelected) {
  const color = getAnnotationColor(annotation)
  return {
    borderColor: isSelected ? '#0ea5e9' : color,
    color: isSelected ? '#bae6fd' : color,
    backgroundColor: hexToRgba(color, isSelected ? 0.18 : 0.08),
  }
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

function removeAnnotation(annotationId) {
  const target = annotations.value.find((item) => item.id === annotationId)
  if (!target) {
    return
  }

  if (pendingDeleteAnnotationId.value !== annotationId) {
    clearPendingDeleteState()
    pendingDeleteAnnotationId.value = annotationId
    undoDeleteMessage.value = `请在2秒内再次点击“删除”以确认删除 ${target.display_id}`
    pendingDeleteTimerId = setTimeout(() => {
      clearPendingDeleteState()
      if (!undoDeleteVisible.value) {
        undoDeleteMessage.value = ''
      }
    }, 2000)
    return
  }

  clearPendingDeleteState()

  const index = annotations.value.findIndex((item) => item.id === annotationId)
  if (index < 0) {
    return
  }

  clearUndoState()
  lastRemovedAnnotation.value = target
  lastRemovedIndex.value = index

  annotations.value = annotations.value.filter((item) => item.id !== annotationId)
  if (selectedAnnotationId.value === annotationId) {
    selectedAnnotationId.value = ''
    selectedLabel.value = 'Unlabeled'
  }

  undoDeleteVisible.value = true
  undoDeleteMessage.value = `已删除 ${target.display_id}`
}

function undoRemoveAnnotation() {
  if (!lastRemovedAnnotation.value) {
    clearUndoState()
    return
  }
  const insertIndex = Math.max(0, Math.min(lastRemovedIndex.value, annotations.value.length))
  annotations.value.splice(insertIndex, 0, lastRemovedAnnotation.value)
  clearUndoState()
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

let middlePanLast = null

function onMiddlePanMove(event) {
  if (!middlePanActive.value || !middlePanLast) {
    return
  }

  const dx = event.clientX - middlePanLast.x
  const dy = event.clientY - middlePanLast.y
  stageX.value += dx
  stageY.value += dy
  middlePanLast = { x: event.clientX, y: event.clientY }
}

function stopMiddlePan() {
  if (!middlePanActive.value) {
    return
  }
  middlePanActive.value = false
  middlePanLast = null
  window.removeEventListener('mousemove', onMiddlePanMove)
  window.removeEventListener('mouseup', stopMiddlePan)
}

function startMiddlePan(rawEvent) {
  if (typeof rawEvent?.clientX !== 'number' || typeof rawEvent?.clientY !== 'number') {
    return
  }

  middlePanActive.value = true
  middlePanLast = { x: rawEvent.clientX, y: rawEvent.clientY }
  window.addEventListener('mousemove', onMiddlePanMove)
  window.addEventListener('mouseup', stopMiddlePan)
}

function onStageMouseDown(event) {
  const raw = event?.evt ?? event
  if (raw?.button === 1) {
    raw.preventDefault?.()
    startMiddlePan(raw)
    return
  }

  if (middlePanActive.value) {
    return
  }

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
  if (middlePanActive.value) {
    return
  }

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
  const raw = event?.evt ?? event
  if (raw?.button === 1 || middlePanActive.value) {
    stopMiddlePan()
    return
  }

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
  applyStretchForNode(node)
})

watch(autoStretchEnabled, () => {
  applyStretchForNode(activeFitsNode.value)
})

watch(
  [stretchedRgba, activeFitsNode],
  () => {
    redrawFitsCanvas()
  },
  { immediate: true },
)

function onKeyDown(event) {
  if (event.key === 'Escape' && taskCatalogVisible.value) {
    event.preventDefault()
    closeTaskCatalog()
    return
  }

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

function resolveTeleportTargets() {
  if (typeof document === 'undefined') {
    return
  }
  const fallbackTarget = canvasHostRef.value?.parentElement ?? null
  hotkeysTeleportTarget.value = document.getElementById('hotkeys-extra') || fallbackTarget
  inspectorTeleportTarget.value = document.getElementById('inspector-extra') || fallbackTarget
}

onMounted(async () => {
  await nextTick()
  resolveTeleportTargets()
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
  stopMiddlePan()
  clearPendingDeleteState()
  clearUndoState()
  if (hostResizeObserver) {
    hostResizeObserver.disconnect()
    hostResizeObserver = null
  }
  window.removeEventListener('keydown', onKeyDown)
  releaseObjectUrls()
})
</script>
