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
              <p
                v-if="taskLockNotice"
                data-testid="task-lock-status"
                class="text-[10px] text-emerald-300"
              >
                {{ taskLockNotice }}
              </p>
              <p
                v-else-if="taskLockError"
                data-testid="task-lock-error"
                class="text-[10px] text-rose-300"
              >
                {{ taskLockError }}
              </p>
              <p
                v-if="taskSwitchError"
                data-testid="task-switch-error"
                class="text-[10px] text-amber-300"
              >
                {{ taskSwitchError }}
              </p>
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
              <div class="grid grid-cols-3 gap-2">
                <label
                  class="text-[10px] text-slate-400 inline-flex items-center justify-center gap-1 rounded border border-slate-800 px-2 py-1"
                >
                  <input
                    data-testid="blink-queue-new"
                    type="checkbox"
                    :checked="blinkViewSelection.new"
                    @change="onBlinkViewToggle('new', $event)"
                  >
                  闪烁
                </label>
                <label
                  class="text-[10px] text-slate-400 inline-flex items-center justify-center gap-1 rounded border border-slate-800 px-2 py-1"
                >
                  <input
                    data-testid="blink-queue-new-marked"
                    type="checkbox"
                    :checked="blinkViewSelection.new_marked"
                    @change="onBlinkViewToggle('new_marked', $event)"
                  >
                  闪烁
                </label>
                <label
                  class="text-[10px] text-slate-400 inline-flex items-center justify-center gap-1 rounded border border-slate-800 px-2 py-1"
                >
                  <input
                    data-testid="blink-queue-old"
                    type="checkbox"
                    :checked="blinkViewSelection.old"
                    @change="onBlinkViewToggle('old', $event)"
                  >
                  闪烁
                </label>
              </div>
              <p class="text-[10px] text-slate-400">快捷键：Space 切换视图 / Tab 开关闪烁</p>
            </div>

            <div class="space-y-2">
              <p class="text-xs text-slate-200">闪烁 (Blink)</p>
              <button
                data-testid="blink-toggle"
                class="w-full text-xs px-2 py-1.5 rounded border"
                :class="blinkEnabled ? 'border-amber-500 text-amber-300 bg-amber-950/30' : 'border-slate-700 text-slate-300'"
                :disabled="blinkQueueViews.length === 0"
                @click="onBlinkToggle"
              >
                {{ blinkEnabled ? '停止闪烁' : '开始闪烁' }}
              </button>
              <p class="text-[10px] text-slate-500" data-testid="blink-queue-text">队列：{{ blinkQueueText }}</p>
              <div class="space-y-1">
                <label class="text-[10px] text-slate-400">间隔: {{ (blinkInterval / 1000).toFixed(1) }}s</label>
                <input
                  data-testid="blink-interval-slider"
                  type="range"
                  class="w-full"
                  min="100"
                  max="3000"
                  step="100"
                  :value="blinkInterval"
                  @input="onBlinkIntervalInput"
                >
              </div>
            </div>

            <div class="space-y-2">
              <p class="text-xs text-slate-200">工具</p>
              <div class="grid grid-cols-2 gap-2">
                <button
                  data-testid="tool-move"
                  class="text-xs px-2 py-1 rounded border"
                  :class="toolMode === 'move' ? 'border-sky-400 text-sky-300' : 'border-slate-700 text-slate-300'"
                  @click="setToolMode('move')"
                >
                  移动
                </button>
                <button
                  data-testid="tool-bbox"
                  class="text-xs px-2 py-1 rounded border"
                  :class="toolMode === 'bbox' ? 'border-emerald-400 text-emerald-300' : 'border-slate-700 text-slate-300'"
                  @click="setToolMode('bbox')"
                >
                  矩形
                </button>
                <button
                  data-testid="tool-point"
                  class="text-xs px-2 py-1 rounded border"
                  :class="toolMode === 'point' ? 'border-amber-400 text-amber-300' : 'border-slate-700 text-slate-300'"
                  @click="setToolMode('point')"
                >
                  点
                </button>
                <button
                  data-testid="tool-polygon"
                  class="text-xs px-2 py-1 rounded border"
                  :class="toolMode === 'polygon' ? 'border-violet-400 text-violet-300' : 'border-slate-700 text-slate-300'"
                  @click="setToolMode('polygon')"
                >
                  多边形
                </button>
                <button
                  data-testid="tool-crop"
                  class="text-xs px-2 py-1 rounded border"
                  :class="toolMode === 'crop' ? 'border-cyan-400 text-cyan-300' : 'border-slate-700 text-slate-300'"
                  @click="setToolMode('crop')"
                >
                  手动裁剪
                </button>
              </div>
              <button
                v-if="toolMode === 'polygon'"
                data-testid="finish-polygon"
                class="w-full text-xs px-2 py-1 rounded border border-violet-700 text-violet-200"
                @click="finishPolygon"
              >
                完成多边形
              </button>
              <div class="space-y-1 rounded border border-cyan-900/60 bg-cyan-950/20 p-2">
                <div class="flex items-center justify-between gap-2">
                  <p class="text-[11px] text-cyan-200">手动裁剪区域</p>
                  <button
                    data-testid="manual-crop-clear"
                    class="text-[10px] px-2 py-0.5 rounded border border-cyan-800 text-cyan-200 disabled:opacity-50"
                    :disabled="!manualCropRect"
                    @click="clearManualCrop"
                  >
                    清除
                  </button>
                </div>
                <p class="text-[10px] text-cyan-300">
                  {{ manualCropRect ? `x=${Math.round(manualCropRect.x)}, y=${Math.round(manualCropRect.y)}, w=${Math.round(manualCropRect.width)}, h=${Math.round(manualCropRect.height)}` : '未设置' }}
                </p>
                <p class="text-[10px] text-slate-400">使用“手动裁剪”工具在图上框选有效区域，裁剪区外将不参与提交与训练。</p>
              </div>
              <p class="text-[10px] text-slate-500">快捷键：H 切换移动工具 / C 切换矩形工具</p>
            </div>

            <div class="space-y-2">
              <div class="flex items-center justify-between gap-2">
                <p class="text-[11px] text-slate-200">拉伸</p>
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
              <button
                data-testid="match-group-stretch"
                class="w-full text-xs px-2 py-1 rounded border border-sky-700 text-sky-200"
                :disabled="!activeTask || !activeFitsNode"
                @click="onMatchGroupStretch"
              >
                匹配另外两图 (M)
              </button>
              <p class="text-[10px] text-slate-500">快捷键：M 将当前图亮度同步到同任务组另外两图</p>
              <label class="text-[11px] text-slate-300 inline-flex items-center gap-2">
                <input
                  data-testid="invert-toggle"
                  type="checkbox"
                  :checked="invertDisplay"
                  @change="onInvertChange"
                >
                反转
              </label>
            </div>

            <div class="space-y-2">
              <p class="text-[11px] text-slate-200">标注样式统一设置</p>
              <div class="space-y-1">
                <label class="text-[10px] text-slate-400">BBox 线宽: {{ bboxStrokeWidth.toFixed(1) }}</label>
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
              <div class="space-y-1">
                <label class="text-[10px] text-slate-400">点半径: {{ pointRadius.toFixed(1) }}</label>
                <input
                  data-testid="point-radius-slider"
                  type="range"
                  class="w-full"
                  min="2"
                  max="12"
                  step="0.5"
                  :value="pointRadius"
                  @input="onPointRadiusInput"
                >
              </div>
              <div class="space-y-1">
                <label class="text-[10px] text-slate-400">多边形线宽: {{ polygonStrokeWidth.toFixed(1) }}</label>
                <input
                  data-testid="polygon-stroke-width-slider"
                  type="range"
                  class="w-full"
                  min="1"
                  max="8"
                  step="0.5"
                  :value="polygonStrokeWidth"
                  @input="onPolygonStrokeWidthInput"
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
                <span class="flex items-center justify-between gap-2">
                  <span class="truncate">{{ item.task.task_id }}</span>
                  <span
                    v-if="item.task.lock_expires_at"
                    data-testid="task-catalog-lock-status"
                    class="shrink-0 rounded border px-1.5 py-0.5 text-[10px]"
                    :class="item.task.locked_by_current_client ? 'border-emerald-700 text-emerald-300 bg-emerald-950/30' : 'border-amber-700 text-amber-300 bg-amber-950/30'"
                  >
                    {{ item.task.locked_by_current_client ? '当前会话占用' : '占用中' }}
                  </span>
                </span>
                <span
                  v-if="item.task.lock_expires_at"
                  class="mt-1 block text-[10px] text-slate-500"
                >
                  占用至 {{ formatTaskLockExpiry(item.task.lock_expires_at) }}
                </span>
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
            <template v-if="manualCropMasks.length > 0">
              <v-rect
                v-for="(mask, index) in manualCropMasks"
                :key="`manual-crop-mask-${index}`"
                :config="{
                  x: mask.x,
                  y: mask.y,
                  width: mask.width,
                  height: mask.height,
                  fill: 'rgba(2, 6, 23, 0.68)',
                  listening: false,
                }"
              />
              <v-rect
                :config="{
                  x: manualCropRect.x,
                  y: manualCropRect.y,
                  width: manualCropRect.width,
                  height: manualCropRect.height,
                  stroke: '#22d3ee',
                  strokeWidth: Math.max(1.5, bboxStrokeWidth),
                  dash: [8, 4],
                  listening: false,
                }"
              />
            </template>
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
                radius: pointRadius,
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
                strokeWidth: polygonStrokeWidth,
              }"
            />
            <v-line
              v-if="toolMode === 'polygon' && currentPolygonPoints.length > 0"
              :config="{
                points: toFlatPoints(currentPolygonPoints),
                closed: false,
                stroke: '#60a5fa',
                strokeWidth: polygonStrokeWidth,
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
          Loading FITS images...
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
              :disabled="isSubmitting || !activeTask || !hasTaskLock"
              @click="submitCurrentAnnotations"
            >
              {{ isSubmitting ? 'Submitting...' : 'Submit' }}
            </button>
            <div class="space-y-2 rounded border border-slate-800 bg-slate-950/60 p-2">
              <div class="flex items-center justify-between gap-2">
                <label class="inline-flex items-center gap-2 text-[11px] text-slate-200">
                  <input
                    data-testid="auto-submit-toggle"
                    type="checkbox"
                    :checked="autoSubmitEnabled"
                    @change="onAutoSubmitToggle"
                  >
                  自动提交
                </label>
                <span data-testid="auto-submit-countdown" class="text-[10px] text-slate-500">
                  {{ autoSubmitCountdownText }}
                </span>
              </div>
              <label class="space-y-1 block">
                <span class="text-[10px] text-slate-400">间隔（秒）</span>
                <input
                  data-testid="auto-submit-interval"
                  type="number"
                  class="w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
                  min="30"
                  max="3600"
                  step="30"
                  :value="autoSubmitIntervalSeconds"
                  @input="onAutoSubmitIntervalInput"
                >
              </label>
              <p class="text-[10px] text-slate-500">自动提交只保存当前任务，不自动切换任务。</p>
              <p
                v-if="autoSubmitStatusMessage"
                data-testid="auto-submit-status"
                class="text-[10px] px-2 py-1 rounded"
                :class="autoSubmitStatusClass"
              >
                {{ autoSubmitStatusMessage }}
              </p>
            </div>
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
                  <option value="bogus:disappeared_asteroid">Disappeared Asteroid (消失小行星)</option>
                  <option value="bogus:disappeared_star">Disappeared Star (消失恒星)</option>
                  <option value="bogus:disappeared_galaxy">Disappeared Galaxy (消失星系)</option>
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

import { BLINK_VIEW_ORDER, useBlinkControl } from '../composables/useBlinkControl'
import { useFitsImagePool } from '../composables/useFitsImagePool'
import { renderStretchToRgba } from '../fits/stretchRenderer'
import {
  buildBrightnessMatchViewStatesByView,
  buildFullRangeViewStatesByView,
  buildViewStretchState,
  DEFAULT_BRIGHTNESS_MATCH_OPTIONS,
  matchViewStatesFromSourceState,
} from '../fits/brightnessMatch'
import { fetchAnnotationHistory, fetchAnnotationRevision } from '../services/annotationHistoryApi'
import { submitAnnotations } from '../services/annotationApi'
import {
  claimNextTask,
  claimTask,
  fetchTasks,
  getTaskClientId,
  heartbeatTask,
  releaseTask,
} from '../services/taskApi'

const emit = defineEmits(['task-changed', 'annotations-saved'])
const props = defineProps({
  revisionOverlay: {
    type: Object,
    default: null,
  },
  taskRefreshKey: {
    type: Number,
    default: 0,
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
const manualCropRect = ref(null)
const taskList = ref([])
const currentTaskIndex = ref(-1)
const isSubmitting = ref(false)
const saveMessage = ref('')
const autoSubmitEnabled = ref(false)
const autoSubmitIntervalSeconds = ref(300)
const autoSubmitStatusMessage = ref('')
const autoSubmitInFlight = ref(false)
const autoSubmitNowMs = ref(Date.now())
const autoSubmitNextRunAtMs = ref(0)
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
const activeTask = ref(null)
const currentView = ref('new')
const error = ref('')
const claimedTaskId = ref('')
const taskLockExpiresAt = ref('')
const taskLockError = ref('')
const taskLockNotice = ref('')
const taskSwitchError = ref('')
const stretchRangeMin = ref(0)
const stretchRangeMax = ref(1)
const stretchMin = ref(0)
const stretchMax = ref(1)
const autoStretchEnabled = ref(true)
const taskAutoStretchById = ref({})
const invertDisplay = ref(false)
const bboxStrokeWidth = ref(2)
const pointRadius = ref(4)
const polygonStrokeWidth = ref(2)
let hostResizeObserver = null
let taskHeartbeatTimerId = null
let autoSubmitTimerId = null
const taskClientId = getTaskClientId()
const AUTO_SUBMIT_MIN_SECONDS = 30
const AUTO_SUBMIT_MAX_SECONDS = 3600

const taskProgressText = computed(() => {
  if (taskList.value.length === 0 || currentTaskIndex.value < 0) {
    return '0 / 0'
  }
  return `${currentTaskIndex.value + 1} / ${taskList.value.length}`
})

const blinkViewLabels = {
  new: '新图',
  new_marked: '新图标记',
  old: '旧图',
}
const blinkViewSelection = ref({
  new: true,
  new_marked: true,
  old: true,
})
const blinkQueueViews = computed(() => BLINK_VIEW_ORDER.filter((view) => blinkViewSelection.value[view]))
const blinkQueueText = computed(() => (
  blinkQueueViews.value.length > 0
    ? blinkQueueViews.value.map((view) => blinkViewLabels[view] || view).join(' -> ')
    : '未选择视图'
))

function formatDurationFromMs(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  return [hours, minutes, seconds]
    .map((value) => String(value).padStart(2, '0'))
    .join(':')
}

const autoSubmitCountdownText = computed(() => {
  if (!autoSubmitEnabled.value) {
    return '已关闭'
  }
  if (!activeTask.value) {
    return '等待任务'
  }
  if (autoSubmitInFlight.value) {
    return '提交中...'
  }
  if (!autoSubmitNextRunAtMs.value) {
    return '等待启动'
  }
  return `下次 ${formatDurationFromMs(autoSubmitNextRunAtMs.value - autoSubmitNowMs.value)}`
})

const autoSubmitStatusClass = computed(() => {
  const message = autoSubmitStatusMessage.value
  if (!message) {
    return 'bg-slate-900 text-slate-300'
  }
  if (message.includes('失败') || message.includes('未被本会话持有')) {
    return 'bg-rose-950/30 text-rose-300'
  }
  if (message.includes('跳过')) {
    return 'bg-amber-950/30 text-amber-200'
  }
  return 'bg-slate-900 text-emerald-300'
})

const filteredTaskCatalog = computed(() => {
  const keyword = String(taskCatalogQuery.value || '').trim().toLowerCase()
  const mapped = taskList.value.map((task, index) => ({ task, index }))
  if (!keyword) {
    return mapped
  }
  return mapped.filter((item) => String(item.task.task_id || '').toLowerCase().includes(keyword))
})

function formatTaskLockExpiry(value) {
  if (!value) {
    return '--'
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return '--'
  }
  return parsed.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

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
const hasTaskLock = computed(() => {
  const taskId = String(activeTask.value?.task_id || '')
  return Boolean(taskId) && claimedTaskId.value === taskId && !taskLockError.value
})

const activeImageRect = computed(() => {
  const width = Number(activeFitsNode.value?.width || stageWidth.value || 0)
  const height = Number(activeFitsNode.value?.height || stageHeight.value || 0)
  return {
    x: 0,
    y: 0,
    width: Math.max(0, width),
    height: Math.max(0, height),
  }
})

const manualCropMasks = computed(() => {
  const crop = manualCropRect.value
  const imageRect = activeImageRect.value
  if (!crop || imageRect.width <= 0 || imageRect.height <= 0) {
    return []
  }

  const x0 = crop.x
  const y0 = crop.y
  const x1 = crop.x + crop.width
  const y1 = crop.y + crop.height
  const maxX = imageRect.width
  const maxY = imageRect.height

  const masks = [
    { x: 0, y: 0, width: maxX, height: Math.max(0, y0) },
    { x: 0, y: Math.max(0, y0), width: Math.max(0, x0), height: Math.max(0, y1 - y0) },
    { x: Math.min(maxX, x1), y: Math.max(0, y0), width: Math.max(0, maxX - x1), height: Math.max(0, y1 - y0) },
    { x: 0, y: Math.min(maxY, y1), width: maxX, height: Math.max(0, maxY - y1) },
  ]
  return masks.filter((item) => item.width > 0 && item.height > 0)
})

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
  fitsError,
  fitsNodes,
  isFitsLoading,
  preloadTaskFits,
} = useFitsImagePool()

const isLoading = computed(() => isFitsLoading.value)

const imageNodes = computed(() => (
  fitsNodes.value.map((node) => ({
    view: node.view,
    visible: node.view === currentView.value,
  }))
))

function clearTaskLockMessages() {
  taskLockError.value = ''
  taskLockNotice.value = ''
}

function clearTaskSwitchError() {
  taskSwitchError.value = ''
}

function isTaskLockedByOtherClient(task) {
  return Boolean(task?.lock_expires_at) && task?.locked_by_current_client === false
}

function setTaskSwitchBlockedMessage(taskId) {
  taskSwitchError.value = `任务组 ${taskId} 当前被其他用户占用，无法切换`
}

function setTaskSwitchFailureMessage(taskId, message) {
  taskSwitchError.value = `切换到任务组 ${taskId} 失败：${message}`
}

function stopTaskHeartbeat() {
  if (taskHeartbeatTimerId) {
    window.clearInterval(taskHeartbeatTimerId)
    taskHeartbeatTimerId = null
  }
}

function startTaskHeartbeat(taskId) {
  stopTaskHeartbeat()
  if (!taskId) {
    return
  }

  taskHeartbeatTimerId = window.setInterval(async () => {
    try {
      const refreshed = await heartbeatTask(taskId, taskClientId)
      taskLockExpiresAt.value = String(refreshed.lock_expires_at || '')
      taskLockNotice.value = 'Current task is locked by this client'
      taskLockError.value = ''
    } catch (err) {
      claimedTaskId.value = ''
      taskLockExpiresAt.value = ''
      taskLockError.value = err instanceof Error ? err.message : 'Task lock expired'
      taskLockNotice.value = ''
      stopTaskHeartbeat()
    }
  }, 60_000)
}

async function releaseActiveTask(taskId = activeTask.value?.task_id) {
  const normalizedTaskId = String(taskId || '')
  if (!normalizedTaskId) {
    claimedTaskId.value = ''
    taskLockExpiresAt.value = ''
    stopTaskHeartbeat()
    return
  }

  const releasingCurrentTask = claimedTaskId.value === normalizedTaskId
  if (releasingCurrentTask) {
    stopTaskHeartbeat()
  }
  try {
    await releaseTask(normalizedTaskId, taskClientId)
  } catch {
    // Ignore release failures; lock expiry still provides fallback protection.
  } finally {
    if (releasingCurrentTask && claimedTaskId.value === normalizedTaskId) {
      claimedTaskId.value = ''
      taskLockExpiresAt.value = ''
    }
  }
}

function setError(message) {
  error.value = message
}

function setCurrentView(view) {
  currentView.value = view
}

function getNodesByView() {
  return fitsNodes.value.reduce((accumulator, node) => {
    accumulator[node.view] = node
    return accumulator
  }, {})
}

function buildPreset(viewStates) {
  return {
    viewStates: { ...viewStates },
  }
}

function saveTaskStretchPreset(taskId, preset) {
  if (!taskId) {
    return
  }
  taskAutoStretchById.value = {
    ...taskAutoStretchById.value,
    [taskId]: preset,
  }
}

function ensureTaskStretchPreset(taskId) {
  if (!taskId) {
    return null
  }

  let preset = taskAutoStretchById.value[taskId]
  if (preset) {
    return preset
  }

  const nodesByView = getNodesByView()
  preset = buildPreset(
    autoStretchEnabled.value
      ? buildBrightnessMatchViewStatesByView(nodesByView, DEFAULT_BRIGHTNESS_MATCH_OPTIONS)
      : buildFullRangeViewStatesByView(nodesByView),
  )
  saveTaskStretchPreset(taskId, preset)
  return preset
}

function syncStretchControlsFromState(state) {
  if (!state) {
    stretchRangeMin.value = 0
    stretchRangeMax.value = 1
    stretchMin.value = 0
    stretchMax.value = 1
    return
  }

  stretchRangeMin.value = Number(state.rangeMin)
  stretchRangeMax.value = Number(state.rangeMax)
  stretchMin.value = Number(state.stretchMin)
  stretchMax.value = Number(state.stretchMax)
}

function syncStretchControlsForActiveView() {
  const taskId = String(activeTask.value?.task_id || '')
  const node = activeFitsNode.value
  if (!taskId || !node?.pixels) {
    syncStretchControlsFromState(buildViewStretchState(node, undefined, undefined))
    return
  }

  const preset = ensureTaskStretchPreset(taskId)
  const existingState = preset?.viewStates?.[currentView.value]
  const state = existingState || buildViewStretchState(node, undefined, undefined)

  if (!existingState && preset) {
    saveTaskStretchPreset(taskId, buildPreset({
      ...preset.viewStates,
      [currentView.value]: state,
    }))
  }

  syncStretchControlsFromState(state)
}

function persistCurrentViewStretch(nextMin, nextMax) {
  const taskId = String(activeTask.value?.task_id || '')
  const node = activeFitsNode.value
  if (!taskId || !node?.pixels) {
    return
  }

  const preset = ensureTaskStretchPreset(taskId)
  const nextState = buildViewStretchState(node, nextMin, nextMax)
  saveTaskStretchPreset(taskId, buildPreset({
    ...preset?.viewStates,
    [currentView.value]: nextState,
  }))
  syncStretchControlsFromState(nextState)
}

const activeFitsNode = computed(
  () => fitsNodes.value.find((node) => node.view === currentView.value) ?? null,
)

const stretchedRgba = computed(() => {
  const pixels = activeFitsNode.value?.pixels
  return renderStretchToRgba(pixels, stretchMin.value, stretchMax.value, invertDisplay.value)
})

const stretchDebug = computed(() => Array.from(stretchedRgba.value.slice(0, 16)).join(','))

const {
  blinkInterval,
  blinkEnabled,
  toggleBlink,
  setBlinkInterval,
} = useBlinkControl({
  currentView,
  setCurrentView,
  blinkOrder: blinkQueueViews,
})

function onBlinkToggle() {
  toggleBlink()
}

function onBlinkIntervalInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  setBlinkInterval(value)
}

function onBlinkViewToggle(view, event) {
  blinkViewSelection.value = {
    ...blinkViewSelection.value,
    [view]: Boolean(event?.target?.checked),
  }
}

function clampAutoSubmitInterval(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return autoSubmitIntervalSeconds.value
  }
  return Math.max(
    AUTO_SUBMIT_MIN_SECONDS,
    Math.min(AUTO_SUBMIT_MAX_SECONDS, Math.round(numeric)),
  )
}

function stopAutoSubmitTimer() {
  if (autoSubmitTimerId) {
    window.clearInterval(autoSubmitTimerId)
    autoSubmitTimerId = null
  }
}

function resetAutoSubmitSchedule(baseMs = Date.now()) {
  autoSubmitNowMs.value = baseMs
  if (!autoSubmitEnabled.value || !activeTask.value) {
    autoSubmitNextRunAtMs.value = 0
    return
  }
  autoSubmitNextRunAtMs.value = baseMs + autoSubmitIntervalSeconds.value * 1000
}

function stopAutoSubmit() {
  stopAutoSubmitTimer()
  autoSubmitNowMs.value = Date.now()
  autoSubmitNextRunAtMs.value = 0
}

function startAutoSubmitTimer() {
  stopAutoSubmitTimer()
  if (!autoSubmitEnabled.value || !activeTask.value) {
    autoSubmitNextRunAtMs.value = 0
    return
  }

  resetAutoSubmitSchedule(Date.now())
  autoSubmitTimerId = window.setInterval(() => {
    const now = Date.now()
    autoSubmitNowMs.value = now
    if (
      autoSubmitNextRunAtMs.value > 0
      && now >= autoSubmitNextRunAtMs.value
      && !isSubmitting.value
      && !autoSubmitInFlight.value
    ) {
      void runAutoSubmit()
    }
  }, 1000)
}

function onAutoSubmitToggle(event) {
  autoSubmitEnabled.value = Boolean(event?.target?.checked)
  if (!autoSubmitEnabled.value) {
    autoSubmitStatusMessage.value = ''
  }
}

function onAutoSubmitIntervalInput(event) {
  const nextValue = clampAutoSubmitInterval(event?.target?.value)
  autoSubmitIntervalSeconds.value = nextValue
  if (event?.target) {
    event.target.value = String(nextValue)
  }
}

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
  manualCropRect.value = null
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

async function activateTask(task, index, options = {}) {
  const previousTaskId = String(activeTask.value?.task_id || '')
  const shouldReleasePrevious = options.releasePrevious !== false

  activeTask.value = task
  claimedTaskId.value = String(task.task_id || '')
  taskLockExpiresAt.value = String(options.lockExpiresAt || '')
  taskLockError.value = ''
  taskLockNotice.value = 'Current task is locked by this client'
  taskSwitchError.value = ''
  autoSubmitStatusMessage.value = ''

  setCurrentView('new')
  await preloadTaskFits(task)
  ensureTaskStretchPreset(task.task_id)
  syncStretchControlsForActiveView()

  currentTaskIndex.value = index
  resetAnnotationStates()
  await loadLatestRevisionAnnotations(task.task_id)
  emit('task-changed', task.task_id)
  startTaskHeartbeat(task.task_id)

  if (shouldReleasePrevious && previousTaskId && previousTaskId !== task.task_id) {
    void releaseActiveTask(previousTaskId)
  }
  return true
}

async function loadTaskAtIndex(index, options = {}) {
  if (index < 0 || index >= taskList.value.length) {
    return false
  }

  const task = taskList.value[index]
  clearTaskSwitchError()

  if (isTaskLockedByOtherClient(task)) {
    setTaskSwitchBlockedMessage(task.task_id)
    return false
  }

  if (claimedTaskId.value === task.task_id && activeTask.value?.task_id === task.task_id) {
    return activateTask(task, index, { releasePrevious: false, lockExpiresAt: taskLockExpiresAt.value })
  }

  clearTaskLockMessages()
  try {
    const claimed = await claimTask(task.task_id, taskClientId)
    const claimedTask = {
      ...task,
      ...claimed,
    }
    return await activateTask(claimedTask, index, {
      releasePrevious: options.releasePrevious !== false,
      lockExpiresAt: claimed.lock_expires_at,
      })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to claim task'
    if (message === 'Task locked by another client') {
      setTaskSwitchBlockedMessage(task.task_id)
    } else {
      setTaskSwitchFailureMessage(task.task_id, message)
    }
    if (activeTask.value?.task_id && claimedTaskId.value === activeTask.value.task_id) {
      taskLockNotice.value = 'Current task is locked by this client'
    } else {
      taskLockError.value = message
      taskLockNotice.value = ''
    }
    return false
  }
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
    manualCropRect.value = normalizeManualCropFromMetadata(detail?.metadata)

    if (detail?.source_view && ['new', 'new_marked', 'old'].includes(detail.source_view)) {
      setCurrentView(detail.source_view)
    }
  } catch {
    // 历史读取失败时保持空状态，避免阻塞任务切换。
  }
}

async function refreshTaskList() {
  const currentTaskId = String(activeTask.value?.task_id || '')
  const tasks = await fetchTasks(taskClientId)
  taskList.value = tasks

  if (tasks.length === 0) {
    currentTaskIndex.value = -1
    return tasks
  }

  if (currentTaskId) {
    const currentIndex = tasks.findIndex((task) => task.task_id === currentTaskId)
    if (currentIndex >= 0) {
      currentTaskIndex.value = currentIndex
      if (activeTask.value?.task_id === currentTaskId) {
        activeTask.value = {
          ...activeTask.value,
          ...tasks[currentIndex],
        }
      }
      return tasks
    }
  }

  currentTaskIndex.value = Math.max(0, Math.min(currentTaskIndex.value, tasks.length - 1))
  return tasks
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

async function openTaskCatalog() {
  taskCatalogQuery.value = ''
  taskCatalogVisible.value = true
  try {
    await refreshTaskList()
  } catch {
    // Keep the current catalog contents if the refresh fails.
  }
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
  persistCurrentViewStretch(Math.min(value, stretchMax.value), stretchMax.value)
}

function onStretchMaxInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  persistCurrentViewStretch(stretchMin.value, Math.max(value, stretchMin.value))
}

function onAutoStretchToggle(event) {
  autoStretchEnabled.value = Boolean(event?.target?.checked)
  const taskId = String(activeTask.value?.task_id || '')
  if (!taskId) {
    syncStretchControlsForActiveView()
    return
  }

  const nodesByView = getNodesByView()
  const viewStates = autoStretchEnabled.value
    ? buildBrightnessMatchViewStatesByView(nodesByView, DEFAULT_BRIGHTNESS_MATCH_OPTIONS)
    : buildFullRangeViewStatesByView(nodesByView)
  saveTaskStretchPreset(taskId, buildPreset(viewStates))
  syncStretchControlsForActiveView()
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function applyStretchForNode(node) {
  if (!node?.pixels || node.pixels.length === 0) {
    return
  }
  syncStretchControlsForActiveView()
  syncStageSizeToHost()
}

function onMatchGroupStretch() {
  const taskId = String(activeTask.value?.task_id || '')
  const node = activeFitsNode.value
  if (!taskId || !node?.pixels) {
    return
  }

  const preset = ensureTaskStretchPreset(taskId)
  const sourceState = preset?.viewStates?.[currentView.value] || buildViewStretchState(node, stretchMin.value, stretchMax.value)
  const nodesByView = getNodesByView()
  const matchedStates = matchViewStatesFromSourceState(
    nodesByView,
    currentView.value,
    sourceState,
    DEFAULT_BRIGHTNESS_MATCH_OPTIONS,
  )
  saveTaskStretchPreset(taskId, buildPreset({
    ...preset?.viewStates,
    ...matchedStates,
  }))
  syncStretchControlsForActiveView()
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

function onPointRadiusInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  pointRadius.value = Math.max(2, Math.min(12, value))
}

function onPolygonStrokeWidthInput(event) {
  const value = Number(event?.target?.value)
  if (!Number.isFinite(value)) {
    return
  }
  polygonStrokeWidth.value = Math.max(1, Math.min(8, value))
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

function getSubmittableBboxAnnotations() {
  return annotations.value
    .filter(
      (ann) => (
        ann.type === 'bbox'
        && ann.label !== 'Unlabeled'
        && Number.isFinite(Number(ann.x))
        && Number.isFinite(Number(ann.y))
        && Number.isFinite(Number(ann.width))
        && Number.isFinite(Number(ann.height))
        && Number(ann.x) >= 0
        && Number(ann.y) >= 0
        && Number(ann.width) >= 0
        && Number(ann.height) >= 0
      )
    )
    .map((ann) => {
      if (!manualCropRect.value) {
        return ann
      }
      const clipped = clipRectToCrop(ann)
      if (!clipped) {
        return null
      }
      return {
        ...ann,
        ...clipped,
      }
    })
    .filter(Boolean)
}

function deriveLegacyBucket(bboxAnnotations) {
  return bboxAnnotations.every((ann) => ann.label === 'bogus') ? 'negative' : 'positive'
}

function buildAnnotationPayload(bboxAnnotations) {
  const metadata = {
    tool: 'bbox',
    format_version: 'v2',
  }
  if (manualCropRect.value) {
    metadata.manual_crop = {
      x: manualCropRect.value.x,
      y: manualCropRect.value.y,
      width: manualCropRect.value.width,
      height: manualCropRect.value.height,
    }
  }

  return {
    // Keep the legacy bucket field for older deployed backends that still require it.
    bucket: deriveLegacyBucket(bboxAnnotations),
    source_view: currentView.value,
    metadata,
    annotations: bboxAnnotations.map((ann) => ({
      x: ann.x,
      y: ann.y,
      width: ann.width,
      height: ann.height,
      label: ann.label,
      detail_type: ann.detail_type,
    })),
  }
}

async function saveActiveTaskAnnotations(options = {}) {
  if (!hasTaskLock.value) {
    throw new Error('Current task is not locked by this client')
  }

  if (!activeTask.value) {
    throw new Error('No active task')
  }

  const bboxAnnotations = getSubmittableBboxAnnotations()
  const hasManualCrop = Boolean(manualCropRect.value)
  if (bboxAnnotations.length === 0 && !hasManualCrop) {
    throw new Error('No valid bbox annotations to submit')
  }

  const savedTaskId = activeTask.value.task_id
  const response = await submitAnnotations(savedTaskId, buildAnnotationPayload(bboxAnnotations), {
    clientId: taskClientId,
    releaseAfterSave: Boolean(options.releaseAfterSave),
  })

  return {
    response,
    savedTaskId,
  }
}

function formatAutoSubmitError(err) {
  const message = err instanceof Error ? err.message : 'Failed to submit annotations'
  if (message === '会话已过期，请重新登录' || message === 'Session expired. Please log in again') {
    return '自动提交已停止：会话已过期，请重新登录'
  }
  if (message === 'Current task is not locked by this client') {
    return '自动提交已跳过：当前任务未被本会话持有'
  }
  if (message === 'No active task') {
    return '自动提交等待任务'
  }
  if (message === 'No valid bbox annotations to submit') {
    return '自动提交已跳过：暂无可提交标注'
  }
  return `自动提交失败：${message}`
}

async function runAutoSubmit() {
  if (isSubmitting.value || autoSubmitInFlight.value) {
    return
  }

  autoSubmitInFlight.value = true
  isSubmitting.value = true
  autoSubmitStatusMessage.value = '正在自动提交当前任务...'

  try {
    const { response, savedTaskId } = await saveActiveTaskAnnotations({
      releaseAfterSave: false,
    })
    autoSubmitStatusMessage.value = `自动提交成功：已保存 ${response.saved_count} 条标注`
    emit('annotations-saved', savedTaskId)
  } catch (err) {
    const nextMessage = formatAutoSubmitError(err)
    autoSubmitStatusMessage.value = nextMessage
    if (nextMessage.includes('会话已过期')) {
      autoSubmitEnabled.value = false
    }
  } finally {
    autoSubmitInFlight.value = false
    isSubmitting.value = false
    resetAutoSubmitSchedule(Date.now())
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
    'bogus:disappeared_asteroid': '#d946ef',
    'bogus:disappeared_star': '#a855f7',
    'bogus:disappeared_galaxy': '#8b5e3c',
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

function clampRectToImage(rect) {
  const imageRect = activeImageRect.value
  if (!rect || imageRect.width <= 0 || imageRect.height <= 0) {
    return null
  }
  const x0 = clamp(rect.x, imageRect.x, imageRect.x + imageRect.width)
  const y0 = clamp(rect.y, imageRect.y, imageRect.y + imageRect.height)
  const x1 = clamp(rect.x + rect.width, imageRect.x, imageRect.x + imageRect.width)
  const y1 = clamp(rect.y + rect.height, imageRect.y, imageRect.y + imageRect.height)
  if (x1 <= x0 || y1 <= y0) {
    return null
  }
  return {
    x: x0,
    y: y0,
    width: x1 - x0,
    height: y1 - y0,
  }
}

function pointInRect(point, rect) {
  if (!point || !rect) {
    return false
  }
  return (
    point.x >= rect.x
    && point.y >= rect.y
    && point.x <= rect.x + rect.width
    && point.y <= rect.y + rect.height
  )
}

function clipRectToCrop(rect) {
  if (!rect) {
    return null
  }
  if (!manualCropRect.value) {
    return {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    }
  }
  const normalizedRect = clampRectToImage(rect)
  if (!normalizedRect) {
    return null
  }
  const crop = manualCropRect.value
  const x0 = Math.max(normalizedRect.x, crop.x)
  const y0 = Math.max(normalizedRect.y, crop.y)
  const x1 = Math.min(normalizedRect.x + normalizedRect.width, crop.x + crop.width)
  const y1 = Math.min(normalizedRect.y + normalizedRect.height, crop.y + crop.height)
  if (x1 <= x0 || y1 <= y0) {
    return null
  }
  return {
    x: x0,
    y: y0,
    width: x1 - x0,
    height: y1 - y0,
  }
}

function normalizeManualCropFromMetadata(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    return null
  }
  const manualCrop = metadata.manual_crop
  if (!manualCrop || typeof manualCrop !== 'object') {
    return null
  }
  const x = Number(manualCrop.x)
  const y = Number(manualCrop.y)
  const width = Number(manualCrop.width)
  const height = Number(manualCrop.height)
  if (![x, y, width, height].every(Number.isFinite)) {
    return null
  }
  if (width <= 1 || height <= 1) {
    return null
  }
  return clampRectToImage({ x, y, width, height })
}

function isPointInsideManualCrop(point) {
  if (!manualCropRect.value) {
    return true
  }
  return pointInRect(point, manualCropRect.value)
}

function applyManualCropRect(nextRect) {
  const normalized = clampRectToImage(nextRect)
  manualCropRect.value = normalized
  if (!normalized) {
    return
  }

  annotations.value = annotations.value
    .map((ann) => {
      if (ann.type === 'bbox') {
        const clipped = clipRectToCrop(ann)
        if (!clipped) {
          return null
        }
        return { ...ann, ...clipped }
      }
      if (ann.type === 'point') {
        return isPointInsideManualCrop(ann) ? ann : null
      }
      if (ann.type === 'polygon') {
        const points = Array.isArray(ann.points) ? ann.points : []
        if (points.length === 0) {
          return null
        }
        return points.every((point) => isPointInsideManualCrop(point)) ? ann : null
      }
      return ann
    })
    .filter(Boolean)

  if (selectedAnnotationId.value) {
    const stillExists = annotations.value.some((item) => item.id === selectedAnnotationId.value)
    if (!stillExists) {
      selectedAnnotationId.value = ''
      selectedLabel.value = 'Unlabeled'
    }
  }
}

function clearManualCrop() {
  manualCropRect.value = null
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

  if (toolMode.value !== 'bbox' && toolMode.value !== 'crop') {
    return
  }

  const pointer = getPointer(event)
  if (!pointer) {
    return
  }

  if (toolMode.value === 'bbox' && !isPointInsideManualCrop(pointer)) {
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

  if ((toolMode.value !== 'bbox' && toolMode.value !== 'crop') || !drawStart.value) {
    return
  }

  const pointer = getPointer(event)
  if (!pointer) {
    return
  }

  const rect = normalizeRect(drawStart.value, pointer)
  draftRect.value = toolMode.value === 'crop' ? clampRectToImage(rect) : clipRectToCrop(rect)
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
    if (!isPointInsideManualCrop(pointer)) {
      return
    }
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
    if (!isPointInsideManualCrop(pointer)) {
      return
    }
    currentPolygonPoints.value = [...currentPolygonPoints.value, { x: pointer.x, y: pointer.y }]
    return
  }

  if (toolMode.value === 'crop' && drawStart.value) {
    const rect = clampRectToImage(normalizeRect(drawStart.value, pointer))
    if (rect && rect.width > 1 && rect.height > 1) {
      applyManualCropRect(rect)
      setToolMode('bbox')
    }
    draftRect.value = null
    drawStart.value = null
    return
  }

  if (toolMode.value === 'bbox' && drawStart.value) {
    const rect = clipRectToCrop(normalizeRect(drawStart.value, pointer))
    if (rect && rect.width > 0 && rect.height > 0) {
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
  autoSubmitStatusMessage.value = ''

  try {
    isSubmitting.value = true
    const shouldReleaseAfterSave = hasNextTask.value
    const { response, savedTaskId } = await saveActiveTaskAnnotations({
      releaseAfterSave: shouldReleaseAfterSave,
    })

    let movedToNextTask = false
    if (hasNextTask.value) {
      movedToNextTask = await loadTaskAtIndex(currentTaskIndex.value + 1)
    }

    if (!movedToNextTask && shouldReleaseAfterSave) {
      const currentIndex = taskList.value.findIndex((task) => task.task_id === savedTaskId)
      if (currentIndex >= 0) {
        await loadTaskAtIndex(currentIndex, { releasePrevious: false })
      }
    }

    saveMessage.value = movedToNextTask
      ? `Saved ${response.saved_count} annotations · switched to next task`
      : `Saved ${response.saved_count} annotations`

    if (!movedToNextTask) {
      resetAnnotationStates()
      await loadLatestRevisionAnnotations(savedTaskId)
    }

    emit('annotations-saved', savedTaskId)
    resetAutoSubmitSchedule(Date.now())
  } catch (err) {
    saveMessage.value = err instanceof Error ? err.message : 'Failed to submit annotations'
  } finally {
    isSubmitting.value = false
  }
}

async function loadInitialTask() {
  try {
    const tasks = await refreshTaskList()
    activeTask.value = null
    claimedTaskId.value = ''
    taskLockExpiresAt.value = ''
    clearTaskLockMessages()
    if (!tasks[0]) {
      return
    }

    const claimed = await claimNextTask(taskClientId)
    const index = taskList.value.findIndex((task) => task.task_id === claimed.task_id)
    if (index < 0) {
      taskList.value = [...taskList.value, claimed]
      await activateTask(claimed, taskList.value.length - 1, {
        releasePrevious: false,
        lockExpiresAt: claimed.lock_expires_at,
      })
      return
    }
    await activateTask(
      {
        ...taskList.value[index],
        ...claimed,
      },
      index,
      {
        releasePrevious: false,
        lockExpiresAt: claimed.lock_expires_at,
      },
    )
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to load initial task'
    taskLockError.value = message
    setError(message === 'No available task' ? '' : message)
  }
}

watch(activeFitsNode, (node) => {
  applyStretchForNode(node)
})

watch(autoStretchEnabled, () => {
  applyStretchForNode(activeFitsNode.value)
})

watch(
  () => props.taskRefreshKey,
  async (value, oldValue) => {
    if (value === oldValue) {
      return
    }
    const taskId = String(activeTask.value?.task_id || '')
    if (!taskId) {
      return
    }
    resetAnnotationStates()
    await loadLatestRevisionAnnotations(taskId)
  },
)

watch(
  [stretchedRgba, activeFitsNode],
  () => {
    redrawFitsCanvas()
  },
  { immediate: true },
)

watch(
  [autoSubmitEnabled, autoSubmitIntervalSeconds, () => activeTask.value?.task_id || ''],
  () => {
    if (!autoSubmitEnabled.value) {
      stopAutoSubmit()
      return
    }
    startAutoSubmitTimer()
  },
)

function onKeyDown(event) {
  const tagName = String(event?.target?.tagName || '').toUpperCase()
  if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT') {
    return
  }

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

  if (event.key === 'm' || event.key === 'M') {
    event.preventDefault()
    onMatchGroupStretch()
    return
  }

  if (event.key === 'h' || event.key === 'H') {
    event.preventDefault()
    setToolMode('move')
    return
  }

  if (event.key === 'c' || event.key === 'C') {
    event.preventDefault()
    setToolMode('bbox')
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
    '9': 'bogus:disappeared_asteroid',
    '0': 'bogus:disappeared_star',
    '-': 'bogus:disappeared_galaxy',
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
  void releaseActiveTask()
  stopTaskHeartbeat()
  stopAutoSubmit()
  stopMiddlePan()
  clearPendingDeleteState()
  clearUndoState()
  if (hostResizeObserver) {
    hostResizeObserver.disconnect()
    hostResizeObserver = null
  }
  window.removeEventListener('keydown', onKeyDown)
})
</script>
