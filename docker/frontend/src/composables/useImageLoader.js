import { computed, ref } from 'vue'

import { authFetch } from '../services/authStore'

const VIEW_TO_PATH_KEY = {
  old: 'old_path',
  new: 'new_path',
  new_marked: 'new_marked_path',
}

const VIEW_ORDER = ['old', 'new', 'new_marked']

export function buildRenderUrl(relativePath) {
  const encoded = relativePath
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')
  return `/api/render/${encoded}`
}

function createEmptyNode(view) {
  return {
    view,
    path: null,
    src: null,
    image: null,
    visible: view === 'new',
  }
}

export function useImageLoader(fetchImpl = authFetch) {
  const currentView = ref('new')
  const imageNodes = ref(VIEW_ORDER.map((view) => createEmptyNode(view)))
  const isLoading = ref(false)
  const activeTask = ref(null)
  const error = ref('')

  function releaseObjectUrls() {
    for (const node of imageNodes.value) {
      if (node.src) {
        URL.revokeObjectURL(node.src)
      }
    }
  }

  function syncVisibility() {
    imageNodes.value = imageNodes.value.map((node) => ({
      ...node,
      visible: node.view === currentView.value,
    }))
  }

  function setCurrentView(view) {
    currentView.value = view
    syncVisibility()
  }

  function setError(message) {
    error.value = message
  }

  async function preloadTaskImages(task) {
    activeTask.value = task
    isLoading.value = true
    error.value = ''
    releaseObjectUrls()

    try {
      const loaded = await Promise.all(
        VIEW_ORDER.map(async (view) => {
          const pathKey = VIEW_TO_PATH_KEY[view]
          const relativePath = task?.[pathKey]
          if (!relativePath) {
            return createEmptyNode(view)
          }

          const response = await fetchImpl(buildRenderUrl(relativePath))
          if (!response.ok) {
            throw new Error(`Failed to load ${view} image`)
          }

          const blob = await response.blob()
          const src = URL.createObjectURL(blob)
          const image = typeof Image === 'undefined' ? null : new Image()
          if (image) {
            image.src = src
          }

          return {
            view,
            path: relativePath,
            src,
            image,
            visible: view === currentView.value,
          }
        }),
      )
      imageNodes.value = loaded
      syncVisibility()
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to preload task images'
      imageNodes.value = VIEW_ORDER.map((view) => createEmptyNode(view))
    } finally {
      isLoading.value = false
    }
  }

  const visibleNode = computed(() => imageNodes.value.find((node) => node.visible) ?? null)

  return {
    activeTask,
    currentView,
    error,
    imageNodes,
    isLoading,
    preloadTaskImages,
    releaseObjectUrls,
    setError,
    setCurrentView,
    visibleNode,
  }
}
