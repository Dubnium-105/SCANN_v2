import { ref } from 'vue'

import { parseFitsArrayBuffer } from '../fits/fitsParser'
import { fetchFitsArrayBuffer } from '../services/fitsApi'

const VIEW_PATH_MAP = {
  old: 'old_path',
  new: 'new_path',
  new_marked: 'new_marked_path',
}

const VIEW_ORDER = ['old', 'new', 'new_marked']

function createEmptyFitsNode(view) {
  return {
    view,
    path: null,
    headers: null,
    pixels: null,
    width: 0,
    height: 0,
  }
}

export function useFitsImagePool(fetchImpl = fetch) {
  const fitsNodes = ref(VIEW_ORDER.map((view) => createEmptyFitsNode(view)))
  const isFitsLoading = ref(false)
  const fitsError = ref('')

  async function preloadTaskFits(task) {
    isFitsLoading.value = true
    fitsError.value = ''

    try {
      const loaded = await Promise.all(
        VIEW_ORDER.map(async (view) => {
          const path = task?.[VIEW_PATH_MAP[view]]
          if (!path) {
            return createEmptyFitsNode(view)
          }

          const buffer = await fetchFitsArrayBuffer(path, fetchImpl)
          const parsed = parseFitsArrayBuffer(buffer)
          return {
            view,
            path,
            headers: parsed.headers,
            pixels: parsed.pixels,
            width: parsed.width,
            height: parsed.height,
          }
        }),
      )

      fitsNodes.value = loaded
    } catch (err) {
      fitsError.value = err instanceof Error ? err.message : 'Failed to preload FITS data'
      fitsNodes.value = VIEW_ORDER.map((view) => createEmptyFitsNode(view))
    } finally {
      isFitsLoading.value = false
    }
  }

  return {
    fitsNodes,
    fitsError,
    isFitsLoading,
    preloadTaskFits,
  }
}
