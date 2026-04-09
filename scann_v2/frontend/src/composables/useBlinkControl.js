import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const BLINK_VIEW_ORDER = ['new', 'new_marked', 'old']
const DEFAULT_BLINK_INTERVAL = 500 // 0.5s in milliseconds

function getNextView(currentView, viewOrder = BLINK_VIEW_ORDER) {
  if (!Array.isArray(viewOrder) || viewOrder.length === 0) {
    return currentView
  }
  const currentIndex = viewOrder.indexOf(currentView)
  if (currentIndex === -1) {
    return viewOrder[0]
  }
  return viewOrder[(currentIndex + 1) % viewOrder.length]
}

export function useBlinkControl({ currentView, setCurrentView, blinkOrder = null }) {
  const blinkInterval = ref(DEFAULT_BLINK_INTERVAL)
  const blinkEnabled = ref(false)
  let blinkTimer = null
  const activeBlinkOrder = computed(() => {
    const customOrder = blinkOrder?.value
    if (Array.isArray(customOrder) && customOrder.length > 0) {
      return customOrder
    }
    if (Array.isArray(customOrder) && customOrder.length === 0) {
      return []
    }
    return BLINK_VIEW_ORDER
  })

  function startBlink() {
    const order = activeBlinkOrder.value
    if (order.length === 0) {
      stopBlink()
      return
    }
    stopBlink()
    blinkEnabled.value = true
    blinkTimer = setInterval(() => {
      const nextView = getNextView(currentView.value, order)
      setCurrentView(nextView)
    }, blinkInterval.value)
  }

  function stopBlink() {
    if (blinkTimer) {
      clearInterval(blinkTimer)
      blinkTimer = null
    }
    blinkEnabled.value = false
  }

  function toggleBlink() {
    if (blinkEnabled.value) {
      stopBlink()
    } else {
      startBlink()
    }
  }

  function setBlinkInterval(interval) {
    const value = Number(interval)
    if (!Number.isFinite(value) || value < 100) {
      return
    }
    blinkInterval.value = Math.min(10000, Math.max(100, value))
    if (blinkEnabled.value) {
      startBlink()
    }
  }

  watch(blinkInterval, () => {
    if (blinkEnabled.value) {
      startBlink()
    }
  })

  watch(activeBlinkOrder, (nextOrder) => {
    if (!blinkEnabled.value) {
      return
    }
    if (nextOrder.length === 0) {
      stopBlink()
      return
    }
    if (!nextOrder.includes(currentView.value)) {
      setCurrentView(nextOrder[0])
    }
    startBlink()
  })

  function handleKeydown(event) {
    if (event.key === 'Tab') {
      if (event.target?.tagName === 'INPUT' || event.target?.tagName === 'TEXTAREA') {
        return
      }
      event.preventDefault()
      const order = activeBlinkOrder.value
      if (order.length === 0) {
        stopBlink()
        return
      }
      const nextView = getNextView(currentView.value, order)
      setCurrentView(nextView)
      toggleBlink()
      return
    }

    if (event.key === ' ') {
      if (event.target?.tagName === 'INPUT' || event.target?.tagName === 'TEXTAREA') {
        return
      }
      event.preventDefault()
      const nextView = getNextView(currentView.value)
      setCurrentView(nextView)
    }
  }

  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })

  onBeforeUnmount(() => {
    stopBlink()
    window.removeEventListener('keydown', handleKeydown)
  })

  const blinkOrderRef = computed(() => activeBlinkOrder.value)

  return {
    blinkOrder: blinkOrderRef,
    blinkInterval,
    blinkEnabled,
    toggleBlink,
    setBlinkInterval,
    startBlink,
    stopBlink,
    handleKeydown,
  }
}

export { BLINK_VIEW_ORDER, getNextView, DEFAULT_BLINK_INTERVAL }
