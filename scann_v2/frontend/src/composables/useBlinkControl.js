import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const BLINK_VIEW_ORDER = ['new', 'new_marked', 'old']
const DEFAULT_BLINK_INTERVAL = 500 // 0.5s in milliseconds

function getNextView(currentView, viewOrder = BLINK_VIEW_ORDER) {
  const currentIndex = viewOrder.indexOf(currentView)
  if (currentIndex === -1) {
    return viewOrder[0]
  }
  return viewOrder[(currentIndex + 1) % viewOrder.length]
}

export function useBlinkControl({ currentView, setCurrentView }) {
  const blinkInterval = ref(DEFAULT_BLINK_INTERVAL)
  const blinkEnabled = ref(false)
  let blinkTimer = null

  function startBlink() {
    if (blinkTimer) {
      clearInterval(blinkTimer)
    }
    blinkTimer = setInterval(() => {
      const nextView = getNextView(currentView.value)
      setCurrentView(nextView)
    }, blinkInterval.value)
  }

  function stopBlink() {
    if (blinkTimer) {
      clearInterval(blinkTimer)
      blinkTimer = null
    }
  }

  function toggleBlink() {
    if (blinkEnabled.value) {
      stopBlink()
      blinkEnabled.value = false
    } else {
      startBlink()
      blinkEnabled.value = true
    }
  }

  function setBlinkInterval(interval) {
    const value = Number(interval)
    if (!Number.isFinite(value) || value < 100) {
      return
    }
    blinkInterval.value = Math.min(10000, Math.max(100, value))
    // If blinking is active, restart with new interval
    if (blinkEnabled.value) {
      stopBlink()
      startBlink()
    }
  }

  // Watch for interval changes to apply immediately when blinking is active
  watch(blinkInterval, () => {
    if (blinkEnabled.value) {
      stopBlink()
      startBlink()
    }
  })

  function handleKeydown(event) {
    // Tab key toggles blink on/off
    if (event.key === 'Tab') {
      // Don't trigger if user is typing in an input
      if (event.target?.tagName === 'INPUT' || event.target?.tagName === 'TEXTAREA') {
        return
      }
      event.preventDefault()
      const nextView = getNextView(currentView.value)
      setCurrentView(nextView)
      toggleBlink()
      return
    }

    // Space key switches to next view
    if (event.key === ' ') {
      // Don't trigger if user is typing in an input
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

  const blinkOrder = computed(() => BLINK_VIEW_ORDER)

  return {
    blinkOrder,
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
