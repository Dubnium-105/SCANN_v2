import { computed, onBeforeUnmount, onMounted } from 'vue'

const BLINK_VIEW_ORDER = ['new', 'new_marked', 'old']

function getNextView(currentView, viewOrder = BLINK_VIEW_ORDER) {
  const currentIndex = viewOrder.indexOf(currentView)
  if (currentIndex === -1) {
    return viewOrder[0]
  }
  return viewOrder[(currentIndex + 1) % viewOrder.length]
}

export function useBlinkControl({ currentView, setCurrentView }) {
  function handleKeydown(event) {
    if (event.key !== ' ' && event.key !== 'Tab') {
      return
    }

    event.preventDefault()
    const nextView = getNextView(currentView.value)
    setCurrentView(nextView)
  }

  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('keydown', handleKeydown)
  })

  const blinkOrder = computed(() => BLINK_VIEW_ORDER)

  return {
    blinkOrder,
    handleKeydown,
  }
}

export { BLINK_VIEW_ORDER, getNextView }