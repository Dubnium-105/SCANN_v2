<template>
  <header class="border-b border-slate-800 bg-slate-900 px-4 flex items-center justify-between">
    <h1 class="text-lg font-semibold tracking-wide">SCANN Native Annotation</h1>
    <div class="flex items-center gap-3">
      <span class="text-xs text-slate-400" data-testid="header-username">{{ usernameText }}</span>
      <button
        type="button"
        data-testid="header-logout"
        class="text-xs rounded border border-slate-700 px-2 py-1 text-slate-300"
        @click="$emit('logout')"
      >
        Logout
      </button>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  username: {
    type: String,
    default: '',
  },
})

defineEmits(['logout'])

const usernameText = computed(() => (props.username ? `User: ${props.username}` : 'User: Guest'))
</script>
