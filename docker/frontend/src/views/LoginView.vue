<template>
  <div class="h-full flex items-center justify-center p-4">
    <div class="w-full max-w-sm rounded-lg border border-slate-800 bg-slate-900 p-4 space-y-3">
      <h1 class="text-lg font-semibold">SCANN Native Annotation</h1>
      <p class="text-xs text-slate-400">请登录以访问标注界面</p>

      <form class="space-y-2" @submit.prevent="onSubmit">
        <input
          v-model="username"
          data-testid="login-username"
          type="text"
          placeholder="用户名"
          class="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        >
        <input
          v-model="password"
          data-testid="login-password"
          type="password"
          placeholder="密码"
          class="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        >
        <button
          data-testid="login-submit"
          class="w-full rounded border border-sky-700 bg-sky-900/40 px-3 py-2 text-sm text-sky-200"
          :disabled="isSubmitting"
          type="submit"
        >
          {{ isSubmitting ? '登录中...' : '登录' }}
        </button>
      </form>

      <button
        data-testid="login-go-register"
        class="w-full rounded border border-slate-700 px-3 py-2 text-xs text-slate-300"
        @click="router.push({ name: 'register' })"
      >
        没有账号？去注册
      </button>

      <p v-if="errorMessage" data-testid="login-error" class="text-xs text-rose-300">{{ errorMessage }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import { loginWithPassword } from '../services/authApi'

const username = ref('annotator')
const password = ref('scann123')
const isSubmitting = ref(false)
const errorMessage = ref('')
const router = useRouter()

async function onSubmit() {
  errorMessage.value = ''
  isSubmitting.value = true
  try {
    await loginWithPassword(username.value.trim(), password.value)
    router.push({ name: 'annotation' })
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : '登录失败'
  } finally {
    isSubmitting.value = false
  }
}
</script>
