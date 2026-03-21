<template>
  <div class="h-full flex items-center justify-center p-4">
    <div class="w-full max-w-sm rounded-lg border border-slate-800 bg-slate-900 p-4 space-y-3">
      <h1 class="text-lg font-semibold">创建账号</h1>
      <p class="text-xs text-slate-400">注册后将以 annotator 身份登录</p>

      <form class="space-y-2" @submit.prevent="onSubmit">
        <input
          v-model="username"
          data-testid="register-username"
          type="text"
          placeholder="用户名"
          class="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        >
        <input
          v-model="password"
          data-testid="register-password"
          type="password"
          placeholder="密码"
          class="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        >
        <input
          v-model="confirmPassword"
          data-testid="register-confirm-password"
          type="password"
          placeholder="确认密码"
          class="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
        >
        <button
          data-testid="register-submit"
          class="w-full rounded border border-emerald-700 bg-emerald-900/40 px-3 py-2 text-sm text-emerald-200"
          :disabled="isSubmitting"
          type="submit"
        >
          {{ isSubmitting ? '注册中...' : '注册' }}
        </button>
      </form>

      <button
        data-testid="register-go-login"
        class="w-full rounded border border-slate-700 px-3 py-2 text-xs text-slate-300"
        @click="router.push({ name: 'login' })"
      >
        返回登录
      </button>

      <p v-if="errorMessage" data-testid="register-error" class="text-xs text-rose-300">{{ errorMessage }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import { registerWithPassword } from '../services/authApi'

const username = ref('')
const password = ref('')
const confirmPassword = ref('')
const isSubmitting = ref(false)
const errorMessage = ref('')
const router = useRouter()

async function onSubmit() {
  errorMessage.value = ''

  const normalizedUsername = username.value.trim()
  if (!normalizedUsername) {
    errorMessage.value = '用户名不能为空'
    return
  }
  if (password.value !== confirmPassword.value) {
    errorMessage.value = '两次输入的密码不一致'
    return
  }

  isSubmitting.value = true
  try {
    await registerWithPassword(normalizedUsername, password.value)
    router.push({ name: 'annotation' })
  } catch (err) {
    errorMessage.value = err instanceof Error ? err.message : '注册失败'
  } finally {
    isSubmitting.value = false
  }
}
</script>
