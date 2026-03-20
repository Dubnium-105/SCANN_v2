import { createRouter, createWebHistory } from 'vue-router'

import { isAuthenticated } from '../services/authStore'
import AnnotationView from '../views/AnnotationView.vue'
import LoginView from '../views/LoginView.vue'
import RegisterView from '../views/RegisterView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: LoginView,
    },
    {
      path: '/register',
      name: 'register',
      component: RegisterView,
    },
    {
      path: '/',
      name: 'annotation',
      component: AnnotationView,
      meta: { requiresAuth: true },
    },
  ],
})

router.beforeEach((to) => {
  if (to.meta.requiresAuth && !isAuthenticated()) {
    return { name: 'login' }
  }
  if ((to.name === 'login' || to.name === 'register') && isAuthenticated()) {
    return { name: 'annotation' }
  }
  return true
})

export default router
