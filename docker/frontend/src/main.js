import { createApp } from 'vue'
import VueKonva from 'vue-konva'
import App from './App.vue'
import router from './router'
import './style.css'

const app = createApp(App)
app.use(VueKonva)
app.use(router)
app.mount('#app')
