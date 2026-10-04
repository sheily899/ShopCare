<script setup>
import { ref } from 'vue'
import ChatView from './views/ChatView.vue'
import AdminView from './views/AdminView.vue'

const activeView = ref('chat')
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">ShopCare</div>
      <nav class="view-nav" aria-label="主视图切换">
        <button
          class="nav-link"
          :class="{ active: activeView === 'chat' }"
          type="button"
          @click="activeView = 'chat'"
        >
          会话
        </button>
        <button
          class="nav-link"
          :class="{ active: activeView === 'admin' }"
          type="button"
          @click="activeView = 'admin'"
        >
          管理
        </button>
      </nav>
    </header>

    <main class="app-main">
      <ChatView v-if="activeView === 'chat'" />
      <AdminView v-else />
    </main>
  </div>
</template>
