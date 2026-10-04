<script setup>
import { computed, nextTick, ref } from 'vue'
import ChatView from './views/ChatView.vue'
import AdminView from './views/AdminView.vue'

const activeView = ref('chat')
const drawerOpen = ref(false)
const sessionSummaries = ref([])
const chatView = ref(null)
const sessionCount = computed(() => sessionSummaries.value.length)

function handleSessionsChanged(sessions) {
  sessionSummaries.value = sessions
}

function selectHistory(id) {
  activeView.value = 'chat'
  chatView.value?.selectSession(id)
  drawerOpen.value = false
}

async function startNewConversation() {
  activeView.value = 'chat'
  await nextTick()
  chatView.value?.startNewConversation()
  drawerOpen.value = false
}
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand" aria-label="ShopCare">
        <span class="brand-mark" aria-hidden="true">S</span>
        <span class="brand-name">ShopCare</span>
      </div>

      <nav class="view-switcher" aria-label="主视图切换">
        <button class="nav-link" :class="{ active: activeView === 'chat' }" type="button" @click="activeView = 'chat'">会话</button>
        <button class="nav-link" :class="{ active: activeView === 'admin' }" type="button" @click="activeView = 'admin'">管理</button>
      </nav>

      <div class="topbar-actions">
        <button class="history-trigger" type="button" @click="drawerOpen = true">
          历史会话 ({{ sessionCount }})
        </button>
        <button class="new-conversation-button" type="button" @click="startNewConversation">新对话</button>
      </div>
    </header>

    <main class="app-main">
      <ChatView ref="chatView" v-show="activeView === 'chat'" @sessions-changed="handleSessionsChanged" />
      <AdminView v-show="activeView === 'admin'" />
    </main>

    <Transition name="drawer-fade">
      <button v-if="drawerOpen" class="drawer-backdrop" type="button" aria-label="关闭历史会话" @click="drawerOpen = false"></button>
    </Transition>
    <aside class="history-drawer" :class="{ open: drawerOpen }" aria-label="历史咨询记录">
      <div class="drawer-header">
        <h2>历史咨询记录</h2>
        <button class="drawer-close" type="button" @click="drawerOpen = false">关闭</button>
      </div>
      <div class="drawer-list" role="list">
        <button v-for="session in sessionSummaries" :key="session.id" class="history-entry" type="button" role="listitem" @click="selectHistory(session.id)">
          <span class="history-entry-title">{{ session.title }}</span>
          <span class="history-entry-time">{{ session.createdAt }}</span>
        </button>
        <p v-if="!sessionSummaries.length" class="drawer-empty">暂无历史咨询记录</p>
      </div>
    </aside>
  </div>
</template>
