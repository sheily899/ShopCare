<script setup>
import { computed, nextTick, ref } from 'vue'
import ChatView from './views/ChatView.vue'
import AdminView from './views/AdminView.vue'
import LoginView from './views/LoginView.vue'
import ServiceWorkbenchView from './views/ServiceWorkbenchView.vue'

const activeView = ref('chat')
const identity = ref(null)
const drawerOpen = ref(false)
const sessionSummaries = ref([])
const chatView = ref(null)
const sessionCount = computed(() => sessionSummaries.value.length)

function handleLogin(nextIdentity) {
  identity.value = nextIdentity
  activeView.value = nextIdentity.role === 'agent' ? 'service-chat' : 'chat'
}

function logout() {
  identity.value = null
  sessionSummaries.value = []
}

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
  <LoginView v-if="!identity" @login="handleLogin" />
  <div v-else class="app-shell" :class="{ 'app-shell-viewport': ['chat', 'service-chat'].includes(activeView), 'app-shell-service': identity.role === 'agent' && activeView === 'service-chat' }">
    <header class="topbar">
      <div class="brand" aria-label="ShopCare">
        <span class="brand-mark" aria-hidden="true">S</span>
        <span class="brand-name">ShopCare</span>
      </div>

      <nav class="view-switcher" aria-label="主视图切换">
        <button class="nav-link" :class="{ active: activeView === 'chat' || activeView === 'service-chat' }" type="button" @click="activeView = identity.role === 'agent' ? 'service-chat' : 'chat'">会话</button>
        <button v-if="identity.role === 'agent'" class="nav-link" :class="{ active: activeView === 'agent' }" type="button" @click="activeView = 'agent'">工作台</button>
        <button v-if="identity.role === 'agent'" class="nav-link" :class="{ active: activeView === 'risk-dashboard' }" type="button" @click="activeView = 'risk-dashboard'">风险看板</button>
      </nav>

      <div class="topbar-actions">
        <button v-if="identity.role === 'consumer'" class="history-trigger" type="button" @click="drawerOpen = true">
          历史会话 ({{ sessionCount }})
        </button>
        <button v-if="identity.role === 'consumer'" class="new-conversation-button" type="button" @click="startNewConversation">新对话</button>
        <button class="history-trigger" type="button" @click="logout">退出</button>
      </div>
    </header>

    <main class="app-main" :class="{ 'app-main-chat': identity.role === 'consumer' && activeView === 'chat', 'app-main-service': identity.role === 'agent' && activeView === 'service-chat' }">
      <ChatView v-if="identity.role === 'consumer'" ref="chatView" v-show="activeView === 'chat'" :user-id="identity.user_id" :token="identity.token" @sessions-changed="handleSessionsChanged" />
      <ServiceWorkbenchView v-if="identity.role === 'agent'" v-show="activeView === 'service-chat'" :agent-id="identity.user_id" :token="identity.token" />
      <AdminView v-if="identity.role === 'agent'" v-show="activeView === 'agent' || activeView === 'risk-dashboard'" :token="identity.token" :dashboard-only="activeView === 'risk-dashboard'" />
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
