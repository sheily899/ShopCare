<script setup>
import { ref } from 'vue'
import { login, registerConsumer } from '../api/index.js'

const emit = defineEmits(['login'])
const role = ref('consumer')
const username = ref('')
const password = ref('')
const errorMessage = ref('')
const mode = ref('login')
const displayName = ref('')

const demoAccounts = {
  consumer: { username: 'consumer-demo', password: 'shopcare', label: '消费者' },
  agent: { username: 'agent-demo', password: 'shopcare', label: '客服' },
}

async function submit() {
  try {
    if (mode.value === 'register') {
      if (role.value !== 'consumer') throw new Error('当前只支持注册消费者账号')
      if (!displayName.value.trim()) throw new Error('请输入昵称')
    }
    const identity = mode.value === 'register'
      ? await registerConsumer(username.value.trim(), password.value, displayName.value.trim())
      : await login(username.value.trim(), password.value)
    if (identity.role !== role.value) throw new Error('请选择与账号匹配的使用端口')
    errorMessage.value = ''
    emit('login', identity)
  } catch (error) {
    errorMessage.value = error.message || '登录失败'
  }
}

function useDemoAccount() {
  const account = demoAccounts[role.value]
  username.value = account.username
  password.value = account.password
}

function selectRole(nextRole) {
  role.value = nextRole
  if (nextRole === 'agent') mode.value = 'login'
}
</script>

<template>
  <main class="login-page" aria-label="ShopCare 登录">
    <section class="login-card">
      <div class="brand login-brand"><span class="brand-mark" aria-hidden="true">S</span><span class="brand-name">ShopCare</span></div>
      <h1>进入服务工作台</h1>
      <p class="muted-copy">请选择你的使用端口</p>
      <div class="role-switch" role="tablist" aria-label="选择端口">
        <button type="button" :class="{ active: role === 'consumer' }" @click="selectRole('consumer')">消费者</button>
        <button type="button" :class="{ active: role === 'agent' }" @click="selectRole('agent')">客服端</button>
      </div>
      <form class="login-form" @submit.prevent="submit">
        <label for="login-username">账号</label>
        <input id="login-username" v-model="username" autocomplete="username" placeholder="请输入演示账号" />
        <label for="login-password">密码</label>
        <input id="login-password" v-model="password" type="password" autocomplete="current-password" placeholder="请输入密码" />
        <label v-if="mode === 'register'" for="display-name">昵称</label>
        <input v-if="mode === 'register'" id="display-name" v-model="displayName" placeholder="请输入昵称" />
        <button class="primary-button" type="submit">{{ mode === 'register' ? '注册并进入' : '登录' }}</button>
      </form>
      <button v-if="role === 'consumer'" class="demo-account-button" type="button" @click="mode = mode === 'login' ? 'register' : 'login'">
        {{ mode === 'login' ? '没有账号？注册消费者账号' : '已有账号？返回登录' }}
      </button>
      <button v-if="mode === 'login'" class="demo-account-button" type="button" @click="useDemoAccount">使用{{ demoAccounts[role].label }}演示账号</button>
      <p v-if="errorMessage" class="error-copy" role="alert">{{ errorMessage }}</p>
    </section>
  </main>
</template>
