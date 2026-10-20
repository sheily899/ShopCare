<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { getHealth, getServiceDashboard, getTraces, searchKnowledge, uploadKnowledge } from '../api/index.js'

const health = ref({ collection_name: '—', doc_count: '—', embedding_model: '—' })
const traces = ref([])
const searchQuery = ref('')
const results = ref([])
const fileInput = ref(null)
const busy = ref(false)
const notice = ref('')
const errorMessage = ref('')
const dashboard = ref({ metrics: {}, alert: '', users: [] })
const props = defineProps({ token: { type: String, default: '' }, dashboardOnly: { type: Boolean, default: false } })

function formatLatency(value) {
  const milliseconds = Number(value)
  if (value === null || value === undefined || value === '' || Number.isNaN(milliseconds) || milliseconds < 0) return '—'
  const seconds = milliseconds / 1000
  if (seconds < 60) return `${seconds.toFixed(2)} 秒`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.round(seconds % 60)
  if (remainingSeconds === 0) return `${minutes} 分钟`
  return `${minutes} 分 ${String(remainingSeconds).padStart(2, '0')} 秒`
}

function shortEmbeddingModel(value) {
  const text = String(value || '').trim()
  if (!text || text === '—') return '—'
  const model = text.split(/[\\/]/).pop()?.replace(/^BAAI[\\/]/, '') || text
  return `${model} (CPU)`
}

async function loadOverview() {
  // 三类数据分别加载：健康检查失败不能阻断客服读取会话记录。
  const [healthResult, traceResult, dashboardResult] = await Promise.allSettled([
    getHealth(),
    getTraces(props.token),
    getServiceDashboard(props.token),
  ])
  if (healthResult.status === 'fulfilled') health.value = healthResult.value.knowledge_base || health.value
  if (traceResult.status === 'fulfilled') traces.value = traceResult.value.traces || []
  if (dashboardResult.status === 'fulfilled') dashboard.value = dashboardResult.value
}

async function runSearch() {
  if (!searchQuery.value.trim()) return
  busy.value = true
  errorMessage.value = ''
  try {
    const response = await searchKnowledge(searchQuery.value.trim(), 5, props.token)
    results.value = response.results || []
  } catch (error) {
    errorMessage.value = error.message || '检索失败。'
  } finally {
    busy.value = false
  }
}

function chooseFile() {
  fileInput.value?.click()
}

async function handleFile(event) {
  const file = event.target.files?.[0]
  if (!file) return
  busy.value = true
  errorMessage.value = ''
  notice.value = ''
  try {
    const response = await uploadKnowledge(file, props.token)
    notice.value = `${file.name} 已入库，生成 ${response.added} 个片段。`
    await loadOverview()
  } catch (error) {
    errorMessage.value = error.message || '文件上传失败。'
  } finally {
    busy.value = false
    event.target.value = ''
  }
}

function dropFile(event) {
  event.preventDefault()
  const file = event.dataTransfer.files?.[0]
  if (file) {
    const transfer = { target: { files: [file], value: '' } }
    handleFile(transfer)
  }
}

let refreshTimer = null
onMounted(() => {
  loadOverview()
  // 看板与消费者会话共享 SQLite 快照，保持短轮询即可让新会话自动出现。
  refreshTimer = window.setInterval(loadOverview, 3000)
})
onUnmounted(() => {
  if (refreshTimer) window.clearInterval(refreshTimer)
})
</script>

<template>
  <section class="admin-page" aria-label="客服工作台">
    <div class="admin-heading">
      <div>
        <h1>{{ dashboardOnly ? '风险看板' : '客服工作台' }}</h1>
        <p class="muted-copy">{{ dashboardOnly ? '汇总全部会话的风险、情绪、工单和处理状态。' : '当前阶段展示客服可见的知识、检索与运行状态。' }}</p>
      </div>
      <div v-if="notice" class="notice-copy" role="status">{{ notice }}</div>
    </div>

    <template v-if="dashboardOnly">
      <div class="stat-grid" aria-label="风险看板概况">
        <div class="stat-card"><span class="stat-label">全部会话</span><strong class="stat-value">{{ dashboard.metrics.total_sessions ?? dashboard.metrics.today_sessions ?? 0 }}</strong></div>
        <div class="stat-card"><span class="stat-label">AI 独立解决率</span><strong class="stat-value">{{ dashboard.metrics.ai_resolution_rate || 0 }}%</strong></div>
        <div class="stat-card"><span class="stat-label">转人工率</span><strong class="stat-value">{{ dashboard.metrics.handoff_rate || 0 }}%</strong></div>
        <div class="stat-card danger-stat"><span class="stat-label">待处理 P0 / P1</span><strong class="stat-value">{{ dashboard.metrics.pending_p01 || 0 }}</strong></div>
        <div class="stat-card"><span class="stat-label">待审核工单</span><strong class="stat-value">{{ dashboard.metrics.pending_tickets || 0 }}</strong></div>
      </div>
      <div class="dashboard-alert">{{ dashboard.alert || '暂无风险提醒' }}</div>
      <section class="admin-card dashboard-users"><div class="section-heading"><h2>用户列表</h2><span class="muted-copy">按紧急程度和更新时间排序</span></div><div class="result-table-wrap"><table class="data-table"><thead><tr><th>用户</th><th>紧急度</th><th>情绪</th><th>风险标签</th><th>工单</th><th>状态</th></tr></thead><tbody><tr v-for="user in dashboard.users" :key="user.conv_id"><td>{{ user.user_id }}</td><td><span :class="`priority-badge priority-${user.urgency}`">{{ user.urgency }}</span></td><td>{{ { calm: '平静', neutral: '平静', 平静: '平静', anxious: '焦虑', worried: '焦虑', 焦虑: '焦虑', upset: '不满', disappointed: '不满', dissatisfied: '不满', 不满: '不满', angry: '愤怒', 愤怒: '愤怒', threatening: '威胁升级', 威胁升级: '威胁升级' }[user.emotion] || '待识别' }}</td><td><span v-for="tag in user.risk_tags" :key="tag" class="summary-tag">{{ tag }}</span><span v-if="!user.risk_tags.length">无</span></td><td>{{ user.ticket_id || '无工单' }} {{ user.ticket_status !== '无工单' ? user.ticket_status : '' }}</td><td>{{ user.status === 'waiting_human' ? '人工中' : user.status === 'ai_processing' ? 'AI处理中' : user.status }}</td></tr><tr v-if="!dashboard.users.length"><td colspan="6" class="empty-cell">暂无会话数据</td></tr></tbody></table></div></section>
    </template>

    <template v-else>
    <div class="stat-grid" aria-label="知识库概况">
      <div class="stat-card"><span class="stat-label">知识库集合</span><strong class="stat-value">{{ health.collection_name }}</strong></div>
      <div class="stat-card"><span class="stat-label">切片总数</span><strong class="stat-value">{{ health.doc_count }}</strong></div>
      <div class="stat-card"><span class="stat-label">嵌入模型</span><strong class="stat-value" :title="shortEmbeddingModel(health.embedding_model)">{{ shortEmbeddingModel(health.embedding_model) }}</strong></div>
      <div class="stat-card"><span class="stat-label">断路器状态</span><strong class="stat-value status-ok">正常</strong></div>
    </div>

    <div class="admin-grid">
      <section class="admin-card">
        <h2>文档上传</h2>
        <div class="upload-box" @dragover.prevent @drop="dropFile">
          <p>拖拽 txt, md, json 文件至此 (上限 10MB)</p>
          <button class="outline-button" type="button" @click="chooseFile">选择文件</button>
          <input ref="fileInput" class="sr-only" type="file" accept=".txt,.md,.json" @change="handleFile" />
        </div>
      </section>

      <section class="admin-card">
        <h2>检索测试探针</h2>
        <form class="search-form" @submit.prevent="runSearch">
          <label class="sr-only" for="search-input">检索词</label>
          <input id="search-input" v-model="searchQuery" type="search" placeholder="输入检索问题" />
          <button class="primary-button" type="submit" :disabled="busy || !searchQuery.trim()">测试检索</button>
        </form>
        <div class="result-table-wrap">
          <table class="data-table">
            <thead><tr><th>标题</th><th>相似度</th><th>文本预览</th></tr></thead>
            <tbody>
              <tr v-for="result in results" :key="result.chunk_id">
                <td>{{ result.title }}</td><td>{{ result.score }}</td><td>{{ result.content }}</td>
              </tr>
              <tr v-if="!results.length"><td colspan="3" class="empty-cell">暂无检索结果</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>

    <section class="admin-card trace-card">
      <div class="section-heading"><h2>调用轨迹</h2><span class="muted-copy">最近 200 条</span></div>
      <div class="result-table-wrap">
        <table class="data-table">
          <thead><tr><th>请求ID</th><th>意图</th><th>处理节点</th><th>编排耗时</th></tr></thead>
          <tbody>
            <tr v-for="trace in traces" :key="trace.request_id"><td>{{ trace.request_id }}</td><td>{{ trace.intent || '—' }}</td><td>{{ (trace.executed_agents || []).join('、') || '—' }}</td><td>{{ formatLatency(trace.latency_ms) }}</td></tr>
            <tr v-if="!traces.length"><td colspan="4" class="empty-cell">暂无调用轨迹</td></tr>
          </tbody>
        </table>
      </div>
      <p class="small-copy">注：仅显示内存中最近记录，服务重启后清空。</p>
    </section>
    </template>
    <p v-if="errorMessage" class="error-copy" role="alert">{{ errorMessage }}</p>
  </section>
</template>
