<script setup>
import { onMounted, ref } from 'vue'
import { getHealth, getTraces, searchKnowledge, uploadKnowledge } from '../api/index.js'

const health = ref({ collection_name: '—', doc_count: '—', embedding_model: '—' })
const traces = ref([])
const searchQuery = ref('')
const results = ref([])
const fileInput = ref(null)
const busy = ref(false)
const notice = ref('')
const errorMessage = ref('')

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
  try {
    const [healthResponse, traceResponse] = await Promise.all([getHealth(), getTraces()])
    health.value = healthResponse.knowledge_base || health.value
    traces.value = traceResponse.traces || []
  } catch (error) {
    errorMessage.value = error.message || '管理数据读取失败。'
  }
}

async function runSearch() {
  if (!searchQuery.value.trim()) return
  busy.value = true
  errorMessage.value = ''
  try {
    const response = await searchKnowledge(searchQuery.value.trim(), 5)
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
    const response = await uploadKnowledge(file)
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

onMounted(loadOverview)
</script>

<template>
  <section class="admin-page" aria-label="管理控制台">
    <div class="admin-heading">
      <div>
        <h1>管理控制台</h1>
        <p class="muted-copy">知识维护、检索测试与运行期调用轨迹。</p>
      </div>
      <div v-if="notice" class="notice-copy" role="status">{{ notice }}</div>
    </div>

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
          <thead><tr><th>请求ID</th><th>意图</th><th>处理节点</th><th>耗时</th></tr></thead>
          <tbody>
            <tr v-for="trace in traces" :key="trace.request_id"><td>{{ trace.request_id }}</td><td>{{ trace.intent || '—' }}</td><td>{{ (trace.executed_agents || []).join('、') || '—' }}</td><td>{{ formatLatency(trace.latency_ms) }}</td></tr>
            <tr v-if="!traces.length"><td colspan="4" class="empty-cell">暂无调用轨迹</td></tr>
          </tbody>
        </table>
      </div>
      <p class="small-copy">注：仅显示内存中最近记录，服务重启后清空。</p>
    </section>
    <p v-if="errorMessage" class="error-copy" role="alert">{{ errorMessage }}</p>
  </section>
</template>
