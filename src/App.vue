<template>
  <div class="app-container">
    <el-container class="main-layout">
      <el-header height="64px" class="header">
        <div class="header-inner">
          <div class="logo-group">
            <el-icon color="#409EFF" size="24">
              <Monitor/>
            </el-icon>
            <span class="logo">AI Function Approximator <small class="version">v2.1</small></span>
          </div>
          <div class="actions">
            <el-button v-if="model" type="info" :icon="View" @click="showWeights = true" plain round>
              View Weights
            </el-button>
            <el-button type="primary" @click="handleTrain" :loading="isTraining" size="large" round>
              {{ isTraining ? 'TRAINING...' : 'RUN TRAINING' }}
            </el-button>
          </div>
        </div>
      </el-header>

      <el-main class="content-wrapper">
        <el-row :gutter="20" class="full-height-row">
          <el-col :lg="14" :md="24" class="scroll-col">
            <el-card shadow="never" class="card">
              <template #header>
                <div class="card-header">
                  <span class="card-title">1. Config & Dataset</span>
                  <el-button type="danger" link :icon="Refresh" @click="resetStorage">Clear Cache</el-button>
                </div>
              </template>

              <div class="config-bar">
                <div class="config-item">
                  <span class="label">Input Dim (N)</span>
                  <el-input-number v-model="config.inputDim" :min="1" size="small"/>
                </div>
                <div class="config-item">
                  <span class="label">Output Dim (M)</span>
                  <el-input-number v-model="config.outputDim" :min="1" size="small"/>
                </div>
                <div class="config-item">
                  <span class="label">Epochs</span>
                  <el-input-number v-model="config.epochs" :step="50" size="small"/>
                </div>
                <div class="config-item">
                  <span class="label">Learning Rate</span>
                  <el-select v-model="config.lr" size="small" style="width: 100px">
                    <el-option label="0.1" :value="0.1"/>
                    <el-option label="0.01" :value="0.01"/>
                    <el-option label="0.001" :value="0.001"/>
                  </el-select>
                </div>
              </div>

              <el-table :data="dataset" border stripe max-height="300px" size="small">
                <el-table-column label="Input Features (X)">
                  <template #default="{ row }">
                    <div class="data-row-inputs">
                      <el-input-number
                          v-for="(_, i) in row.inputs" :key="i"
                          v-model="row.inputs[i]" :controls="false" class="cell-input"
                      />
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="Target Labels (Y)" width="160">
                  <template #default="{ row }">
                    <div class="data-row-inputs">
                      <el-input-number
                          v-for="(_, i) in row.outputs" :key="i"
                          v-model="row.outputs[i]" :controls="false" class="cell-input y-text"
                      />
                    </div>
                  </template>
                </el-table-column>
                <el-table-column width="50" align="center">
                  <template #default="scope">
                    <el-button link type="danger" :icon="Delete" @click="removeRow(scope.$index)"/>
                  </template>
                </el-table-column>
              </el-table>
              <el-button :icon="Plus" class="add-btn" @click="addRow" size="small">Add Data Sample</el-button>
            </el-card>

            <el-card shadow="never" class="card" style="margin-top: 16px">
              <template #header>
                <div class="card-title">3. Live Inference</div>
              </template>
              <div v-if="model" class="inference-container">
                <div class="predict-grid">
                  <div v-for="(_, i) in predictInputs" :key="i" class="predict-field">
                    <label>x{{ i + 1 }}</label>
                    <el-input-number v-model="predictInputs[i]" @change="handlePredict" size="default"
                                     style="width: 100%"/>
                  </div>
                </div>
                <div class="result-panel">
                  <div class="res-header">Prediction Result:</div>
                  <div class="res-tags">
                    <div v-for="(val, i) in predictResults" :key="i" class="res-item">
                      <span class="y-label">y{{ i + 1 }}</span>
                      <span class="y-value">{{ val }}</span>
                    </div>
                  </div>
                </div>
              </div>
              <el-empty v-else description="Train model to enable prediction" :image-size="40"/>
            </el-card>
          </el-col>

          <el-col :lg="10" :md="24" class="scroll-col">
            <el-card shadow="never" class="card monitor-card">
              <template #header>
                <div class="card-title">2. Training Monitor</div>
              </template>
              <div v-if="isTraining || lossValue > 0" class="monitor-stats">
                <div class="stat-row">
                  <span>Progress: {{ trainProgress }}%</span>
                  <el-progress :percentage="trainProgress" :show-text="false" stroke-width="8"
                               style="flex: 1; margin-left: 10px;"/>
                </div>
                <div class="stat-row loss-row">
                  <span>MSE Loss:</span>
                  <span class="loss-val">{{ lossValue.toFixed(6) }}</span>
                </div>
              </div>
              <div id="viz-container" class="viz-wrapper"></div>
              <div v-if="!isTraining && lossValue === 0" class="viz-placeholder">
                <el-icon size="40" color="#DCDFE6">
                  <Monitor/>
                </el-icon>
                <p>Waiting for training...</p>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>

    <el-dialog v-model="showWeights" title="Neural Network Matrix (Weights & Biases)" width="850px">
      <div v-if="model" class="matrix-viewer">
        <div v-for="(layer, index) in model.layers" :key="index" class="layer-block">
          <div class="layer-title">Layer {{ index + 1 }}: {{ layer.getClassName() }} ({{ layer.name }})</div>
          <div class="matrix-grid-view">
            <div class="matrix-item">
              <p class="matrix-label">Weights (W) - Shape: {{ layer.getWeights()[0]?.shape }}</p>
              <div class="matrix-scroll">
                <table class="m-table">
                  <tr v-for="(row, ri) in getLayerData(layer, 0)" :key="ri">
                    <td v-for="(val, ci) in row" :key="ci" :class="getColorClass(val)">{{ val.toFixed(3) }}</td>
                  </tr>
                </table>
              </div>
            </div>
            <div class="matrix-item" v-if="layer.getWeights()[1]">
              <p class="matrix-label">Bias (b) - Shape: {{ layer.getWeights()[1]?.shape }}</p>
              <div class="matrix-scroll">
                <table class="m-table">
                  <tr>
                    <td v-for="(val, bi) in getLayerData(layer, 1)[0]" :key="bi" :class="getColorClass(val)">
                      {{ val.toFixed(3) }}
                    </td>
                  </tr>
                </table>
              </div>
            </div>
          </div>
          <el-divider v-if="index < model.layers.length - 1"/>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import {ref, reactive, watch, nextTick, shallowRef, onMounted, onBeforeUnmount} from 'vue'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {ElMessage, ElMessageBox} from 'element-plus'
import {Delete, Plus, Monitor, View, Refresh} from '@element-plus/icons-vue'

// --- Storage Key ---
const STORAGE_KEY = 'ai_model_lab_data'

const config = reactive({
  inputDim: 3,
  outputDim: 1,
  epochs: 100,
  lr: 0.01
})

const dataset = ref([
  {inputs: [1, 2, 3], outputs: [6]},
  {inputs: [2, 3, 4], outputs: [9]},
  {inputs: [5, 1, 2], outputs: [8]}
])

const model = shallowRef(null)
const predictInputs = ref([])
const predictResults = ref([])
const isTraining = ref(false)
const trainProgress = ref(0)
const lossValue = ref(0)
const showWeights = ref(false)

let normStats = {inputMin: null, inputMax: null, outputMin: null, outputMax: null}

// --- Initialization & Persistance ---
onMounted(() => {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved) {
    try {
      const parsed = JSON.parse(saved)
      Object.assign(config, parsed.config)
      dataset.value = parsed.dataset
    } catch (e) {
      console.error("Cache load failed", e)
    }
  }
  predictInputs.value = new Array(config.inputDim).fill(0)
})

watch([config, dataset], () => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({config, dataset: dataset.value}))
}, {deep: true})

const resetStorage = () => {
  ElMessageBox.confirm('This will clear all saved data. Proceed?').then(() => {
    localStorage.removeItem(STORAGE_KEY)
    location.reload()
  })
}

// --- Data Management ---
watch(() => config.inputDim, (val) => {
  dataset.value.forEach(row => {
    row.inputs = Array.from({length: val}, (_, i) => row.inputs[i] ?? 0)
  })
  predictInputs.value = new Array(val).fill(0)
})

watch(() => config.outputDim, (val) => {
  dataset.value.forEach(row => {
    row.outputs = Array.from({length: val}, (_, i) => row.outputs[i] ?? 0)
  })
})

const addRow = () => {
  dataset.value.push({
    inputs: new Array(config.inputDim).fill(0),
    outputs: new Array(config.outputDim).fill(0)
  })
}

const removeRow = (index) => {
  dataset.value.splice(index, 1)
}

// --- Matrix Extraction ---
const getLayerData = (layer, index) => {
  const weights = layer.getWeights()[index]
  if (!weights) return []
  const data = weights.arraySync()
  return index === 1 ? [data] : data
}

const getColorClass = (val) => {
  if (val > 0.5) return 'v-pos-high';
  if (val > 0) return 'v-pos'
  if (val < -0.5) return 'v-neg-high';
  return 'v-neg'
}

// --- Training Engine ---
const handleTrain = async () => {
  if (dataset.value.length < 2) return ElMessage.warning("Need at least 2 samples.")

  isTraining.value = true
  trainProgress.value = 0

  if (normStats.inputMin) Object.values(normStats).forEach(t => t?.dispose())

  const {inputs, outputs, stats} = tf.tidy(() => {
    const rawX = tf.tensor2d(dataset.value.map(d => d.inputs))
    const rawY = tf.tensor2d(dataset.value.map(d => d.outputs))
    const iMin = rawX.min(0), iMax = rawX.max(0), oMin = rawY.min(0), oMax = rawY.max(0)
    const eps = tf.scalar(1e-7)
    return {
      inputs: rawX.sub(iMin).div(iMax.sub(iMin).add(eps)),
      outputs: rawY.sub(oMin).div(oMax.sub(oMin).add(eps)),
      stats: {inputMin: tf.keep(iMin), inputMax: tf.keep(iMax), outputMin: tf.keep(oMin), outputMax: tf.keep(oMax)}
    }
  })

  normStats = stats

  const newModel = tf.sequential()
  newModel.add(tf.layers.dense({units: 16, activation: 'relu', inputShape: [config.inputDim]}))
  newModel.add(tf.layers.dense({units: 8, activation: 'relu'}))
  newModel.add(tf.layers.dense({units: config.outputDim}))

  newModel.compile({optimizer: tf.train.adam(config.lr), loss: 'meanSquaredError'})

  try {
    const container = document.getElementById('viz-container')
    await newModel.fit(inputs, outputs, {
      epochs: config.epochs,
      callbacks: [
        tfvis.show.fitCallbacks(container, ['loss'], {height: 180}),
        {
          onEpochEnd: (epoch, logs) => {
            lossValue.value = logs.loss
            trainProgress.value = Math.floor(((epoch + 1) / config.epochs) * 100)
          }
        }
      ]
    })
    model.value = newModel
    await nextTick()
    handlePredict()
    ElMessage.success("Training Complete!")
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    isTraining.value = false
    inputs.dispose();
    outputs.dispose()
  }
}

const handlePredict = () => {
  if (!model.value || !normStats.inputMin) return
  tf.tidy(() => {
    const inputTensor = tf.tensor2d([predictInputs.value])
    const eps = tf.scalar(1e-7)
    const normIn = inputTensor.sub(normStats.inputMin).div(normStats.inputMax.sub(normStats.inputMin).add(eps))
    const prediction = model.value.predict(normIn)
    const unNormRes = prediction.mul(normStats.outputMax.sub(normStats.outputMin).add(eps)).add(normStats.outputMin)
    predictResults.value = Array.from(unNormRes.dataSync()).map(v => v.toFixed(4))
  })
}

onBeforeUnmount(() => {
  if (model.value) model.value.dispose()
  Object.values(normStats).forEach(t => t?.dispose())
})
</script>

<style scoped>
/* Force No Scrollbar on Body */
:global(body) {
  margin: 0;
  overflow: hidden;
}

.app-container {
  background-color: #f8f9fb;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.header {
  background: #fff;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.header-inner {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 10px;
}

.logo {
  font-size: 1.1rem;
  font-weight: bold;
  color: #1e293b;
}

.version {
  font-weight: normal;
  font-size: 0.7em;
  opacity: 0.5;
  margin-left: 5px;
}

/* Content Layout to prevent global scroll */
.content-wrapper {
  flex: 1;
  padding: 16px;
  overflow: hidden; /* Important */
}

.full-height-row {
  height: 100%;
}

.scroll-col {
  height: 100%;
  overflow-y: auto; /* Columns can scroll independently if content is too tall */
  padding-bottom: 20px;
}

/* Scrollbar styling for better look */
.scroll-col::-webkit-scrollbar {
  width: 6px;
}

.scroll-col::-webkit-scrollbar-thumb {
  background: #e2e8f0;
  border-radius: 10px;
}

.card {
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  margin-bottom: 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-weight: 600;
  color: #475569;
  font-size: 0.9rem;
}

.config-bar {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
  background: #f8fafc;
  padding: 12px;
  border-radius: 8px;
  border: 1px solid #f1f5f9;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.label {
  font-size: 10px;
  color: #64748b;
  font-weight: bold;
  text-transform: uppercase;
}

.data-row-inputs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.cell-input {
  width: 64px !important;
}

.y-text :deep(.el-input__inner) {
  color: #3b82f6;
  font-weight: bold;
}

.add-btn {
  margin-top: 10px;
  width: 100%;
  border-style: dashed;
}

.predict-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}

.predict-field label {
  font-size: 11px;
  color: #64748b;
  display: block;
  margin-bottom: 3px;
}

.result-panel {
  background: #f0fdf4;
  padding: 12px;
  border-radius: 8px;
  border: 1px solid #dcfce7;
}

.res-header {
  font-size: 12px;
  color: #166534;
  font-weight: bold;
  margin-bottom: 8px;
}

.res-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.res-item {
  background: #16a34a;
  color: white;
  padding: 4px 10px;
  border-radius: 6px;
  font-family: monospace;
  display: flex;
  gap: 6px;
  font-size: 13px;
}

.y-label {
  opacity: 0.7;
  font-size: 0.8em;
}

.monitor-stats {
  margin-bottom: 12px;
  background: #fff;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.stat-row {
  display: flex;
  align-items: center;
  font-size: 11px;
  margin-bottom: 5px;
}

.loss-row {
  color: #16a34a;
  font-weight: bold;
}

.loss-val {
  margin-left: 8px;
  font-family: monospace;
}

.viz-placeholder {
  height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #cbd5e1;
  gap: 10px;
}

/* Matrix Dialog */
.layer-title {
  font-weight: bold;
  color: #1e293b;
  margin-bottom: 12px;
  font-size: 14px;
}

.matrix-grid-view {
  display: flex;
  gap: 16px;
  overflow-x: auto;
  padding-bottom: 8px;
}

.matrix-item {
  flex: 1;
  min-width: 220px;
}

.matrix-label {
  font-size: 11px;
  color: #94a3b8;
  margin-bottom: 6px;
  font-family: monospace;
}

.matrix-scroll {
  max-height: 250px;
  overflow: auto;
  border: 1px solid #f1f5f9;
  border-radius: 4px;
}

.m-table {
  border-collapse: collapse;
  width: 100%;
  font-family: monospace;
  font-size: 11px;
}

.m-table td {
  border: 1px solid #f1f5f9;
  padding: 4px;
  text-align: center;
}

.v-pos-high {
  background: #dcfce7;
  color: #166534;
}

.v-pos {
  background: #f0fdf4;
  color: #16a34a;
}

.v-neg-high {
  background: #fee2e2;
  color: #991b1b;
}

.v-neg {
  background: #fef2f2;
  color: #ef4444;
}
</style>