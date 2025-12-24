<script setup>
import {ref, reactive, watch, nextTick} from 'vue'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import {ElMessage} from 'element-plus'

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

const predictInputs = ref(new Array(config.inputDim).fill(0))
const predictResults = ref([])
const isTraining = ref(false)
const trainProgress = ref(0)
const lossValue = ref(0)

let model = null
let stats = {minX: null, maxX: null, minY: null, maxY: null}

// 2. Dynamic Dimension Logic
watch(() => config.inputDim, (newVal) => {
  dataset.value.forEach(row => {
    while (row.inputs.length < newVal) row.inputs.push(0)
    if (row.inputs.length > newVal) row.inputs.length = newVal
  })
  predictInputs.value = new Array(newVal).fill(0)
})

watch(() => config.outputDim, (newVal) => {
  dataset.value.forEach(row => {
    while (row.outputs.length < newVal) row.outputs.push(0)
    if (row.outputs.length > newVal) row.outputs.length = newVal
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

const normalize = (tensor) => {
  const min = tensor.min(0)
  const max = tensor.max(0)
  const delta = max.sub(min).add(tf.scalar(1e-7))
  return {normalized: tensor.sub(min).div(delta), min, max}
}

// 3. Training Logic with Embedded Visualization
const handleTrain = async () => {
  if (dataset.value.length < 2) return ElMessage.error("Need at least 2 samples to train.")

  isTraining.value = true
  trainProgress.value = 0

  const rawX = dataset.value.map(d => d.inputs)
  const rawY = dataset.value.map(d => d.outputs)

  const xTensor = tf.tensor2d(rawX)
  const yTensor = tf.tensor2d(rawY)

  const normX = normalize(xTensor)
  const normY = normalize(yTensor)
  stats = {minX: normX.min, maxX: normX.max, minY: normY.min, maxY: normY.max}

  model = tf.sequential()
  model.add(tf.layers.dense({
    units: Math.max(32, config.inputDim * 4),
    activation: 'relu',
    inputShape: [config.inputDim]
  }))
  model.add(tf.layers.dropout({rate: 0.1}))
  model.add(tf.layers.dense({units: 16, activation: 'relu'}))
  model.add(tf.layers.dense({units: config.outputDim}))

  model.compile({
    optimizer: tf.train.adam(config.lr),
    loss: 'meanSquaredError',
    metrics: ['mse']
  })

  // Set up tfjs-vis to render in our specific div
  const vizContainer = document.getElementById('viz-container');

  try {
    await model.fit(normX.normalized, normY.normalized, {
      epochs: config.epochs,
      shuffle: true,
      callbacks: [
        // This targets the specific DOM element instead of the default visor
        tfvis.show.fitCallbacks(vizContainer, ['loss', 'mse'], {
          height: 200,
          callbacks: ['onEpochEnd']
        }),
        {
          onEpochEnd: (epoch, logs) => {
            lossValue.value = logs.loss
            trainProgress.value = Math.round(((epoch + 1) / config.epochs) * 100)
          }
        }
      ]
    })

    ElMessage.success("Training Complete!")
    handlePredict()
  } catch (err) {
    console.error(err)
  } finally {
    isTraining.value = false
  }
}

const handlePredict = () => {
  if (!model) return
  tf.tidy(() => {
    let inputTensor = tf.tensor2d([predictInputs.value])
    const normIn = inputTensor.sub(stats.minX).div(stats.maxX.sub(stats.minX).add(1e-7))
    const prediction = model.predict(normIn)
    const denormed = prediction.mul(stats.maxY.sub(stats.minY).add(1e-7)).add(stats.minY)
    predictResults.value = Array.from(denormed.dataSync()).map(v => v.toFixed(4))
  })
}
</script>

<template>
  <div class="app-wrapper">
    <el-container class="main-container">
      <el-header height="60px" class="main-header">
        <h1>AI General Function Approximator</h1>
        <el-button type="primary" @click="handleTrain" :loading="isTraining" size="large">
          RUN TRAINING
        </el-button>
      </el-header>

      <el-main>
        <el-row :gutter="20">
          <el-col :md="14" :sm="24">
            <el-card shadow="hover" class="card-section">
              <template #header><b>1. Configuration & Dataset</b></template>

              <div class="config-grid">
                <div class="config-item">
                  <label>Input Dim (N)</label>
                  <el-input-number v-model="config.inputDim" :min="1" :max="10" size="small"/>
                </div>
                <div class="config-item">
                  <label>Output Dim (M)</label>
                  <el-input-number v-model="config.outputDim" :min="1" :max="5" size="small"/>
                </div>
                <div class="config-item">
                  <label>Epochs</label>
                  <el-input-number v-model="config.epochs" :step="50" size="small"/>
                </div>
                <div class="config-item">
                  <label>Learning Rate</label>
                  <el-select v-model="config.lr" size="small" style="width: 100px">
                    <el-option label="0.1" :value="0.1"/>
                    <el-option label="0.01" :value="0.01"/>
                    <el-option label="0.001" :value="0.001"/>
                  </el-select>
                </div>
              </div>

              <el-table :data="dataset" border stripe size="small" class="data-table">
                <el-table-column label="Inputs (X)" min-width="180">
                  <template #default="scope">
                    <div class="input-cell-group">
                      <el-input-number v-for="(n, i) in scope.row.inputs" :key="i"
                                       v-model="scope.row.inputs[i]" :controls="false" class="nano-input"/>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="Outputs (Y)" min-width="120">
                  <template #default="scope">
                    <div class="input-cell-group">
                      <el-input-number v-for="(n, i) in scope.row.outputs" :key="i"
                                       v-model="scope.row.outputs[i]" :controls="false"
                                       class="nano-input output-style"/>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column width="70" align="center">
                  <template #default="scope">
                    <el-button link type="danger" icon="Delete" @click="removeRow(scope.$index)"/>
                  </template>
                </el-table-column>
              </el-table>
              <el-button icon="Plus" size="small" @click="addRow" style="margin-top: 10px">Add Sample</el-button>
            </el-card>

            <el-card shadow="hover" class="card-section" style="margin-top: 20px">
              <template #header><b>3. Live Inference</b></template>
              <div v-if="model">
                <div class="predict-input-grid">
                  <div v-for="(v, i) in predictInputs" :key="i" class="p-field">
                    <span>x{{ i + 1 }}</span>
                    <el-input-number v-model="predictInputs[i]" @change="handlePredict" size="default"/>
                  </div>
                </div>
                <div class="result-area">
                  <span class="res-label">Predicted Output:</span>
                  <div class="res-tags">
                    <el-tag v-for="(res, i) in predictResults" :key="i" type="success" effect="dark" size="large">
                      y{{ i + 1 }}: {{ res }}
                    </el-tag>
                  </div>
                </div>
              </div>
              <el-empty v-else description="Train the model first to enable prediction" :image-size="40"/>
            </el-card>
          </el-col>

          <el-col :md="10" :sm="24">
            <el-card shadow="hover" class="card-section sticky-viz">
              <template #header><b>2. Training Monitor (Real-time)</b></template>
              <div class="status-bar" v-if="trainProgress > 0">
                <span>Progress: {{ trainProgress }}%</span>
                <span>Loss: {{ lossValue.toFixed(6) }}</span>
              </div>
              <div id="viz-container" class="viz-panel"></div>
              <div v-if="trainProgress === 0" class="viz-placeholder">
                Charts will appear here during training...
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
  </div>
</template>

<style scoped>
.app-wrapper {
  background-color: #f4f6f8;
  min-height: 100vh;
}

.main-header {
  background: #fff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #dcdfe6;
  padding: 0 40px;
}

.main-header h1 {
  font-size: 1.2rem;
  color: #303133;
  margin: 0;
}

.card-section {
  border-radius: 8px;
}

.config-grid {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.config-item label {
  font-size: 12px;
  color: #909399;
  font-weight: bold;
}

.input-cell-group {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.nano-input {
  width: 75px !important;
}

.status-bar {
  display: flex;
  justify-content: space-between;
  margin-bottom: 15px;
  font-family: monospace;
  font-size: 13px;
  color: #67c23a;
  background: #f0f9eb;
  padding: 8px;
  border-radius: 4px;
}

.viz-panel {
  min-height: 400px;
  width: 100%;
}

.viz-placeholder {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
  border: 2px dashed #ebeef5;
  border-radius: 8px;
}

.predict-input-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.p-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.p-field span {
  font-size: 11px;
  color: #909399;
}

.result-area {
  padding: 15px;
  background: #fdf6ec;
  border-radius: 6px;
}

.res-label {
  display: block;
  margin-bottom: 10px;
  font-weight: bold;
  color: #e6a23c;
}

.res-tags {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.sticky-viz {
  position: sticky;
  top: 20px;
}
</style>