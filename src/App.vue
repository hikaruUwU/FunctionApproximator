<script setup>
import {ref, reactive, computed, watch} from 'vue'
import * as tf from '@tensorflow/tfjs'
import {ElMessage} from 'element-plus'

const config = reactive({
  inputDim: 3,
  outputDim: 1,
  epochs: 50,
  lr: 0.01
})

const dataset = ref([
  {inputs: [0, 0, 0], outputs: [0]}
])

const predictInputs = ref(new Array(config.inputDim).fill(0))
const predictResults = ref([])
const isTraining = ref(false)
const trainProgress = ref(0)
const lossValue = ref(0)

let model = null
let stats = {min: null, max: null}

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
  return {
    normalized: tensor.sub(min).div(max.sub(min).add(tf.scalar(1e-7))),
    min, max
  }
}

const handleTrain = async () => {
  if (dataset.value.length < 2) return ElMessage.error("请至少输入2组数据进行训练")

  isTraining.value = true
  trainProgress.value = 0

  const rawX = dataset.value.map(d => d.inputs)
  const rawY = dataset.value.map(d => d.outputs)

  const xTensor = tf.tensor2d(rawX, [rawX.length, config.inputDim])
  const yTensor = tf.tensor2d(rawY, [rawY.length, config.outputDim])

  const normX = normalize(xTensor)
  const normY = normalize(yTensor)
  stats.minX = normX.min;
  stats.maxX = normX.max
  stats.minY = normY.min;
  stats.maxY = normY.max

  model = tf.sequential()
  model.add(tf.layers.dense({units: 32, activation: 'relu', inputShape: [config.inputDim]}))
  model.add(tf.layers.dense({units: 16, activation: 'relu'}))
  model.add(tf.layers.dense({units: config.outputDim}))

  model.compile({optimizer: tf.train.adam(config.lr), loss: 'meanSquaredError'})

  await model.fit(normX.normalized, normY.normalized, {
    epochs: config.epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        lossValue.value = logs.loss
        trainProgress.value = Math.round(((epoch + 1) / config.epochs) * 100)
      }
    }
  })

  ElMessage.success("推理完成")
  isTraining.value = false
  handlePredict()
}

const handlePredict = () => {
  if (!model) return
  tf.tidy(() => {
    let inputTensor = tf.tensor2d([predictInputs.value])

    inputTensor = inputTensor.sub(stats.minX).div(stats.maxX.sub(stats.minX).add(1e-7))

    const prediction = model.predict(inputTensor)

    const denormed = prediction.mul(stats.maxY.sub(stats.minY).add(1e-7)).add(stats.minY)
    predictResults.value = Array.from(denormed.dataSync()).map(v => v.toFixed(4))
  })
}
</script>

<template>
  <div class="container">
    <el-card class="box-card">
      <template #header>
        <div class="card-header">
          <span>FunctionApproximator</span>
        </div>
      </template>

      <el-row :gutter="20" class="config-row">
        <el-col :span="6">
          <label>Input Dimension(s) (N):</label>
          <el-input-number v-model="config.inputDim" :min="1" :max="10" size="small"/>
        </el-col>
        <el-col :span="6">
          <label>Output Dimension(s) (M):</label>
          <el-input-number v-model="config.outputDim" :min="1" :max="5" size="small"/>
        </el-col>
        <el-col :span="12" style="text-align: right">
          <el-button type="primary" @click="handleTrain" :loading="isTraining">START TRAINING</el-button>
        </el-col>
      </el-row>

      <el-divider>Start</el-divider>

      <el-table :data="dataset" style="width: 100%" max-height="400">
        <el-table-column label="输入因子 (Inputs)" min-width="250">
          <template #default="scope">
            <div class="cell-inputs">
              <el-input-number v-for="(v, i) in scope.row.inputs" :key="i"
                               v-model="scope.row.inputs[i]" size="small" class="m-1" :controls="false"/>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="输出因子 (Outputs)" min-width="150">
          <template #default="scope">
            <div class="cell-inputs">
              <el-input-number v-for="(v, i) in scope.row.outputs" :key="i"
                               v-model="scope.row.outputs[i]" size="small" class="m-1" :controls="false"/>
            </div>
          </template>
        </el-table-column>
        <el-table-column fixed="right" label="操作" width="80">
          <template #default="scope">
            <el-button link type="danger" @click="removeRow(scope.$index)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div style="margin-top: 10px">
        <el-button size="small" @click="addRow">ADD</el-button>
      </div>

      <div v-if="isTraining || trainProgress > 0" class="progress-box">
        <span>Progress: {{ trainProgress }}% (Loss: {{ lossValue.toFixed(6) }})</span>
        <el-progress :percentage="trainProgress" :status="trainProgress === 100 ? 'success' : ''"/>
      </div>

      <el-divider>Result</el-divider>

      <div v-if="model" class="predict-box">
        <p>Input#</p>
        <div class="flex-wrap">
          <div v-for="(v, i) in predictInputs" :key="i" class="p-input">
            <span class="p-label">x{{ i + 1 }}</span>
            <el-input-number v-model="predictInputs[i]" size="default" @change="handlePredict"/>
          </div>
        </div>

        <div class="result-display">
          <h3>Output#</h3>
          <div class="tags">
            <el-tag v-for="(res, i) in predictResults" :key="i" size="large" type="success" effect="dark">
              Output {{ i + 1 }}: {{ res }}
            </el-tag>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.container {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.box-card {
  max-width: 1000px;
  margin: 0 auto;
}

.config-row {
  margin-bottom: 20px;
  align-items: center;
}

.config-row label {
  font-size: 14px;
  margin-right: 10px;
  font-weight: bold;
}

.cell-inputs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.m-1 {
  width: 70px !important;
}

.progress-box {
  margin-top: 20px;
  padding: 15px;
  background: #fafafa;
  border-radius: 8px;
}

.predict-box {
  background: #f0f9eb;
  padding: 20px;
  border-radius: 8px;
}

.flex-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin-bottom: 20px;
}

.p-input {
  display: flex;
  flex-direction: column;
}

.p-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
}

.result-display {
  margin-top: 10px;
}

.tags {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}
</style>