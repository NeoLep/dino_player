import './style.css'
import './game/entry'
import { DinoGameProxy, DinoActionCodes } from "./proxy"
import { NNModel, RandomModel } from './ai/models'
import type * as tf from '@tensorflow/tfjs'

function startGame() {
    const runner = window._runner
    const proxy = new DinoGameProxy(runner)

    const model = new NNModel({})
    document.querySelector('#modelName')!.innerHTML = model.modelName
    model.init();

    training = { inputs: [], labels: [] }
    return { runner, proxy, model }
}

const autoStart = false
const autoRestart = true

let tryCount = 0
const lastDinoState = {
  vector: null as null | ReturnType<typeof convertStateToVector>,
  lastJumpingState: false,
}
let training = {
    inputs: [] as any[],
    labels: [] as any[],
}

const { model, runner, proxy } = startGame()
const recordsListEl = document.getElementById('recordsList')

function renderTryCount(cb?: Function) {
    cb?.()
    document.getElementById('gameTitle')!.innerText = `tryCount: ${tryCount}`
}

function formatRecordTime(time: Date) {
  return new Intl.DateTimeFormat('zh-CN', {
    hour12: false,
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(time)
}

function renderRecords() {
  if (!recordsListEl) return

  if (!records.length) {
    recordsListEl.innerHTML = '<div class="record-empty">No Record...</div>'
    return
  }

  const recentRecords = records.slice(-10).reverse()
  recordsListEl.innerHTML = recentRecords.map((record, index) => `
    <div class="record-item">
      <span class="record-rank">#${index + 1}</span>
      <span class="record-time">${formatRecordTime(record.time)}</span>
      <span class="record-score">${record.score}</span>
    </div>
  `).join('')
}

let PlayByAI = true
const forward = () => {
  if (!PlayByAI) return
  requestAnimationFrame(() => window._runner && handleRunning())
}
document.querySelector('#startBtn')?.addEventListener('click', () => {
    if (proxy.isPlaying) return
    if (!proxy.activated) {
        renderTryCount(() => tryCount++)
        proxy.start()
        document.querySelector('#startBtn')!.innerHTML = 'Restart'
    } else {
        proxy.restart()
    }
    forward()
})
if (autoStart) {
    (document.querySelector('#startBtn') as any)?.click()
}

// Toast 提示函数
function showToast(message: string, type: 'success' | 'error' = 'success') {
    const toast = document.getElementById('toast') as HTMLElement
    toast.textContent = message
    toast.className = `toast ${type} show`
    
    setTimeout(() => {
        toast.classList.remove('show')
    }, 2000)
}

document.querySelector('#exportBtn')?.addEventListener('click', () => {
    model.export()
    showToast('模型导出成功！')
})

document.querySelector('#importBtn')?.addEventListener('click', () => {
    const success = model.import()
    if (success) {
        showToast('模型导入成功！')
    } else {
        showToast('未找到模型文件', 'error')
    }
})

// PlayByAI 开关控制
const playByAiSwitch = document.querySelector('#playByAiSwitch') as HTMLInputElement
playByAiSwitch.checked = PlayByAI

const playByAiLabel = document.querySelector('#playByAiLabel') as HTMLElement

playByAiSwitch?.addEventListener('change', (e) => {
    PlayByAI = (e.target as HTMLInputElement).checked
    playByAiLabel.innerText = `AI Play: ${PlayByAI ? 'On' : 'Off'}`
    // 如果关闭AI模式且游戏正在进行，继续运行
})

let lastObstacle: any
const CanvasHeight = runner.dimensions.HEIGHT
const VECTOR_RATIO = 100
const convertStateToVector = () => {
  const obstalce = window._runner.horizon.obstacles?.[0]
  if (!obstalce) return [0, 0, 0, 0]
  if (lastObstacle !== obstalce) {
    lastObstacle = obstalce
  }
  const r = [
      (obstalce.xPos - window._runner.tRex.xPos) / VECTOR_RATIO,      // 障碍物离暴龙的距离
      obstalce.width / VECTOR_RATIO,  // 障碍物宽度
      ((CanvasHeight - obstalce.yPos - obstalce.typeConfig.height - 10) / 10) > 2.5 ? 1 : 0,
      window._runner.currentSpeed / VECTOR_RATIO                    // 当前游戏全局速度
    ];
    return r
}

window.addEventListener('dinoRestart', () => {
    renderTryCount(() => tryCount++)
    forward()
})

const reverseLabel = (needJump?: boolean): [number, number] => needJump ? [0, 1] : [1, 0]
const afterCrashed = () => {
  const MAX_TRAINING_SIZE = 3000;

  if (lastDinoState.lastJumpingState) {
    const inRising = runner.tRex.jumpVelocity < 0
    if (lastDinoState.vector![2] !== 1 && inRising) {
      // 获取最后一次跳跃向量 - 判断是在上升时碰撞，上升时碰撞说明跳跃过晚
      lastDinoState.vector![0] -= (10/VECTOR_RATIO + lastDinoState.vector![3]) // 跳跃前提 - 不同速度下需要前提的距离不同
      training.labels.push(reverseLabel(true))
    } else {
      training.labels.push(reverseLabel(false))
    }
  } else {
    // 总结失败规律
    training.labels.push(reverseLabel(true))
  }
  training.inputs.push(lastDinoState.vector)
  
 

  /** 当达到最大值时，去掉前面的数据 */
  if (training.inputs.length > MAX_TRAINING_SIZE) {
    const diff = training.inputs.length - MAX_TRAINING_SIZE
    training.inputs.splice(0, diff)
    training.labels.splice(0, diff)
  }

  console.log('destory by:', training.inputs[training.inputs.length - 1], lastDinoState.lastJumpingState)
  return [training.inputs, training.labels] as any[]
}

const records: { time: Date, score: string }[] = []
const handleRunning = async () => {
    if (!window._runner) return
    if (proxy.isCrashed) {
      const [inputs, labels] = afterCrashed()
      model.review(inputs, labels)
      console.log(`reviewed - [input length] = ${inputs.length}, [labels length] = ${labels.length}`, lastObstacle)
      records.push({
        time: new Date(),
        score: runner.distanceMeter.getScore(runner.distanceRan)
      })
      renderRecords()
      if (autoRestart) {
        proxy.restart()
      }
    }

    if (proxy.isPlaying) {
      if (!proxy.inJump) {
        const v = convertStateToVector()
        lastDinoState.vector = v
        if (v[2] === 3.5 && !proxy.inDuck) {
          proxy.duck()
        } else {
          const prediction = model.predictSingle(v) as tf.Tensor<tf.Rank>;
          const [isStay, isJump] = await prediction.data()
          lastDinoState.lastJumpingState = isJump > isStay
          ;(lastDinoState.lastJumpingState && proxy.inDuck && proxy.releaseDuck())
          lastDinoState.lastJumpingState && proxy.jump()
        }
      }
      requestAnimationFrame(() => handleRunning())
    }
}

document.getElementById('tryCount')!.innerText = `${tryCount}`
renderRecords()
