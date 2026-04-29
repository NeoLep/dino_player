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

const forward = () => requestAnimationFrame(() => window._runner && handleRunning())
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

document.querySelector('#exportBtn')?.addEventListener('click', () => {
    model.export()
})

document.querySelector('#importBtn')?.addEventListener('click', () => {
    model.import()
})

let lastObstacle: any
const CanvasHeight = runner.dimensions.HEIGHT
const convertStateToVector = () => {
  const obstalce = window._runner.horizon.obstacles?.[0]
  if (!obstalce) return [0, 0, 0, 0]
  if (lastObstacle !== obstalce) {
    lastObstacle = obstalce
  }
  const r = [
      (obstalce.xPos - window._runner.tRex.xPos) / 100,      // 障碍物离暴龙的距离
      obstalce.width / 100,  // 障碍物宽度
      ((CanvasHeight - obstalce.yPos - obstalce.typeConfig.height - 10) / 10) > 0.46 ? 1 : 0,
      window._runner.currentSpeed / 100                    // 当前游戏全局速度
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
  const newLabel = reverseLabel(!lastDinoState.lastJumpingState); // 将最后一次决策反转
  
  // 总结失败规律
  training.inputs.push(lastDinoState.vector)
  training.labels.push(newLabel)

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
      console.log(`reviewed - [input length] = ${inputs.length}, [labels length] = ${labels.length}`)
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
