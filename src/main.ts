import './style.css'
import './game/entry'
import { DinoGameProxy, DinoActionCodes } from "./proxy"
import { NNModel } from './ai/models'
import type * as tf from '@tensorflow/tfjs'

function startGame() {
    const runner = window._runner
    const proxy = new DinoGameProxy(runner)

    const model = new NNModel({})
    model.init();

    training = { inputs: [], labels: [] }
    return { runner, proxy, model }
}

const autoStart = false
const autoRestart = false

let tryCount = 0
let lastJumpingState = false
let training = {
    inputs: [] as any[],
    labels: [] as any[],
}

const { model, runner, proxy } = startGame()

function renderTryCount(cb?: Function) {
    cb?.()
    document.getElementById('gameTitle')!.innerText = `tryCount: ${tryCount}`
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

const convertStateToVector = () => {
  const obstalce = window._runner.horizon.obstacles?.[0]
  if (!obstalce) return [0, 0, 0]
  let isHighPTE = obstalce.typeConfig.type === 'PTERODACTYL' && obstalce.yPos <= 50
  if (isHighPTE) return [0, 0, 0]

  const speed = window._runner.currentSpeed
  return [obstalce.xPos, obstalce.yPos, speed]
}


window.addEventListener('dinoRestart', () => {
    renderTryCount(() => tryCount++)
    forward()
})
const handleRunning = async () => {
    if (!window._runner) return
    if (proxy.isCrashed) {
        review() // 复盘一次
        proxy.restart()
    }
    if (proxy.isPlaying) {
      const v = convertStateToVector()
      const prediction = model.predictSingle(v) as tf.Tensor<tf.Rank>;
      const result = await prediction.data()
      if (result && result[1] > result[0]) {
        proxy.jump()
        lastJumpingState = true // 记录崩溃前最后的状态
      } else {
        lastJumpingState = false // 记录崩溃前最后的状态
      }
      requestAnimationFrame(() => handleRunning())
    }
}

document.getElementById('tryCount')!.innerText = `${tryCount}`
const review = () => {
  let input = null;
  let label = null;
  input = convertStateToVector();

  if (lastJumpingState) {
    label = [1, 0]; // 跳错了，应该保持不变！下次记住了！
  } else {
    label = [0, 1]; // 不应该保守的，应该跳跃才对！下次记住了！
  }

  training.inputs.push(input);
  training.labels.push(label);

  // 游戏结束了，我们需要将训练数据喂给模型进行训练
  model.fit(training.inputs, training.labels)
  console.log('reviewed: ', lastJumpingState, input, label[0], tryCount)
}