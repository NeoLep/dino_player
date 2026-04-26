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
const autoRestart = true

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

let lastObstacle: any
const convertStateToVector = () => {
  const obstalce = window._runner.horizon.obstacles?.[0]
  if (lastObstacle !== obstalce) {
    lastObstacle = obstalce
    console.log(lastObstacle)
  }
  if (!obstalce) return [0, 0, 0]
  let isHighPTE = obstalce.typeConfig.type === 'PTERODACTYL' && obstalce.yPos <= 50
  if (isHighPTE) return [0, 0, 0]

  const speed = window._runner.currentSpeed
  return [obstalce.xPos, obstalce.width, speed]
}


window.addEventListener('dinoRestart', () => {
    renderTryCount(() => tryCount++)
    forward()
})
const handleRunning = async () => {
    if (!window._runner) return
    if (proxy.isCrashed) {
        review() // 复盘一次
        if (autoRestart) {
          proxy.restart()
        }
    }
    if (proxy.isPlaying) {
      const v = convertStateToVector()
      const prediction = model.predictSingle(v) as tf.Tensor<tf.Rank>;
      const result = await prediction.data()
      
      if (result && result[1] > result[0]) {
        proxy.jump()
        lastJumpingState = true 
      } else {
        lastJumpingState = false 
      }
      
      // 必须手动 dispose 预测结果 Tensor，否则会造成内存泄漏
      prediction.dispose();
      
      requestAnimationFrame(() => handleRunning())
    }
}

document.getElementById('tryCount')!.innerText = `${tryCount}`
const MAX_TRAINING_SIZE = 1000;
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

  // 限制训练集大小，避免 O(N^2) 导致的性能下降
  if (training.inputs.length > MAX_TRAINING_SIZE) {
    training.inputs.shift();
    training.labels.shift();
  }

  // 游戏结束了，我们需要将训练数据喂给模型进行训练
  // 减少迭代次数，因为数据量大时，100次迭代太慢了
  // 也可以根据数据量动态调整迭代次数
  const iterations = Math.max(10, Math.min(100, Math.floor(10000 / training.inputs.length)));
  model.fit(training.inputs, training.labels, iterations)
  console.log('reviewed: ', lastJumpingState, input, label[0], tryCount, 'iterations:', iterations)
}