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
  if (!obstalce) return [0, 0, 0]
  if (lastObstacle !== obstalce) {
    lastObstacle = obstalce
  }
  const r = [
      (obstalce.xPos - window._runner.tRex.xPos) / 100,      // 障碍物离暴龙的距离
      obstalce.width / 100,  // 障碍物宽度
      window._runner.currentSpeed / 100                    // 当前游戏全局速度
    ];
    return r
}


window.addEventListener('dinoRestart', () => {
    renderTryCount(() => tryCount++)
    forward()
})

const afterCrashed = () => {
  console.log('destory by:', lastObstacle)
  const MAX_TRAINING_SIZE = 1000;
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
  return [training.inputs, training.labels, MAX_TRAINING_SIZE] as any[]
}
const handleRunning = async () => {
    if (!window._runner) return
    if (proxy.isCrashed) {
      const [inputs, labels, MAX_TRAINING_SIZE] = afterCrashed()
      model.review(inputs, labels, MAX_TRAINING_SIZE)
      training.inputs = []
      training.labels = []

        if (autoRestart) {
          proxy.restart()
        }
    }
    if (proxy.isPlaying) {
      const v = convertStateToVector()
      const prediction = model.predictSingle(v) as tf.Tensor<tf.Rank>;
      const [isStay, isJump] = await prediction.data()
      if (isJump > isStay){ 
        proxy.jump()
        lastJumpingState = true
      } else {
        lastJumpingState = false
      }
      
      requestAnimationFrame(() => handleRunning())
    }
}

document.getElementById('tryCount')!.innerText = `${tryCount}`