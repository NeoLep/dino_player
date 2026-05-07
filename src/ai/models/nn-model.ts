import * as tf from '@tensorflow/tfjs'
import { tensor } from '../tensor'
import Model from './base'

type ConsArgType = {
  inputSize?: number,
  hiddenLayerSize?: number,
  outputSize?: number,
  learningRate?: number
}

export default class NNModel extends Model {
  modelName: string = 'nn-model'
  optimizer: tf.AdamOptimizer
  hiddenLayerSize: number
  inputSize: number
  outputSize: number

  constructor({ inputSize = 4, hiddenLayerSize = inputSize * 2, outputSize = 3, learningRate = 0.01 }: ConsArgType) {
    super()

    this.hiddenLayerSize = hiddenLayerSize;
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    // 我们使用 ADAM 作为优化器
    this.optimizer = tf.train.adam(learningRate);
  }

  init() {
    // 隐藏层
    this.weights[0] = tf.variable(
      tf.randomNormal([this.inputSize, this.hiddenLayerSize])
    );
    this.biases[0] = tf.variable(tf.scalar(Math.random()));
    // 输出层tput layer
    this.weights[1] = tf.variable(
      tf.randomNormal([this.hiddenLayerSize, this.outputSize])
    );
    this.biases[1] = tf.variable(tf.scalar(Math.random()));
  }

  predict(inputXs: number[] | tf.Tensor) {
    const x = tensor(inputXs);
    // 预测的是值
    const prediction = tf.tidy(() => {
      const hiddenLayer = tf.sigmoid(x.matMul(this.weights[0]).add(this.biases[0]));
      const outputLayer = tf.sigmoid(hiddenLayer.matMul(this.weights[1]).add(this.biases[1]));
      return outputLayer;
    });

    return prediction;
  }

  train(inputXs: tf.Tensor, inputYs: tf.Tensor): void {
    // 训练的过程其实就是将带标签的数据交给内置的 optimizer 进行优化
    this.optimizer.minimize(() => {
      const predictedYs = this.predict(inputXs);
      // 计算损失值，优化器的目标就是最小化该值
      return this.loss(predictedYs, inputYs);
    });
  }
}