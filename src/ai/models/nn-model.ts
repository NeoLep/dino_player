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
  optimizer: tf.AdamOptimizer
  hiddenLayerSize: number
  inputSize: number
  outputSize: number

  weights: tf.Variable[] = []
  biases: tf.Variable[] = []

  constructor({ inputSize = 3, hiddenLayerSize = inputSize * 2, outputSize = 2, learningRate = 0.1 }: ConsArgType) {
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

  predict(inputXs: number[]) {
    const x = tensor(inputXs);
    // 预测的是指
    const prediction = tf.tidy(() => {
      const hiddenLayer = tf.sigmoid(x.matMul(this.weights[0]).add(this.biases[0]));
      const outputLayer = tf.sigmoid(hiddenLayer.matMul(this.weights[1]).add(this.biases[1]));
      return outputLayer;
    });
    return prediction;
  }

  train(inputXs: any[], inputYs: any[]): void {
    // 训练的过程其实就是将带标签的数据交给内置的 optimizer 进行优化
    this.optimizer.minimize(() => {
      const predictedYs = this.predict(inputXs);
      // 计算损失值，优化器的目标就是最小化该值
      return this.loss(predictedYs, inputYs);
    });
  }

  /**
   * Export model weights and biases to localStorage
   */
  export(modelName: string = 'nn-model'): void {
    const modelData: Record<string, number[]> = {};
    
    // Extract weights
    this.weights.forEach((weight, index) => {
      modelData[`weight_${index}`] = Array.from(weight.dataSync());
    });
    
    // Extract biases
    this.biases.forEach((bias, index) => {
      modelData[`bias_${index}`] = Array.from(bias.dataSync());
    });
    
    // Save metadata
    modelData['_metadata'] = [
      this.inputSize,
      this.hiddenLayerSize,
      this.outputSize,
      Date.now()
    ];
    
    // Store in localStorage
    localStorage.setItem(modelName, JSON.stringify(modelData));
    console.log(`Model exported to localStorage as "${modelName}"`);
  }

  /**
   * Import model weights and biases from localStorage
   * @returns boolean indicating success
   */
  import(modelName: string = 'nn-model'): boolean {
    try {
      const stored = localStorage.getItem(modelName);
      if (!stored) {
        console.warn(`No model found in localStorage with key "${modelName}"`);
        return false;
      }
      
      const modelData: Record<string, any> = JSON.parse(stored);
      
      // Validate architecture matches (optional but recommended)
      const metadata = modelData['_metadata'];
      if (metadata) {
        const [inputSize, hiddenLayerSize, outputSize, timestamp] = metadata;
        if (inputSize !== this.inputSize || 
            hiddenLayerSize !== this.hiddenLayerSize || 
            outputSize !== this.outputSize) {
          console.warn('Model architecture mismatch! Importing anyway, but predictions may be incorrect.');
        }
      }
      
      // Restore weights
      this.weights.forEach((weight, index) => {
        const key = `weight_${index}`;
        if (modelData[key]) {
          const newValues = tf.tensor(modelData[key], weight.shape);
          weight.assign(newValues);
          newValues.dispose(); // Clean up temporary tensor
        }
      });
      
      // Restore biases
      this.biases.forEach((bias, index) => {
        const key = `bias_${index}`;
        if (modelData[key]) {
          const newValues = tf.tensor(modelData[key], bias.shape);
          bias.assign(newValues);
          newValues.dispose(); // Clean up temporary tensor
        }
      });
      
      console.log(`Model imported from localStorage "${modelName}"`);
      return true;
    } catch (error) {
      console.error('Failed to import model:', error);
      return false;
    }
  }

  /**
   * Optional: Clear saved model from localStorage
   */
  clear(modelName: string = 'nn-model'): void {
    localStorage.removeItem(modelName);
    console.log(`Model "${modelName}" cleared from localStorage`);
  }

  /**
   * Optional: Check if a model exists in localStorage
   */
  exists(modelName: string = 'nn-model'): boolean {
    return localStorage.getItem(modelName) !== null;
  }
}