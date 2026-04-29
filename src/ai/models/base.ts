import * as tf from '@tensorflow/tfjs'

import { tensor, isTensor } from '../tensor';

export default class Model {
  public modelName: string = 'model'
  public hiddenLayerSize: number = 0
  public inputSize: number = 0
  public outputSize: number = 0
  
  public weights: any[] = []
  public biases: any[] = []

  init() {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

  /* @ts-ignore */
  predict(inputXs: number[]): unknown {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

  predictSingle(inputX: any) {
    return this.predict([inputX]);
  }

  /* @ts-ignore */
  train(inputXs: any, inputYs: any) {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

  fit(inputXs: any, inputYs: any, iterationCount = 100) {
    const x = tensor(inputXs);
    const y = tensor(inputYs);

    for (let i = 0; i < iterationCount; i += 1) {
      this.train(x, y);
    }

    // 只有在输入不是 Tensor 的情况下才由 fit 负责 dispose
    // 这样如果外部传入 Tensor，外部可以继续复用
    if (!isTensor(inputXs)) {
      x.dispose();
    }
    if (!isTensor(inputYs)) {
      y.dispose();
    }
  }

  loss(predictedYs: any, labels: any) {
    const meanSquareError = predictedYs
      .sub(tensor(labels))
      .square()
      .mean();
    return meanSquareError;
  }

  review(inputs: any[], labels: any[], maxTrainingSize = 1000) {
    // 限制训练集大小，避免 O(N^2) 导致的性能下降
    if (inputs.length > maxTrainingSize) {
      inputs.shift();
      labels.shift();
    }
    const iterations = Math.max(10, Math.min(100, Math.floor(10000 / inputs.length)));
    this.fit(inputs, labels, iterations)
  }

  /**
   * Export model weights and biases to localStorage
   */
  export(): void {
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
    localStorage.setItem(this.modelName, JSON.stringify(modelData));
    console.log(`Model exported to localStorage as "${this.modelName}"`);
  }

  /**
   * Import model weights and biases from localStorage
   * @returns boolean indicating success
   */
  import(): boolean {
    try {
      const stored = localStorage.getItem(this.modelName);
      if (!stored) {
        console.warn(`No model found in localStorage with key "${this.modelName}"`);
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
      
      console.log(`Model imported from localStorage "${this.modelName}"`);
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
