import Model from './base';

// 随机模型继承自 Model
export default class RandomModel extends Model {
  // weights 和 biases 是 RandomModel 的模型参数
  weights: any = [];
  biases: any = [];

  init() {
    // 初始化就是随机的过程
    this.randomize();
  }

  predict(inputXs: any) {
    // 最简单的线性模型
    const inputX = inputXs[0];
    const y =
      this.weights[0] * inputX[0] +
      this.weights[1] * inputX[1] +
      this.weights[2] * inputX[2] +
      this.biases[0];
    return y < 0 ? 1 : 0;
  }

  /* @ts-ignore */
  train(inputs: any, labels: any) {
    // 随机模型还要啥训练，直接随机！
    this.randomize();
  }

  randomize() {
    // 随机生成所有模型参数
    this.weights[0] = random();
    this.weights[1] = random();
    this.weights[2] = random();
    this.biases[0] = random();
  }
}

function random() {
  return (Math.random() - 0.5) * 2;
}