import { tensor, isTensor } from '../tensor';

export default class Model {
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

  export(_modelName: string) {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

  import(_modelName: string) {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

  clear(_modelName: string) {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

}
