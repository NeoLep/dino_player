import { tensor } from '../tensor';

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
  train(inputXs: any[], inputYs: any[]) {
    throw new Error(
      'Abstract method must be implemented in the derived class.'
    );
  }

  fit(inputXs: any, inputYs: any, iterationCount = 100) {
    for (let i = 0; i < iterationCount; i += 1) {
      this.train(inputXs, inputYs);
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
