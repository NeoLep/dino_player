import Runner from "../game/Runner"

export enum DinoActionCodes {
  JUMP = 38,
  DUCK = 40,
  RESTART = 13,
}

export class DinoGameProxy {
  runner: Runner
  constructor(runner: Runner) {
    this.runner = runner
  }
  get activated() {
    return this.runner.activated
  }
  get isPlaying() {
    return this.runner.playing
  }
  get isCrashed() {
    return this.runner.crashed
  }
  get inDuck() {
    return this.runner.tRex.ducking
  }
  get inJump() {
    return this.runner.tRex.jumping
  }
  get state() {
    return {
      status: this.runner.tRex.status,
      speed: this.runner.currentSpeed,
      trex: this.runner.tRex,
      obstacles: this.runner.horizon.obstacles,
      time: this.runner.time
    }
  }
  action(act: DinoActionCodes) {
    switch (act) {
      case DinoActionCodes.JUMP:
        this.runner.onKeyDown(new KeyboardEvent('keydown', {
          keyCode: DinoActionCodes.JUMP
        }))
        break
      case DinoActionCodes.DUCK:
        this.runner.onKeyDown(new KeyboardEvent('keydown', {
          keyCode: DinoActionCodes.DUCK
        }))
        break
      case DinoActionCodes.RESTART:
        this.runner.restart()
        break
    }
  }
  duck() {
    this.action(DinoActionCodes.DUCK)
  }
  jump() {
    this.action(DinoActionCodes.JUMP)
  }
  start() {
    this.action(DinoActionCodes.JUMP)
  }
  restart() {
    this.action(DinoActionCodes.RESTART)
  }
  needJumping(): -1 | 0 | 1 {
    return 1
  }
}
