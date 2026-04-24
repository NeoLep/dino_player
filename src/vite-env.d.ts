/// <reference types="vite/client" />

import type Runner from "./game/Runner";

declare global {
  interface Window {
    _runner: Runner
  }
}