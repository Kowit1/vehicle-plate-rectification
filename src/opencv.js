import cvModule from '@techstark/opencv-js'

let cached

export async function loadOpenCv() {
  if (cached) return cached
  if (cvModule instanceof Promise) {
    cached = await cvModule
    return cached
  }
  if (cvModule.Mat) {
    cached = cvModule
    return cached
  }
  cached = await new Promise((resolve) => {
    cvModule.onRuntimeInitialized = () => resolve(cvModule)
  })
  return cached
}
