// Tensor-unit matmul throughput by operand type: swiftc -O main.swift -o mmbench && ./mmbench kern.metal
import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let src = try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let opts = MTLCompileOptions(); opts.languageVersion = .version4_0
let lib = try! dev.makeLibrary(source: src, options: opts)
let q = dev.makeCommandQueue()!
let names = ["bf16_bf16_f32", "i8_i8_i32", "bf16_i8_f32", "bf16_i4_f32", "i8_i4_i32"]
let iters: Int32 = 512, groups = 2048, tgThreads = 128
let out = dev.makeBuffer(length: groups * tgThreads * 4, options: .storageModeShared)!
let raw = dev.makeBuffer(length: 262144, options: .storageModeShared)!
memset(raw.contents(), 0x11, 262144)
for n in names {
  let fc = MTLFunctionConstantValues(); var it = iters
  fc.setConstantValue(&it, type: .int, index: 0)
  guard let fn = try? lib.makeFunction(name: n, constantValues: fc), let p = try? dev.makeComputePipelineState(function: fn) else { print("\(n): not available"); continue }
  var best = 1e9
  for _ in 0..<4 {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0); e.setBuffer(raw, offset: 0, index: 1)
    e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tgThreads, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let ops = Double(groups * tgThreads / 32) * Double(iters) * 2 * 16 * 32 * 512
  print(String(format: "%-16@ %8.1f T(FL)OPS  (%.2f ms)", n as NSString, ops / best / 1e12, best * 1e3))
}
