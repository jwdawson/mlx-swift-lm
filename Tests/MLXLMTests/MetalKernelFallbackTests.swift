// Copyright © 2026 Apple Inc.
//
// Pins each custom Metal kernel's plain-ops fallback (the path builds without Metal,
// such as the Linux CPU build, take) to the kernel on the same inputs, so the two
// cannot drift apart. Runs only where the kernels exist.

#if canImport(Metal)

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class MetalKernelFallbackTests: XCTestCase {

    private func assertBitIdentical(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, "\(label): dtype", file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape", file: file, line: line)
        let a = actual.asType(.float32).asArray(Float.self)
        let b = expected.asType(.float32).asArray(Float.self)
        let mismatches = zip(a, b).filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(label): \(mismatches)/\(a.count) elements differ bitwise",
            file: file, line: line)
    }

    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: - SwitchGLU sorted reduction (exact, as DirectExpertReductionTests)

    func testWeightedExpertUnsortOpsMatchesKernelExactly() {
        MLXRandom.seed(17)
        for (tokens, hidden) in [(8, 64), (17, 128), (257, 256)] {
            let assignmentCount = tokens * 8
            let expertIndices = MLXArray(
                (0 ..< assignmentCount).map { UInt32(($0 * 13 + $0 / 7) % 29) })
            let order = argSort(expertIndices)
            let inverseOrder = argSort(order)
            let outputs = MLXRandom.normal([tokens, 8, hidden]).asType(.bfloat16)
            let weights = softmax(MLXRandom.normal([tokens, 8]), axis: -1).asType(.bfloat16)
            let sortedOutputs = outputs.reshaped(assignmentCount, hidden)[order]

            let kernel = weightedExpertUnsort(
                sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, weights: weights)
            let ops = weightedExpertUnsortOps(
                sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, weights: weights)
            eval(kernel, ops)
            assertBitIdentical(ops, kernel, "tokens=\(tokens), hidden=\(hidden)")
        }
    }

    // MARK: - Fused router top-k (bitwise, as Qwen35RouterTopKBitwiseTests)

    func testRouterTopKOpsMatchesKernelBitwise() {
        let rows = 1
        for (e, k) in [(256, 8), (128, 8), (64, 4)] {
            for dtype in [DType.float16, DType.bfloat16, DType.float32] {
                for order in [FusedRouterTopKOrder.ascending, .descending] {
                    for normalize in [false, true] {
                        withRandomState(MLXRandom.RandomState(seed: UInt64(e + k))) {
                            let soft = MLX.softmax(
                                MLXRandom.normal([rows, e]), axis: -1, precise: true
                            ).asType(dtype)
                            let ties = (MLX.round(MLXRandom.normal([rows, e]) * 2) / 2)
                                .asType(dtype)
                            for (name, selection, values) in [
                                ("softmax", soft, soft), ("ties", ties, soft),
                            ] {
                                let label =
                                    "\(name) E=\(e) K=\(k) \(dtype) \(order) norm=\(normalize)"
                                let kernel = fusedRouterTopK(
                                    selection: selection, values: values, k: k,
                                    normalize: normalize, order: order)
                                let ops = routerTopKOps(
                                    selection: selection, values: values, k: k,
                                    normalize: normalize, order: order)
                                eval(kernel.indices, kernel.scores, ops.indices, ops.scores)
                                XCTAssertEqual(ops.indices.dtype, kernel.indices.dtype, label)
                                XCTAssertEqual(
                                    ops.indices.reshaped(-1).asArray(UInt32.self),
                                    kernel.indices.reshaped(-1).asArray(UInt32.self),
                                    "\(label): expert order")
                                assertBitIdentical(ops.scores, kernel.scores, "\(label): scores")
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Gated delta (GatedDeltaTests' tolerance: relative drift < 1e-4)

    func testGatedDeltaOpsAsKernelMatchesTheKernelMaskedAndUnmasked() {
        withRandomState(MLXRandom.RandomState(seed: 53)) {
            let (b, t, hk, dk, hv, dv) = (2, 24, 2, 64, 4, 16)
            let q = MLXRandom.normal([b, t, hk, dk]).asType(.bfloat16)
            let k = MLXRandom.normal([b, t, hk, dk]).asType(.bfloat16)
            let v = MLXRandom.normal([b, t, hv, dv]).asType(.bfloat16)
            let a = MLXRandom.normal([b, t, hv]).asType(.bfloat16)
            let gate = MLXRandom.normal([b, t, hv]).asType(.bfloat16)
            let aLog = (MLXRandom.normal([hv]) * 0.1).asType(.bfloat16)
            let dtBias = MLXRandom.normal([hv]).asType(.bfloat16)
            let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
            let beta = sigmoid(gate).asType(.float32)
            let state = MLXArray.zeros([b, hv, dv, dk], dtype: .float32)
            // Left padding on the first sequence, a hole in the second.
            let mask = MLXArray(
                (0 ..< b * t).map { i in i < 5 || (i >= t + 9 && i < t + 12) ? false : true }
            )
            .reshaped(b, t)

            for (name, m) in [("unmasked", nil), ("masked", mask)] as [(String, MLXArray?)] {
                let (yKernel, stateKernel) = gatedDeltaKernel(
                    q: q, k: k, v: v, g: g, beta: beta, state: state, mask: m)
                let (yOps, stateOps) = gatedDeltaOpsAsKernel(
                    q: q, k: k, v: v, g: g, beta: beta, state: state, mask: m)
                eval(yKernel, stateKernel, yOps, stateOps)

                func relativeDrift(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
                    let scale = max(
                        abs(lhs.asType(.float32)).max().item(Float.self),
                        abs(rhs.asType(.float32)).max().item(Float.self))
                    let drift = maxAbsDifference(lhs, rhs)
                    return scale > 0 ? drift / scale : drift
                }
                print(
                    "gated delta \(name): output drift \(relativeDrift(yOps, yKernel)), "
                        + "state drift \(relativeDrift(stateOps, stateKernel))")
                XCTAssertEqual(yOps.dtype, yKernel.dtype, name)
                XCTAssertLessThan(relativeDrift(yOps, yKernel), 1e-4, "\(name): output")
                XCTAssertLessThan(relativeDrift(stateOps, stateKernel), 1e-4, "\(name): state")
            }
        }
    }

    // MARK: - SSM decode step (SSMTests' tolerance: rtol 1e-4, atol 1e-5)

    func testSSMAttnMatchesTheDecodeKernel() {
        withRandomState(MLXRandom.RandomState(seed: 31)) {
            let (batch, heads, headDimension, groups, stateDimension) = (2, 8, 16, 2, 64)
            let x = MLXRandom.normal([batch, 1, heads, headDimension])
            let aLog = MLXRandom.normal([heads]) * 0.1
            let b = MLXRandom.normal([batch, 1, groups, stateDimension]) * 0.2
            let c = MLXRandom.normal([batch, 1, groups, stateDimension]) * 0.2
            let d = MLXRandom.normal([heads]) * 0.2
            let dt = MLXRandom.normal([batch, 1, heads]) * 0.1
            let dtBias = MLXRandom.normal([heads]) * 0.1
            let state = MLXRandom.normal([batch, heads, headDimension, stateDimension]) * 0.2

            // seqLen 1 with a state takes the kernel; ssmAttn is the path without Metal.
            let (kernelY, kernelState) = ssmUpdate(
                hiddenStates: x, ALog: aLog, B: b, C: c, D: d, dt: dt, dtBias: dtBias,
                state: state)
            let (opsY, opsState) = ssmAttn(
                x: x, ALog: aLog, B: b, C: c, D: d, dt: dt, dtBias: dtBias, state: state)
            eval(kernelY, kernelState, opsY, opsState)

            print(
                "SSM decode: max |Δy| \(maxAbsDifference(opsY, kernelY)), "
                    + "max |Δstate| \(maxAbsDifference(opsState, kernelState))")
            XCTAssertEqual(opsY.shape, kernelY.shape)
            XCTAssertEqual(opsState.shape, kernelState.shape)
            XCTAssertTrue(allClose(opsY, kernelY, rtol: 1e-4, atol: 1e-5).item(Bool.self), "output")
            XCTAssertTrue(
                allClose(opsState, kernelState, rtol: 1e-4, atol: 1e-5).item(Bool.self), "state")
        }
    }

    // MARK: - BitLinear (no existing tolerance; float32 accumulation in both)

    func testBitLinearOpsMatchesKernel() throws {
        for (inFeatures, outFeatures, invert) in [(64, 32, false), (256, 128, true)] {
            for dtype in [DType.float32, DType.float16, DType.bfloat16] {
                do {
                    MLXRandom.seed(UInt64(inFeatures + outFeatures))
                    let layer = BitLinear(
                        inFeatures, outFeatures, bias: false, invertWeightScales: invert)
                    // Each 2-bit field is 0, 1 or 2 (ternary weight + 1).
                    let fields = MLXRandom.randInt(
                        low: 0, high: 3, [4, outFeatures / 4, inFeatures]
                    ).asType(.uint8)
                    let packed =
                        fields[0] | (fields[1] << 2) | (fields[2] << 4) | (fields[3] << 6)
                    let scale = MLXArray([Float(0.37)]).asType(dtype)
                    try layer.update(
                        parameters: ModuleParameters.unflattened([
                            "weight": packed, "weight_scale": scale,
                        ]),
                        verify: [.noUnusedKeys])
                    let x = MLXRandom.normal([3, 5, inFeatures]).asType(dtype)

                    let kernel = layer(x)
                    let ops = bitLinearMatmulOps(
                        x.reshaped(-1, inFeatures), packedWeights: layer.weight,
                        weightScale: layer.weightScale, outFeatures: outFeatures,
                        invertWeightScales: invert
                    ).reshaped(3, 5, outFeatures)
                    eval(kernel, ops)

                    let label = "in=\(inFeatures) out=\(outFeatures) invert=\(invert) \(dtype)"
                    print("BitLinear \(label): max |Δ| \(maxAbsDifference(ops, kernel))")
                    XCTAssertEqual(ops.dtype, kernel.dtype, label)
                    XCTAssertEqual(ops.shape, kernel.shape, label)
                    // One rounding to the output dtype in both; the float32 sums differ
                    // only in accumulation order.
                    let (rtol, atol): (Double, Double) =
                        switch dtype {
                        case .float32: (1e-5, 1e-4)
                        case .float16: (1e-3, 1e-3)
                        default: (1e-2, 1e-2)
                        }
                    XCTAssertTrue(
                        allClose(ops, kernel, rtol: rtol, atol: atol).item(Bool.self), label)
                }
            }
        }
    }

    // MARK: - Interpolation (no existing tolerance; float32 in both)

    func testBicubicInterpolateOpsMatchesKernel() {
        withRandomState(MLXRandom.RandomState(seed: 41)) {
            let x = MLXRandom.normal([2, 3, 13, 17])
            let cases: [(size: (Int, Int), alignCorners: Bool, antialias: Bool)] = [
                ((26, 34), false, false),
                ((26, 34), true, false),
                ((7, 9), false, false),
                ((7, 9), false, true),
                ((5, 30), false, true),
                ((13, 17), false, false),
            ]
            for testCase in cases {
                let kernel = bicubicInterpolate(
                    x, size: testCase.size, alignCorners: testCase.alignCorners,
                    antialias: testCase.antialias)
                let ops = bicubicOpsForTesting(
                    x, size: testCase.size, alignCorners: testCase.alignCorners,
                    antialias: testCase.antialias)
                eval(kernel, ops)
                let label = "\(testCase)"
                print("bicubic \(label): max |Δ| \(maxAbsDifference(ops, kernel))")
                XCTAssertEqual(ops.shape, kernel.shape, label)
                XCTAssertTrue(allClose(ops, kernel, rtol: 1e-5, atol: 1e-5).item(Bool.self), label)
            }
        }
    }

    func testNearestInterpolateOpsMatchesKernelExactly() {
        withRandomState(MLXRandom.RandomState(seed: 43)) {
            for dtype in [DType.float32, DType.float16] {
                let x = MLXRandom.normal([2, 3, 13, 17]).asType(dtype)
                for size in [(26, 34), (7, 9), (13, 17), (1, 40)] {
                    let kernel = nearestInterpolate(x, size: size)
                    let ops = nearestInterpolateOps(x, outH: size.0, outW: size.1)
                    eval(kernel, ops)
                    assertBitIdentical(ops, kernel, "\(dtype) \(size)")
                }
            }
        }
    }

    /// `bicubicInterpolate`'s parameter derivation, feeding the ops fallback.
    private func bicubicOpsForTesting(
        _ x: MLXArray, size: (Int, Int), alignCorners: Bool, antialias: Bool
    ) -> MLXArray {
        let scaleH = Float(size.0) / Float(x.dim(2))
        let scaleW = Float(size.1) / Float(x.dim(3))
        return bicubicInterpolateOps(
            x, outH: size.0, outW: size.1, alignCorners: alignCorners,
            antialias: antialias && (scaleH < 1 || scaleW < 1),
            filterScaleH: (antialias && scaleH < 1) ? 1 / scaleH : 1,
            filterScaleW: (antialias && scaleW < 1) ? 1 / scaleW : 1,
            support: 2)
    }

    // MARK: - ParoQuant pairwise rotation (ParoQuantTests' tolerances)

    func testPairwiseRotationOpsMatchesKernel() throws {
        let krot = 3
        for groupSize in [8, 128] {
            let dim = groupSize * 2
            let rot = PairwiseRotation(dims: dim, groupSize: groupSize, krot: krot)
            let theta = (MLXRandom.normal([krot, dim / 2]) * 0.5).asType(.float16)
            var pairs = [Int16]()
            for _ in 0 ..< krot {
                for _ in 0 ..< (dim / groupSize) {
                    pairs.append(contentsOf: (0 ..< groupSize).map { Int16($0) }.shuffled())
                }
            }
            let channelScales = (MLXRandom.normal([1, dim]) * 0.1 + 1.0).asType(.float16)
            try rot.update(
                parameters: ModuleParameters.unflattened([
                    "theta": theta, "pairs": MLXArray(pairs).reshaped(krot, dim),
                    "channel_scales": channelScales,
                ]),
                verify: [.all])
            rot.prepareDerivedRotationState()
            var state = RotationDerivedState(dims: dim, krot: krot)
            state.prepare(
                theta: theta, pairs: MLXArray(pairs).reshaped(krot, dim),
                channelScales: channelScales, groupSize: groupSize)

            for (dtype, tolerance) in [
                (DType.float16, Float(0.01)), (.bfloat16, 0.05), (.float32, 0.001),
            ] {
                for batch in [1, 5] {
                    let x = MLXRandom.normal([batch, dim]).asType(dtype)
                    let kernel = rot.rotate(x)
                    let ops = pairwiseRotationOps(
                        x, state: state, groupSize: groupSize, krot: krot)
                    eval(kernel, ops)

                    let label = "groupSize \(groupSize) \(dtype) batch \(batch)"
                    let diff = (ops - kernel).asType(.float32)
                    let reference = kernel.asType(.float32)
                    let relativeRMS = sqrt(
                        mean(diff * diff).item(Float.self)
                            / max(mean(reference * reference).item(Float.self), 1e-10))
                    print(
                        "rotation \(label): relative RMS \(relativeRMS), "
                            + "max |Δ| \(maxAbsDifference(ops, kernel))")
                    XCTAssertEqual(ops.dtype, kernel.dtype, label)
                    XCTAssertLessThan(relativeRMS, tolerance, label)
                }
            }
        }
    }
}

#endif
