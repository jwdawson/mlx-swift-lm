// Copyright © 2026 Apple Inc.

import Dispatch

/// Runs one generation's blocking work outside Swift's cooperative pool.
actor GenerationWorker {
    #if canImport(Darwin)
    private nonisolated let queue = DispatchSerialQueue(
        label: "ml-explore.mlx-swift-lm.generation")
    #else
    // swift-corelibs-libdispatch has no DispatchSerialQueue (no SerialExecutor conformance).
    private nonisolated let queue = DispatchQueueSerialExecutor(
        label: "ml-explore.mlx-swift-lm.generation")
    #endif

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        queue.asUnownedSerialExecutor()
    }

    // Keep the body synchronous so all evaluation and cleanup stay on this executor.
    func run<R: Sendable>(_ body: @Sendable () -> R) -> R {
        body()
    }
}

#if !canImport(Darwin)
/// A serial executor that runs each job on a private serial `DispatchQueue`.
private final class DispatchQueueSerialExecutor: SerialExecutor, @unchecked Sendable {
    private let queue: DispatchQueue

    init(label: String) {
        queue = DispatchQueue(label: label)
    }

    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        queue.async {
            job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }
}
#endif
