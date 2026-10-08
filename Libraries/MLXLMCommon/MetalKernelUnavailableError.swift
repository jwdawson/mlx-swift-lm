// Copyright © 2026 Apple Inc.

import Foundation

#if !canImport(Metal)

/// Thrown off Apple platforms (MLX without Metal, e.g. the Linux CPU build) when a
/// requested feature exists only as a custom Metal kernel, with no plain-MLX-ops
/// equivalent. Model code with a reasonable unfused form uses that form instead and
/// never reaches this error.
public struct MetalKernelUnavailableError: Error, LocalizedError, Sendable, Equatable {
    /// What needed the kernel, e.g. "TurboQuant KV-cache compression".
    public let feature: String

    public init(feature: String) {
        self.feature = feature
    }

    public var errorDescription: String? {
        "\(feature) needs Metal, which this build of MLX does not have."
    }
}

#endif
