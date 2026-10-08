// Copyright © 2026 Apple Inc.

#if canImport(os)

import os

typealias Logger = os.Logger

#else

/// Prints to standard output off Apple platforms. Like `os.Logger`, it takes only
/// interpolated messages (``LogMessage``), and every interpolated value that is not
/// marked `privacy: .public` prints as `<private>`.
final class Logger: Sendable {
    private let subsystem: String
    private let category: String

    init(subsystem: String, category: String) {
        self.subsystem = subsystem
        self.category = category
    }

    func debug(_ message: LogMessage) {
        print("[DEBUG] [\(subsystem).\(category)] \(message.text)")
    }

    func info(_ message: LogMessage) {
        print("[INFO] [\(subsystem).\(category)] \(message.text)")
    }

    func notice(_ message: LogMessage) {
        print("[NOTICE] [\(subsystem).\(category)] \(message.text)")
    }

    func warning(_ message: LogMessage) {
        print("[WARNING] [\(subsystem).\(category)] \(message.text)")
    }

    func error(_ message: LogMessage) {
        print("[ERROR] [\(subsystem).\(category)] \(message.text)")
    }

    func fault(_ message: LogMessage) {
        print("[FAULT] [\(subsystem).\(category)] \(message.text)")
    }
}

/// Accepts os.Logger-style `\(value, privacy: .public)` interpolations off Apple platforms.
/// Values are redacted to `<private>` unless marked `.public` (stricter than os.Logger's
/// `.auto`, which also shows scalars: nothing is printed in clear unless asked for).
struct LogMessage: ExpressibleByStringInterpolation, Sendable {
    enum Privacy: Sendable {
        case `public`, `private`, auto
    }

    struct StringInterpolation: StringInterpolationProtocol {
        var text = ""

        init(literalCapacity: Int, interpolationCount: Int) {
            text.reserveCapacity(literalCapacity)
        }

        mutating func appendLiteral(_ literal: String) {
            text += literal
        }

        mutating func appendInterpolation<T>(_ value: T, privacy: Privacy = .auto) {
            text += privacy == .public ? String(describing: value) : "<private>"
        }
    }

    let text: String

    init(stringLiteral value: String) {
        text = value
    }

    init(stringInterpolation: StringInterpolation) {
        text = stringInterpolation.text
    }
}

#endif
