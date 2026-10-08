// Copyright © 2026 Apple Inc.

#if canImport(os)

import os

typealias Logger = os.Logger

#else

final class Logger: Sendable {
    private let subsystem: String
    private let category: String

    init(subsystem: String, category: String) {
        self.subsystem = subsystem
        self.category = category
    }

    func debug(_ message: String) {
        print("[DEBUG] [\(subsystem).\(category)] \(message)")
    }

    func info(_ message: String) {
        print("[INFO] [\(subsystem).\(category)] \(message)")
    }

    func notice(_ message: String) {
        print("[NOTICE] [\(subsystem).\(category)] \(message)")
    }

    func warning(_ message: String) {
        print("[WARNING] [\(subsystem).\(category)] \(message)")
    }

    func error(_ message: String) {
        print("[ERROR] [\(subsystem).\(category)] \(message)")
    }

    func fault(_ message: String) {
        print("[FAULT] [\(subsystem).\(category)] \(message)")
    }
}

/// Accepts os.Logger-style `\(value, privacy: .public)` interpolations off Apple platforms,
/// so the same call sites compile; privacy has no effect on printed output.
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
            text += String(describing: value)
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

extension Logger {
    func debug(_ message: LogMessage) { debug(message.text) }
    func info(_ message: LogMessage) { info(message.text) }
    func notice(_ message: LogMessage) { notice(message.text) }
    func warning(_ message: LogMessage) { warning(message.text) }
    func error(_ message: LogMessage) { error(message.text) }
    func fault(_ message: LogMessage) { fault(message.text) }
}

#endif
