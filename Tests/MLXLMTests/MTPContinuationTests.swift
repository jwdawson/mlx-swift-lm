import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The public continuation seam: a caller that owns the target cache keeps a Qwen MTP
/// drafter in step with it across rounds, through `generateTokensTask` and `mtpPrefill`.
@Suite(.serialized)
struct MTPContinuationTests {
    private let promptA = [1, 2, 3, 4, 5]
    private let promptB = [6, 7, 8]

    // `hybrid`: a linear-attention layer before the full-attention one. `agreeable`: drafts
    // are mostly accepted; otherwise they are mostly rejected.
    @Test(arguments: [true, false], [true, false])
    func aResumedRoundSpeculatesAndMatchesPlainGreedy(hybrid: Bool, agreeable: Bool) async throws {
        let fixture = try ContinuationFixture(hybrid: hybrid, agreeable: agreeable)

        let cache = try fixture.newCache()
        let (tokens1, round1) = try await fixture.mtpRound(
            promptA, cache: cache, state: nil, continuation: nil)
        #expect(round1.passthroughReason == nil)
        #expect(round1.continuation != nil)
        // The round's last token is not in the cache, so it leads the next round's input.
        let lastToken = try #require(tokens1.last)
        #expect(fixture.position(of: cache) == promptA.count + tokens1.count - 1)
        let (tokens2, round2) = try await fixture.mtpRound(
            [lastToken] + promptB, cache: cache, state: round1.state,
            continuation: round1.continuation)
        #expect(round2.passthroughReason == nil)
        #expect(round2.drafted > 0)
        #expect(!agreeable || round2.accepted > 0)
        #expect(round2.continuation?.targetProcessedTokenCount == fixture.position(of: cache))

        // The plain iterator runs one step ahead, so its cache already holds that token.
        let plainCache = try fixture.newCache()
        let plain1 = try await fixture.plainRound(promptA, cache: plainCache)
        #expect(fixture.position(of: plainCache) == promptA.count + plain1.count)
        let plain2 = try await fixture.plainRound(promptB, cache: plainCache)
        #expect(tokens1 + tokens2 == plain1 + plain2)
    }

    @Test(arguments: [true, false], [true, false])
    func prefillThenGenerateMatchesPlainGreedy(hybrid: Bool, agreeable: Bool) async throws {
        let fixture = try ContinuationFixture(hybrid: hybrid, agreeable: agreeable)

        let cache = try fixture.newCache()
        let prefill = try fixture.prefill(promptA, cache: cache, state: nil, continuation: nil)
        #expect(prefill.passthroughReason == nil)
        #expect(prefill.drafted == 0)
        #expect(prefill.accepted == 0)
        #expect(fixture.position(of: cache) == promptA.count)
        #expect(prefill.continuation?.targetProcessedTokenCount == promptA.count)

        let (tokens, result) = try await fixture.mtpRound(
            promptB, cache: cache, state: prefill.state, continuation: prefill.continuation)
        #expect(result.passthroughReason == nil)
        #expect(result.drafted > 0)
        #expect(!agreeable || result.accepted > 0)

        let plain = try await fixture.plainRound(promptA + promptB, cache: fixture.newCache())
        #expect(tokens == plain)
    }

    @Test(arguments: [true, false], [true, false])
    func aCopiedContinuationResumesTheSameAsTheOriginal(
        hybrid: Bool, agreeable: Bool
    ) async throws {
        let fixture = try ContinuationFixture(hybrid: hybrid, agreeable: agreeable)

        let cache = try fixture.newCache()
        let prefill = try fixture.prefill(promptA, cache: cache, state: nil, continuation: nil)
        let continuation = try #require(prefill.continuation)
        let snapshot = continuation.copy()
        let cacheSnapshot = cache.map { $0.copy() }

        let (first, firstResult) = try await fixture.mtpRound(
            promptB, cache: cache, state: prefill.state, continuation: continuation)
        let (second, secondResult) = try await fixture.mtpRound(
            promptB, cache: cacheSnapshot, state: prefill.state, continuation: snapshot)
        #expect(firstResult.passthroughReason == nil)
        #expect(second == first)
        #expect(secondResult.passthroughReason == nil)
        #expect(secondResult.drafted == firstResult.drafted)
        #expect(secondResult.accepted == firstResult.accepted)
    }

    @Test(arguments: [true, false])
    func aMisalignedContinuationFallsBackToPlain(hybrid: Bool) async throws {
        let fixture = try ContinuationFixture(hybrid: hybrid, agreeable: true)

        let prefill = try fixture.prefill(
            promptA, cache: fixture.newCache(), state: nil, continuation: nil)
        let continuation = try #require(prefill.continuation)

        // A cache at another position than the one the continuation was finalized with.
        let other = try fixture.newCache()
        _ = try await fixture.plainRound(promptA + [9, 10], cache: other)
        #expect(fixture.position(of: other) != continuation.targetProcessedTokenCount)
        let otherSnapshot = other.map { $0.copy() }

        let (tokens, result) = try await fixture.mtpRound(
            promptB, cache: other, state: nil, continuation: continuation)
        #expect(result.passthroughReason != nil)
        #expect(result.continuation == nil)

        let plain = try await fixture.plainRound(promptB, cache: otherSnapshot)
        #expect(tokens == plain)
    }
}

private struct ContinuationFixture {
    let context: ModelContext
    let target: MLXLLM.Qwen35TextModel
    let drafter: MLXLLM.Qwen35MTPDraftModel
    let parameters = GenerateParameters(maxTokens: 12, temperature: 0)

    init(hybrid: Bool, agreeable: Bool) throws {
        MLXRandom.seed(1)
        let configuration = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(continuationTextConfigJSON(hybrid: hybrid).utf8))
        target = MLXLLM.Qwen35TextModel(configuration)
        drafter = MLXLLM.Qwen35MTPDraftModel(configuration)
        if agreeable { makeDrafterRepeatItsInput(drafter, hiddenSize: configuration.hiddenSize) }
        let tokenizer = ContinuationTokenizer()
        context = ModelContext(
            configuration: ModelConfiguration(id: "tiny-qwen-mtp-continuation"),
            model: target,
            processor: ContinuationInputProcessor(tokenizer: tokenizer),
            tokenizer: tokenizer)
    }

    func newCache() throws -> [KVCache] {
        try target.newCache(parameters: parameters)
    }

    func position(of cache: [KVCache]) -> Int {
        KVCacheStorage(cache, plan: .disabled).processedTokenCount
    }

    func input(_ tokens: [Int]) -> LMInput {
        LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    func mtpRound(
        _ tokens: [Int], cache: [KVCache], state: LMOutput.State?,
        continuation: MTPDrafterContinuation?
    ) async throws -> ([Int], MTPRoundResult) {
        let (stream, task) = try generateTokensTask(
            input: input(tokens), cache: cache, state: state, mtpContinuation: continuation,
            parameters: parameters, context: context, mtpDrafter: drafter, blockSize: 2,
            includeStopToken: true)
        let generated = await collectTokens(stream)
        return (generated, await task.value)
    }

    func prefill(
        _ tokens: [Int], cache: [KVCache], state: LMOutput.State?,
        continuation: MTPDrafterContinuation?
    ) throws -> MTPRoundResult {
        try mtpPrefill(
            input: input(tokens), cache: cache, state: state, mtpContinuation: continuation,
            parameters: parameters, context: context, mtpDrafter: drafter)
    }

    func plainRound(_ tokens: [Int], cache: [KVCache]) async throws -> [Int] {
        let (stream, task) = try generateTokensTask(
            input: input(tokens), cache: cache, state: nil, parameters: parameters,
            context: context, includeStopToken: true)
        let generated = await collectTokens(stream)
        await task.value
        return generated
    }
}

/// The tiny target mostly repeats its last token; a drafter that does the same gets
/// accepted drafts, so rounds exercise the accept path as well as the reject path.
private func makeDrafterRepeatItsInput(_ drafter: MLXLLM.Qwen35MTPDraftModel, hiddenSize: Int) {
    var updates: [String: MLXArray] = [:]
    for (key, value) in drafter.parameters().flattened() {
        if key.hasSuffix("o_proj.weight") || key.hasSuffix("down_proj.weight") {
            updates[key] = MLXArray.zeros(like: value)
        } else if key.hasSuffix("fc.weight") {
            updates[key] = concatenated(
                [MLXArray.eye(hiddenSize), MLXArray.zeros([hiddenSize, hiddenSize])], axis: 1
            ).asType(value.dtype)
        }
    }
    drafter.update(parameters: ModuleParameters.unflattened(updates))
}

private func collectTokens(_ stream: AsyncStream<TokenGeneration>) async -> [Int] {
    var tokens: [Int] = []
    for await event in stream {
        if case .token(let token) = event { tokens.append(token) }
    }
    return tokens
}

/// Hybrid: a linear-attention layer then a full-attention one, as the real Qwen3.5 targets.
private func continuationTextConfigJSON(hybrid: Bool) -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": \(hybrid ? 2 : 1),
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 16,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 64,
      "tie_word_embeddings": true,
      "attention_bias": false,
      "full_attention_interval": \(hybrid ? 2 : 1),
      "mtp_num_hidden_layers": 1,
      "mtp_use_dedicated_embeddings": false,
      "num_experts": 0,
      "num_experts_per_tok": 0,
      "moe_intermediate_size": 16,
      "shared_expert_intermediate_size": 16,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

private struct ContinuationTokenizer: Tokenizer {
    private let scalarBase = 0xE000

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            let value = Int(scalar.value) - scalarBase
            return (0 ..< 16).contains(value) ? value : 1
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map { String(UnicodeScalar(scalarBase + $0)!) }.joined()
    }

    func convertTokenToId(_: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(UnicodeScalar(scalarBase + id)!) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap {
            encode(text: $0["content"] as? String ?? "", addSpecialTokens: false)
        }
    }
}

private struct ContinuationInputProcessor: UserInputProcessor {
    let tokenizer: ContinuationTokenizer

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        return LMInput(
            tokens: MLXArray(
                try tokenizer.applyChatTemplate(
                    messages: messages, tools: input.tools,
                    additionalContext: input.additionalContext)))
    }
}
