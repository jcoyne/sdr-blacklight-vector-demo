# frozen_string_literal: true

require "open3"
require "json"

class Qwen3Embedding
  # Default instruction used when embedding search queries.
  DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

  # Path to the Python helper script, relative to the Rails root.
  SCRIPT_PATH = Rails.root.join("script", "embed.py").to_s

  # Creates embeddings using Qwen/Qwen3-Embedding-0.6B running locally via a
  # Python subprocess. The model is downloaded automatically by HuggingFace on
  # first use and cached in ~/.cache/huggingface.
  #
  # @param input [Array<String>] texts to embed
  # @param instruction [String, nil] when provided, each text is prefixed with
  #   "Instruct: {instruction}\nQuery: {text}" before embedding. Pass
  #   DEFAULT_QUERY_INSTRUCTION (or any task-specific string) for query inputs;
  #   leave nil when embedding documents/passages.
  # @return [Array<Array<Float>>] one embedding vector per input string
  # @raise [RuntimeError] if the subprocess exits with a non-zero status
  def embedding(input:, instruction: nil)
    return [] if input.empty?

    payload = { texts: input }
    payload[:instruction] = instruction if instruction

    stdout, stderr, status = Open3.capture3(
      "uv", "run", "--script", SCRIPT_PATH,
      stdin_data: JSON.generate(payload)
    )

    unless status.success?
      raise "Qwen3Embedding subprocess failed (exit #{status.exitstatus}):\n#{stderr}"
    end

    JSON.parse(stdout)
  end
end
