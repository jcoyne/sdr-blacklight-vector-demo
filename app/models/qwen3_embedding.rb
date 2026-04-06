# frozen_string_literal: true

require "open3"
require "json"

class Qwen3Embedding
  # Path to the Python helper script, relative to the Rails root.
  SCRIPT_PATH = Rails.root.join("script", "embed.py").to_s

  # Creates embeddings using Qwen/Qwen3-Embedding-0.6B running locally via a
  # Python subprocess.  The model is downloaded automatically by HuggingFace on
  # first use and cached in ~/.cache/huggingface.
  #
  # @param input [Array<String>] texts to embed
  # @return [Array<Array<Float>>] one embedding vector per input string
  # @raise [RuntimeError] if the subprocess exits with a non-zero status
  def embedding(input:)
    return [] if input.empty?

    stdout, stderr, status = Open3.capture3(
      python_executable, "run", "--script", SCRIPT_PATH,
      stdin_data: JSON.generate(input)
    )

    unless status.success?
      raise "Qwen3Embedding subprocess failed (exit #{status.exitstatus}):\n#{stderr}"
    end

    JSON.parse(stdout)
  end

  private

  # Runs the embed script via uv, which automatically manages the Python
  # environment and dependencies declared in the script's inline metadata.
  def python_executable
    "uv"
  end
end
