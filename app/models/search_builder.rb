# frozen_string_literal: true

class SearchBuilder < Blacklight::SearchBuilder
  include Blacklight::Solr::SearchBuilderBehavior

  self.default_processor_chain += [ :add_embedding_to_query ]

  def add_embedding_to_query(solr_parameters)
    return unless blacklight_params[:q].present?

    must = solr_parameters.dig(:json, :query, :bool, :must)
    # must = []
    solr_parameters[:json][:query][:bool] = {
      should: must + [
        parent: {
          which: "doc_type_ssi:parent",
          query: {
            knn: {
              f: "vector",
              topK: 10,
              query:  "[#{retrieve_embedding(blacklight_params[:q]).join(', ')}]"
            }
          }
        }
      ]
    }
  end

  def retrieve_embedding(input)
    Rails.cache.fetch("embedding/#{input}") do
      client = Qwen3Embedding.new
      client.embedding(input: [ input ],
                       instruction: Qwen3Embedding::DEFAULT_QUERY_INSTRUCTION).first
    end
  end
end
