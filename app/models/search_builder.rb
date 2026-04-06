# frozen_string_literal: true

class SearchBuilder < Blacklight::SearchBuilder
  include Blacklight::Solr::SearchBuilderBehavior

  self.default_processor_chain += [ :add_embedding_to_query ]

  def add_embedding_to_query(solr_parameters)
    return unless blacklight_params[:q].present?

    solr_parameters[:q] = "{!parent which='doc_type_ssi:parent'}{!knn f=vector topK=10}[#{retrieve_embedding(blacklight_params[:q]).join(',')}]"
  end

  def retrieve_embedding(input)
    Rails.cache.fetch("embedding/#{input}") do
      client = Qwen3Embedding.new
      client.embedding(input: [ input ]).first
    end
  end
end
