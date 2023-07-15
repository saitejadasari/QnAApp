import pinecone


class PineconeService:
    def __init__(self):
        # connect to pinecone environment
        pinecone.init(
            api_key="45682637-3050-4490-b886-884db0f41035",
            environment="us-west1-gcp-free"
        )

    def get_index(self, retriever, index_name="extractive-question-answering", dimension=512):
        # check if the extractive-question-answering index exists
        if index_name not in pinecone.list_indexes():
            # create the index if it does not exist
            pinecone.create_index(
                index_name,
                dimension=retriever.get_sentence_embedding_dimension() if retriever else dimension,
                metric="cosine"
            )

        # connect to extractive-question-answering index we created
        index = pinecone.Index(index_name)
        return index, index_name

    def get_index_status(self, index):
        return index.describe_index_stats()

    def embed_and_store(self, index, retriever, chunks):
        embeds = retriever.encode(chunks).tolist()
        metadata = [{'id': i, 'context': chunk} for i, chunk in enumerate(chunks)]
        index_status = self.get_index_status(index)
        total_vectors = index_status['total_vector_count']
        ids = [f"{idx}" for idx in range(total_vectors, total_vectors+len(chunks))]
        to_upsert = list(zip(ids, embeds, metadata))
        # print(to_upsert)
        _ = index.upsert(vectors=to_upsert)

    # gets context passages from the pinecone index
    def get_context(self, index, retriever, question, top_k=1):
        # generate embeddings for the question
        xq = retriever.encode([question]).tolist()
        # search pinecone index for context passage with the answer
        xc = index.query(xq, top_k=top_k, include_metadata=True)
        # extract the context passage from pinecone search result
        c = [x["metadata"]["context"] for x in xc["matches"]]
        return c

    def delete_vectors(self, ids):
        index, _ = self.get_index(None)
        index.delete(ids)
