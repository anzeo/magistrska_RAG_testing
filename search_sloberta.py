import chromadb
import numpy as np
from chromadb.api.types import IncludeEnum
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

COLLECTION_NAME = 'sloberta_embeddings_collection'

model_name = "EMBEDDIA/sloberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_data")

class EmbeddingManager:
    """Singleton class to manage embeddings."""
    _instance = None
    _collection = None

    @staticmethod
    def get_instance():
        """Get the singleton instance of the class."""
        if EmbeddingManager._instance is None:
            EmbeddingManager._instance = EmbeddingManager()
        return EmbeddingManager._instance

    def get_collection(self):
        """Retrieve or create the embeddings collection."""
        if self._collection is None:
            self._collection = chroma_client.get_or_create_collection(COLLECTION_NAME,
                                                                                  metadata={"hnsw:space": "cosine"})
        return self._collection

    def prepare_data(self):
        """Prepare and store embeddings in ChromaDB."""
        with open('ai_act.yaml', 'r') as file:
            data = yaml.safe_load(file)

        collection = self.get_collection()

        for d in data['cleni']:
            text = (
                    d['poglavje']['naslov'] + "\n" +
                    (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') +
                    d['naslov'] + "\n" +
                    d['vsebina']
            )
            embedding = preprocess(text)
            collection.add(
                ids=[d['id_elementa']],
                embeddings=[embedding],
                metadatas=[{"type": "cleni"}]
            )

        for d in data['tocke']:
            text = d['vsebina']
            embedding = preprocess(text)
            collection.add(
                ids=[d['id_elementa']],
                embeddings=[embedding],
                metadatas=[{"type": "tocke"}]
            )

    def load_embeddings(self):
        """Ensure embeddings are loaded into ChromaDB."""
        collection = self.get_collection()
        if collection.count() == 0:  # Check if the collection is empty
            print("Storing embeddings in ChromaDB...")
            self.prepare_data()


# def chunk_text(text, chunk_size=512):
#     words = text.split()
#     chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
#     return [' '.join(chunk) for chunk in chunks]

# def preprocess(text):
#     chunks = chunk_text(text)
#
#     encoding = tokenizer.batch_encode_plus(chunks, padding=True, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**encoding)
#
#     masked_hidden_states = outputs.last_hidden_state * encoding['attention_mask'].unsqueeze(-1)
#
#     # for i in range(len(encoding['input_ids'])):
#     #     tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][i])
#     #     for token, embedding in zip(tokens, masked_hidden_states[i]):
#     #         print(f"Token: {token}, Embedding Dimension: {embedding.shape}, Zero: {torch.all(embedding == 0)}")
#     #     print()
#
#     valid_token_counts = encoding['attention_mask'].sum(dim=1).unsqueeze(1)  # Count of non padding tokens
#     summed_embeddings = masked_hidden_states.sum(dim=1)  # Sum the embedding vectors of all tokens in each chunk
#     sentence_embeddings = summed_embeddings / valid_token_counts.clamp(
#         min=1e-9)  # Get sentence embeddings of each chunk
#
#     mean_embedding = sentence_embeddings.mean(dim=0)
#     normalized_embedding = mean_embedding / mean_embedding.norm(p=2, dim=0, keepdim=True)
#
#     return normalized_embedding.numpy()


def chunk_encoding(encoding, max_chunk_size=512, stride=256):
    """Chunk tokenized text, so it doesn't exceed model's max input length (for BERT models typically 512)"""
    input_ids = encoding['input_ids'][0]
    attention_mask = encoding['attention_mask'][0]

    # Chunk input_ids and attention_mask
    input_id_chunks = [input_ids[i:i + max_chunk_size] for i in range(0, len(input_ids), max_chunk_size - stride)]
    attention_mask_chunks = [attention_mask[i:i + max_chunk_size] for i in
                             range(0, len(attention_mask), max_chunk_size - stride)]

    # Pad chunks to max_chunk_size and stack them
    input_ids_padded = torch.stack([torch.cat([chunk, torch.zeros(max_chunk_size - len(chunk), dtype=torch.long)])
                                    for chunk in input_id_chunks])
    attention_mask_padded = torch.stack([torch.cat([chunk, torch.zeros(max_chunk_size - len(chunk), dtype=torch.long)])
                                         for chunk in attention_mask_chunks])

    return {'input_ids': input_ids_padded, 'attention_mask': attention_mask_padded}


def preprocess(text):
    encoding = tokenizer.encode_plus(text, padding=True, truncation=False, return_tensors='pt')
    chunked_encoding = chunk_encoding(encoding)
    with torch.no_grad():
        outputs = model(**chunked_encoding)

    masked_hidden_states = outputs.last_hidden_state * chunked_encoding['attention_mask'].unsqueeze(-1).float()

    valid_token_counts = chunked_encoding['attention_mask'].sum(dim=1).unsqueeze(1)  # Count of non padding tokens
    summed_embeddings = masked_hidden_states.sum(dim=1)  # Sum the embedding vectors of all tokens in each chunk
    sentence_embeddings = summed_embeddings / valid_token_counts.clamp(
        min=1e-9)  # Get sentence embeddings of each chunk

    mean_embedding = sentence_embeddings.mean(dim=0)
    normalized_embedding = mean_embedding / mean_embedding.norm(p=2, dim=0, keepdim=True)

    return normalized_embedding.numpy()


def search(query, top_n=None, unit_conditions: list[str] = None):
    if unit_conditions is None:
        unit_conditions = list(['cleni', 'tocke'])

    embedding_manager = EmbeddingManager.get_instance()
    embedding_manager.load_embeddings()

    collection = embedding_manager.get_collection()
    query_embedding = preprocess(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n or collection.count(),
        include=[IncludeEnum.distances],
        where={"type": {"$in": unit_conditions}},
    )

    return list(zip(results["ids"][0], np.subtract(1.0, results["distances"])[0]))


def get_relevant_results(
        query="Katere zahteve morajo izpolnjevati visokotvegani sistemi UI v zvezi s preglednostjo in zagotavljanjem informacij uvajalcem?",
        top_n=None):
    results = search(query, top_n)

    print("Relevantne enote:")
    for idx, score in results:
        print(f"{idx} s podobnostjo {score}")
