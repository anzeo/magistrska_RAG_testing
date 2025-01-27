import os
import chromadb
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity

COLLECTION_NAME = 'Llama_embeddings_collection'

model_name = "meta-llama/Llama-2-7b-hf"  # Adjust to 7B, 13B, or 70B based on your resources
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer.pad_token = tokenizer.eos_token
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
        if EmbeddingManager._collection is None:
            EmbeddingManager._collection = chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        return EmbeddingManager._collection

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


def preprocess(text):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]
    
    weights_for_non_padding = inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    embedding = sum_embeddings / num_of_none_padding_tokens

    normalized_embedding = embedding / embedding.norm(p=2, dim=1, keepdim=True)

    return normalized_embedding.squeeze(0).numpy()


def search(query, top_n=None):
    embedding_manager = EmbeddingManager.get_instance()
    embedding_manager.load_embeddings()

    collection = embedding_manager.get_collection()
    query_embedding = preprocess(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n or collection.count(),
        include=['distances']
    )

    return list(zip(results["ids"][0], np.subtract(1.0, results["distances"])[0]))


def get_relevant_results(query="Katere zahteve morajo izpolnjevati visokotvegani sistemi UI v zvezi s preglednostjo in zagotavljanjem informacij uvajalcem?", top_n=None):
    results = search(query, top_n)

    print("Relevantne enote:")
    for idx, score in results:
        print(f"{idx} s podobnostjo {score}")

get_relevant_results("Kdaj začne uredba veljati in se uporabljati?")