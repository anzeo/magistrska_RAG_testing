import chromadb
from sentence_transformers import SentenceTransformer
import yaml
import numpy as np
import torch

COLLECTION_NAME = 'sbert_embeddings_collection'

# Load the pre-trained sBERT model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

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


def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]


def preprocess(text):
    chunks = chunk_text(text)
    chunk_embeddings = torch.tensor(model.encode(chunks))
    mean_embedding = torch.mean(chunk_embeddings, dim=0).squeeze(0)

    normalized_embedding = mean_embedding / mean_embedding.norm(p=2, dim=0, keepdim=True)

    return normalized_embedding.numpy()


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