import chromadb
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

COLLECTION_NAME = 'sloberta_embeddings_collection'

model_name = "EMBEDDIA/sloberta"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
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
    encoding = tokenizer.batch_encode_plus([text], padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)

    input_ids = encoding['input_ids']  # Token IDs
    attention_mask = encoding['attention_mask']  # Attention mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)

    normalized_embedding = sentence_embedding / sentence_embedding.norm(p=2, dim=1, keepdim=True)

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
