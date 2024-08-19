from sentence_transformers import SentenceTransformer
import yaml
import numpy as np

# Load the pre-trained sBERT model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [' '.join(chunk) for chunk in chunks]


def preprocess(text):
    chunks = chunk_text(text)
    chunk_embeddings = model.encode(chunks)
    return np.mean(chunk_embeddings, axis=0)


def search(query, embeddings, top_n=10):
    query_embedding = model.encode(query)

    similarity_scores = model.similarity(query_embedding, embeddings).numpy().flatten()

    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    return top_indices, similarity_scores[top_indices]


if __name__ == '__main__':
    with open('ai_act.yaml', 'r') as file:
        data = yaml.safe_load(file)

    query = "Katere zahteve morajo izpolnjevati visokotvegani sistemi UI v zvezi s preglednostjo in zagotavljanjem informacij uvajalcem?"

    cleni = [f"{d['id_elementa']}" for d in data['cleni']]
    tocke = [f"{d['id_elementa']}" for d in data['tocke']]
    enote = cleni + tocke

    preprocessed_cleni_embeddings = [
        preprocess(
            d['poglavje']['naslov'] + "\n" + 
            (d['oddelek']['naslov'] + "\n" if d['oddelek'] else '') + 
            d['naslov'] + "\n" + 
            d['vsebina']
        ) for d in data['cleni']]
    
    preprocessed_tocke_embeddings = [
        preprocess(d['vsebina']) for d in data['tocke']
    ]

    preprocessed_enote_embeddings = preprocessed_cleni_embeddings + preprocessed_tocke_embeddings
    
    top_indices, scores = search(query, np.array(preprocessed_enote_embeddings))

    print("Relevantne enote:")
    for idx, score in zip(top_indices, scores):
        print(f"{enote[idx]} s podobnostjo {score}")
