import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = 'embeddings/sloberta/embeddings.npy'

model_name = "EMBEDDIA/sloberta"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
model = AutoModel.from_pretrained(model_name)

def preprocess(text):
    encoding = tokenizer.batch_encode_plus([text], padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)

    input_ids = encoding['input_ids']  # Token IDs
    attention_mask = encoding['attention_mask']  # Attention mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)

    normalized_embedding = sentence_embedding / sentence_embedding.norm(p=2, dim=1, keepdim=True)

    return normalized_embedding.squeeze(0)


def search(query, embeddings, top_n=None):
    query_embedding = preprocess(query)

    similarity_scores = cosine_similarity([query_embedding], embeddings).flatten()

    if top_n == None:
        # Take all relevant results
        top_indices = np.argsort(similarity_scores)[::-1]
    else:
        # Take only the top n relevant results
        top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    return top_indices, similarity_scores[top_indices]


def prepare_data():
    with open('ai_act.yaml', 'r') as file:
        data = yaml.safe_load(file)

    cleni = [f"{d['id_elementa']}" for d in data['cleni']]
    tocke = [f"{d['id_elementa']}" for d in data['tocke']]
    enote = cleni + tocke

    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading existing embeddings...")
        preprocessed_enote_embeddings = np.load(EMBEDDINGS_FILE)
    else:
        print("Getting new embeddings and storing them to file...\n")

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

        if not os.path.exists(os.path.dirname(EMBEDDINGS_FILE)):
            os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)

        np.save(EMBEDDINGS_FILE, preprocessed_enote_embeddings)

    return enote, np.array(preprocessed_enote_embeddings) 


def get_relevant_results(query="Katere zahteve morajo izpolnjevati visokotvegani sistemi UI v zvezi s preglednostjo in zagotavljanjem informacij uvajalcem?", top_n=None):
    enote, preprocessed_enote_embeddings = prepare_data()

    top_indices, scores = search(query, preprocessed_enote_embeddings, top_n)

    print("Relevantne enote:")
    for idx, score in zip(top_indices, scores):
        print(f"{enote[idx]} s podobnostjo {score}")


get_relevant_results("Kdaj začne uredba veljati in se uporabljati?")