import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_FILE = 'embeddings/Llama/embeddings.npy'

model_name = "meta-llama/Llama-2-7b-hf"  # Adjust to 7B, 13B, or 70B based on your resources
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained(model_name)

def preprocess(text):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = model(**inputs, output_hidden_states=True).hidden_states[-1]
    
    weights_for_non_padding = inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    embedding = sum_embeddings / num_of_none_padding_tokens
    return embedding.squeeze(0)


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
