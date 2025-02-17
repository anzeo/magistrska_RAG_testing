import shutil
from pathlib import Path
import chromadb
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent

# Mapping of short names to full collection names
collection_map = {
    "openai": "openai_embeddings_collection",
    "xlm-roberta": "xlm-roberta_embeddings_collection",
    "llama": "Llama_embeddings_collection",
    "sloberta": "sloberta_embeddings_collection",
    "sbert": "sbert_embeddings_collection",
}

files_map = {
    "tfidf": f"{ROOT_DIR}/tfidf_embeddings",
}


def delete_folder(key):
    try:
        folder = Path(files_map[key])
        if folder.exists() and folder.is_dir():
            print(f"Deleting embeddings: {folder.name}...")
            shutil.rmtree(folder)
            print(f"Embeddings '{folder.name}' deleted successfully!")
    except Exception as e:
        print("Error: ", e)


def delete_collection(key):
    try:
        collection_name = collection_map[key]

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=f"{ROOT_DIR}/chroma_data")

        # Delete the collection
        print(f"Deleting embeddings: {collection_name}...")
        chroma_client.delete_collection(collection_name)
        print(f"Embeddings '{collection_name}' deleted successfully!")
    except Exception as e:
        print("Error: ", e)


def delete_embeddings(key):
    if key == "all":
        for k in collection_map.keys():
            delete_collection(k)
        for k in files_map.keys():
            delete_folder(k)
    else:
        if key not in collection_map and key not in files_map:
            print(
                f"Error: Invalid collection name '{key}'. Choose from {list(collection_map.keys()) + list(files_map.keys())}.")
            return

        if key in files_map:
            delete_folder(key)
        else:
            delete_collection(key)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_embeddings.py <collection_name> | all")
        sys.exit(1)

    delete_embeddings(sys.argv[1])
