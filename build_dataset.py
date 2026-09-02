import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import argparse
import yaml
import os

# load yaml config
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# build the dataset
def load_dataset(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["prompt"].strip()
            a = item["response"].strip()
            text = f"Question: {q}\nAnswer: {a}"
            docs.append(text)
    return docs

def build_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    DATA_PATH = config['build_dataset']['data_path']
    INDEX_PATH = config['build_dataset']['index_path'] # vector database
    DOCS_PATH  = config['build_dataset']['docs_path'] # Original document list
    
    
    EMBED_MODEL_NAME = config['build_dataset']['embed_model']

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    
    # load dataset
    print("Loading dataset...")
    docs = load_dataset(DATA_PATH)
    print(f"Loaded {len(docs)} docs")

    # load embed model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # embed the ds into token vectors
    print("Encoding docs to embeddings ...")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    print(f"Embeddings shape: {embeddings.shape}")

    # build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # normalization first
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print("Saving index and docs...")
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("The dataset building is finished.")

if __name__ == "__main__":
    build_index()
