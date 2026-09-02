import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import argparse
import yaml

# load yaml config
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
class RAGQA:
    def __init__(self, config):
        self.index_path = config['build_dataset']['index_path']
        self.docs_path = config['build_dataset']['docs_path']
        self.embed_model_name = config['build_dataset']['embed_model']
        self.llama_model_name = config["model_name"]

        self.top_k = config["rag_qa"]['top_k']

        # 2. load faiss index and docs
        print(f"Loading FAISS index from {self.index_path} ...")
        self.index = faiss.read_index(self.index_path)

        print(f"Loading docs from {self.docs_path} ...")
        with open(self.docs_path, "rb") as f:
            self.docs = pickle.load(f)
        print(f"Loaded {len(self.docs)} docs.")

        # 3. load embed model
        print(f"Loading embedding model: {self.embed_model_name} ...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.emb_model = SentenceTransformer(self.embed_model_name, device=self.device)

        # 4. load model and tokenizer
        print(f"Loading Llama model: {self.llama_model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_name,
            torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
        ).to(self.device)

    # retrival part    
    def retrieve(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k

        # embed questions
        query_emb = self.emb_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # normalization
        faiss.normalize_L2(query_emb)

        # find top_k similar answers
        D, I = self.index.search(query_emb, top_k)

        retrieved_docs = [self.docs[i] for i in I[0]]
        scores = D[0].tolist()
        return retrieved_docs, scores

    # build prompt
    def build_prompt(self, user_query: str, retrieved_docs):
        context_block = "\n\n".join(
            f"[Note {i+1}]\n{doc}"
            for i, doc in enumerate(retrieved_docs)
        )

        system_content = (
            "You are a trading psychology and risk management coach in the style of Jesse Livermore.\n"
            "You will answer the user’s question using the retrieved Q&A notes below as your primary reference.\n"
            "Use the notes to guide and support your answer.\n"
            "You may summarize, rephrase, or generalize the ideas from the notes when answering.\n"
            "Do NOT hallucinate information that does not appear in the notes.\n"
            "Only say 'I'm unsure' if the notes contain no concepts that are relevant to the question.\n\n"
            "================= PLAYBOOK NOTES ================\n"
            f"{context_block}\n"
            "=================================================\n"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    # generate answers
    def answer(self, user_query: str, max_new_tokens, temperature, top_p, top_k: int = None):
        # retrival
        retrieved_docs, scores = self.retrieve(user_query, top_k=top_k)

        # build prompt
        prompt = self.build_prompt(user_query, retrieved_docs)

        # gengerate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        
        answer = self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer, retrieved_docs, scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    rag_qa = RAGQA(config)

    print("\nRAG QA ready. Type 'quit' to exit.\n")
    while True:
        q = input("USER: ").strip()
        if not q:
            continue
        if q.lower() in ["quit", "exit", "q"]:
            break

        max_new_tokens = config['rag_qa']['max_token']
        temperature = config['rag_qa']['temperature']
        top_p = config['rag_qa']['top_p']

        answer, docs, scores = rag_qa.answer(q, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

        print("\nBot:\n", answer)
        print("\n[Retrieved notes:]")
        for i, (d, s) in enumerate(zip(docs, scores)):
            print(f"\n--- Note {i+1} (score={s:.4f}) ---")
            print(d[:400], "...")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
