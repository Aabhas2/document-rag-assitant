import json 
from app.rag.rag_pipeline import RAGPipeline 

def main(): 
    rag = RAGPipeline() 

    with open("scripts/eval_questions.json", "r") as f: 
        data = json.load(f) 

    hits = 0 
    for ex in data: 
        q = ex["question"]
        expected = ex["expected_source"]

        retrieved = rag.retrieve(q, top_k=5) 
        sources = [r["source"] for r in retrieved]
        ok = any(expected in (s or "") for s in sources) 

        hits += int(ok) 
        print(f"Q: {q}")
        print("Top sources:", sources[:3], "HIT" if ok else "MISS")
        print("-"*60)

    print(f"Retrieval hit-rate: {hits}/{len(data)} = {hits/len(data)}:.2f")

if __name__ == "__main__": 
    main() 