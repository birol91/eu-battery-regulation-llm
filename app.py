"""FastAPI backend — EU Battery Regulation LLM Compliance Assistant v1.0

On first launch, downloads model (~7.2 GB) and RAG index (~0.5 GB) from HuggingFace automatically.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel

HF_REPO = "birol91/eu-battery-regulation-llm"
CACHE_DIR = Path.home() / ".cache" / "eu-battery-llm"
GGUF_PATH = CACHE_DIR / "eu-battery-mistral-q8.gguf"
INDEX_DIR = CACHE_DIR / "rag_index"
STATIC_DIR = Path(__file__).parent
SRC_DIR = Path(__file__).parent / "src"

sys.path.insert(0, str(SRC_DIR))


def ensure_assets():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not GGUF_PATH.exists():
        print("Downloading model weights (~7.2 GB) from HuggingFace...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename="eu-battery-mistral-q8.gguf",
            local_dir=CACHE_DIR,
            token=os.getenv("HF_TOKEN"),
        )
        print("Model downloaded.")

    index_file = INDEX_DIR / "faiss.index"
    if not index_file.exists():
        print("Downloading RAG index from HuggingFace...")
        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns="rag_index/*",
            local_dir=CACHE_DIR,
            token=os.getenv("HF_TOKEN"),
        )
        print("RAG index downloaded.")


ensure_assets()

from retriever import Retriever  # noqa: E402
from llama_cpp import Llama  # noqa: E402

SYSTEM = (
    "You are an EU Battery Regulation compliance assistant. "
    "Answer ONLY based on the provided regulation excerpts below. "
    "CITATION RULE: After every factual sentence, cite the source in this exact format: "
    "*(Source, Heading)* — for example: "
    "'Batteries must achieve 70% recycling efficiency by 2027. *(EU 2023/1542, Annex XII)*' "
    "or 'REESS stores electrical energy for traction. *(ECE R100 Part II, Section 6.1)*'. "
    "Use the Source and Heading fields from the excerpts. Every factual claim needs a citation. "
    "If the excerpts do not contain enough information to answer, reply exactly: "
    '"I don\'t have enough information in the regulation to answer this question."'
)

ABSTAIN = "I don't have enough information in the regulation to answer this question."

SOURCE_FILE_MAP = {
    "EU_2023_1542": "CELEX_32023R1542_EN_TXT.pdf",
    "EU_2025_R1561": "CELEX_32025R1561_EN_TXT.pdf",
    "EU_2025_R0606": "CELEX_32025R0606_EN_TXT.pdf",
    "EU_2025_R0606_memo": "CELEX_32025R0606_EN_TXT.pdf",
    "EU_2025_D0934": "CELEX_32025D0934_EN_TXT.pdf",
    "ECE_R100_PartII": "R100r3e.pdf",
    "ECE_R155": "R155e.pdf",
    "UN_38_3": "UN38.3.pdf",
    "BatteryPass": "BatteryPass.pdf",
    "BatteryPass_ValueAssessment": "2024_BatteryPassport_Value_Assessment.pdf",
    "OJ_C_2025_214": "OJ_C_202500214_EN_TXT.pdf",
}

n_gpu_layers = int(os.getenv("N_GPU_LAYERS", "-1"))
port = int(os.getenv("PORT", "7860"))

print("Loading model...")
llm = Llama(model_path=str(GGUF_PATH), n_gpu_layers=n_gpu_layers, n_ctx=4096, verbose=False)
retriever = Retriever(index_dir=INDEX_DIR)
print(f"App ready → http://localhost:{port}  (open this in your browser)")

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c['source']} — {c['heading']})\n{c['text']}" for c in chunks
    )
    return f"[INST] {SYSTEM}\n\nREGULATION EXCERPTS:\n{context}\n\nQUESTION: {question} [/INST]"


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    chunks = retriever.retrieve(question, top_k=5, similarity_floor=0.45)
    if not chunks:
        return {"answer": ABSTAIN, "sources": []}

    prompt = build_prompt(question, chunks)
    out = llm(prompt, max_tokens=512, temperature=0.2, stop=["[/INST]", "[INST]"])
    answer = out["choices"][0]["text"].strip()

    seen = set()
    unique_sources = []
    for c in chunks:
        fname = SOURCE_FILE_MAP.get(c["source"], c["source"])
        if fname not in seen:
            seen.add(fname)
            unique_sources.append(fname)
    return {"answer": answer, "sources": unique_sources}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)
