# NovaTech Enterprise RAG — Streamlit chatbot

A VS Code-ready Streamlit chat frontend for the n8n `NovaTech_Hybrid_RAG_API` workflow. The n8n backend uses Pinecone vector retrieval + keyword/BM25 retrieval over the NovaTech canonical JSON file, Reciprocal Rank Fusion (RRF), LLM reranking, and grounded cited answers.

## 1. Prepare n8n backend

Open: https://sowmyasharma.app.n8n.cloud/workflow/OlXh1M03Emq1zpAA

1. Open `Streamlit RAG API` (Webhook) node and create/select a **Header Auth** credential. Example header name `X-NovaTech-Key`, with a long random secret you generate. **Do not paste the secret into GitHub.**
2. Verify `Load Keyword Corpus` points to your public canonical JSON URL and retrieves all 72 documents.
3. Verify Pinecone index `novatech-enterprise-rag`, namespace `novatech-dev`, embedding model `text-embedding-3-small` (1536 dimensions).
4. Test the webhook using its **Test URL** while the n8n editor listens for a test event; for normal usage publish/activate the workflow and use its **Production URL**.
5. Open the n8n **Executions** tab if there are errors. Note that n8n cloud/plan execution timeouts can affect synchronous webhook replies.

## 2. Run locally in VS Code

```bash
cd novatech_streamlit_chatbot
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` to match your actual webhook URL and Header Auth credential value from n8n.

```bash
streamlit run app.py
```

The app will open at http://localhost:8501.

## Request contract

`POST` to the n8n webhook with header auth:

```json
{"question":"What is the current US meal per diem?","sessionId":"browser-session-uuid"}
```

Expected response:

```json
{
  "answer": "The current US meal per diem is $90/day [FIN-001].",
  "sources": [{"document_id":"FIN-001","title":"Travel Expense Policy 2026","department":"FIN","region":"US","status":"Active"}],
  "retrieval": {"method":"BM25 + Pinecone dense + RRF + LLM rerank","reranked_count":5},
  "sessionId":"browser-session-uuid"
}
```

This is a *sample contract*, not a verified live response. The API now filters `sources` to document IDs actually cited in the generated answer. This is a reference-integrity check, not proof that each cited document fully supports every claim.

## Retrieval algorithm

1. Pinecone dense search returns up to 12 semantically similar results.
2. BM25 keyword retrieval scores all 72 canonical documents; top 12 proceed.
3. RRF combines rankings using `1/(60 + rank)` per result list (not score normalization).
4. GPT reranks up to 10 fused candidate documents; the top five become the grounded answer context.
5. GPT generates an answer citing document IDs and the webhook returns JSON.

This uses **hybrid retrieval at the application layer**; no Pinecone sparse-vector index is required. The reranker is an LLM, **not a cross-encoder**, and is nondeterministic. The keyword index is rebuilt per query from GitHub, which is fine for this 72-document project but should be cached or indexed separately in production.

## Important limitations

- No role-based access control. Synthetic documents only.
- Backend citations and retrieval rankings need evaluation again after these changes.
- The workflow has not been verified to run successfully through the public production webhook; test and inspect n8n execution logs first.
- Secret must be held only in Streamlit secrets and n8n Credentials.
