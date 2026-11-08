"""NovaTech Enterprise RAG chat. Start with: streamlit run app.py"""
from __future__ import annotations

import json
import uuid
import re
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote, urlparse

import requests
import streamlit as st

st.set_page_config(page_title="NovaTech | Knowledge Assistant", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

STYLE = """
<style>
:root { --navy:#10243c; --blue:#2467da; --mint:#dff6ec; }
.block-container {max-width:1160px;padding-top:1.5rem;padding-bottom:5rem;}
.hero {border-radius:22px;background:linear-gradient(120deg,#102641 0%,#205d91 100%);
       padding:25px 30px 23px;margin-bottom:18px;color:white;box-shadow:0 10px 30px rgba(12,42,78,.12)}
.hero h1 {color:#fff;margin:0;font-size:2rem;letter-spacing:-.03em;}
.hero p {color:#e4effd;margin:.45rem 0 0;line-height:1.5}
.hero-tag {display:inline-block;font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;
           border-radius:999px;background:rgba(255,255,255,.17);padding:5px 10px;margin-bottom:9px;}
.section-label {font-weight:700;font-size:.9rem;color:#66768b;margin:5px 0 10px;}
.source-card {border:1px solid #dfe7f0;border-left:4px solid #3586dc;border-radius:10px;
              padding:11px 14px;margin:9px 0;background:rgba(109,159,208,.06)}
.source-title {font-weight:700;color:#245c9b;font-size:.92rem;}
.source-meta {font-size:.78rem;color:#6c7a8d;margin-top:4px;}
.status-chip {padding:4px 9px;background:#e6f5ea;color:#1e7845;font-size:.78rem;border-radius:999px;font-weight:700}
.note {font-size:.85rem;color:#7c8ca1;}
[data-testid="stSidebar"] {border-right:1px solid rgba(150,170,190,.18)}
.stButton button {border-radius:12px;}
[data-testid="stChatMessage"] {border-radius:14px;}
@media(max-width:760px){.hero{padding:18px}.hero h1{font-size:1.6rem}}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# Remote GitHub is the authoritative document source; no bundled data directory.
DEFAULT_GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "sowmyasharma/novatech_rag_datasetNovaTech-Enterprise-RAG/"
    "refs/heads/main"
)

def github_raw_base() -> str:
    """Use a trusted raw.githubusercontent.com repository prefix."""
    try:
        value = str(st.secrets.get("GITHUB_RAW_BASE_URL", DEFAULT_GITHUB_RAW)).strip().rstrip("/")
    except (FileNotFoundError, KeyError):
        value = DEFAULT_GITHUB_RAW
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
        raise RuntimeError("GITHUB_RAW_BASE_URL must use https://raw.githubusercontent.com/")
    if not parsed.path.strip("/"):
        raise RuntimeError("GITHUB_RAW_BASE_URL must identify a repository and branch.")
    return value

def github_url(path: str) -> str:
    # Disallow path traversal, query strings, and external hosts.
    pieces = path.replace("\\\\", "/").split("/")
    if not pieces or any(s in ("", ".", "..") for s in pieces):
        raise ValueError("Invalid document path")
    if any("?" in s or "#" in s for s in pieces):
        raise ValueError("Unexpected URL punctuation in document path")
    return github_raw_base() + "/" + "/".join(quote(s, safe="") for s in pieces)

def github_page_url(path: str) -> str:
    raw = github_raw_base().replace("https://raw.githubusercontent.com/", "")
    parts = raw.split("/")
    if len(parts) < 4:
        raise RuntimeError("The GitHub raw base must include owner, repo, and branch.")
    owner, repo, ref_type, *rest = parts
    if ref_type == "refs" and len(rest) >= 2 and rest[0] == "heads":
        branch = "/".join(rest[1:])
    else:
        branch = "/".join(parts[2:])
    # The blob URL is convenient for human browsing.
    return "https://github.com/" + quote(owner, safe="") + "/" + quote(repo, safe="") + "/blob/" + quote(branch, safe="/") + "/" + "/".join(quote(s, safe="") for s in path.split("/"))

@st.cache_data(ttl=600, show_spinner=False)
def _download_file(url: str) -> bytes:
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    if len(response.content) > 15_000_000:
        raise ValueError("Document exceeds preview download limit.")
    return response.content

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_corpus(url: str) -> dict[str, dict]:
    data = json.loads(_download_file(url).decode("utf-8"))
    if isinstance(data, dict):
        data = data.get("documents", [])
    if not isinstance(data, list):
        raise ValueError("Canonical JSON must contain a list of documents.")
    return {str(d["document_id"]): d for d in data if isinstance(d, dict) and d.get("document_id")}

def load_corpus() -> dict[str, dict]:
    return _fetch_corpus(github_url("documents_canonical.json"))

def resolve_document(doc_id: str) -> tuple[dict | None, str | None]:
    doc = load_corpus().get(doc_id)
    if not doc:
        return None, None
    file_path = str(doc.get("file_path") or doc.get("source_path") or "")
    if not file_path.startswith("documents/"):
        return doc, None
    try:
        return doc, github_url(file_path)
    except ValueError:
        return doc, None

def render_document_preview(doc: dict) -> None:
    for section in doc.get("sections", []):
        st.markdown(f"**{section.get('heading','Section')}**")
        st.markdown(str(section.get("text", "")))

@st.dialog("NovaTech source document", width="large")
def document_preview(doc_id: str):
    doc, original_url = resolve_document(doc_id)
    if doc is None:
        st.warning(f"{doc_id} wasn't found in the GitHub document collection.")
        return
    st.subheader(f"{doc_id} · {doc.get('title','Document')}")
    st.caption(" · ".join(str(doc.get(k,"")) for k in ("department","region","status","version") if doc.get(k)))
    if original_url:
        file_path = str(doc["file_path"])
        st.link_button("↗ Open original file on GitHub", github_page_url(file_path), use_container_width=True)
        try:
            contents = _download_file(original_url)
            st.download_button(
                "↓ Download original file",
                data=contents,
                file_name=file_path.rsplit("/", 1)[-1],
                mime={".pdf":"application/pdf",".docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",".json":"application/json",".md":"text/markdown"}.get("." + file_path.rsplit(".",1)[-1],"application/octet-stream"),
                key=f"download_{doc_id}",use_container_width=True,
            )
        except (requests.RequestException, ValueError) as ex:
            st.info("The original source is not uploaded to GitHub yet or is unavailable. The normalized document preview below still works.")
    st.divider()
    render_document_preview(doc)
    st.caption("Live document preview from GitHub canonical JSON.")

def render_cited_answer(answer: str):
    known = load_corpus()
    def replace(match):
        ident = match.group(1)
        return f"[Source {ident}](?doc={quote(ident)})" if ident in known else match.group(0)
    st.markdown(re.sub(r"\[([A-Z]+-\d{3})\]", replace, str(answer)))

try:
    document_count = len(load_corpus())
except (requests.RequestException, ValueError, RuntimeError, json.JSONDecodeError):
    st.error("Could not load the canonical dataset from GitHub. Check GITHUB_RAW_BASE_URL, repository visibility, and the file path.")
    st.stop()

query_doc = st.query_params.get("doc")
if query_doc and re.fullmatch(r"[A-Z]+-\\d{3}", str(query_doc)):
    if str(query_doc) in load_corpus():
        document_preview(str(query_doc))
    st.query_params.clear()


PROMPTS = {
    "HR": ["What are the US PTO entitlements?", "What happens during employee offboarding?"],
    "IT & Security": ["What should I do if my laptop is stolen?", "What are the MFA requirements?"],
    "Finance": ["What is the current US meal per diem?", "What approvals are needed for conference expenses?"],
    "Product & Sales": ["Can Starter customers use SSO?", "What is the P1 customer support response SLA?"],
    "Legal & Compliance": ["Can customer PII be uploaded to a public chatbot?", "What are the data privacy requirements?"],
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pending" not in st.session_state:
    st.session_state.pending = None
def reset():
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pending = None


def secrets():
    try:
        url = str(st.secrets["N8N_WEBHOOK_URL"]).strip()
        name = str(st.secrets["N8N_AUTH_HEADER"]).strip()
        value = str(st.secrets["N8N_AUTH_VALUE"]).strip()
        if not (url.startswith("https://") and name and value):
            return None
        return url, name, value
    except (KeyError, FileNotFoundError):
        return None


def query_backend(question: str) -> dict:
    cfg = secrets()
    if cfg is None:
        raise RuntimeError("Add your production webhook URL and Header Auth values in .streamlit/secrets.toml.")
    url, header_name, header_value = cfg
    response = requests.post(
        url,
        headers={header_name: header_value, "Content-Type": "application/json"},
        json={"question": question, "sessionId": st.session_state.session_id},
        timeout=150,
    )
    if response.status_code == 404:
        raise RuntimeError("n8n returned 404. Verify that N8N_WEBHOOK_URL is the published /webhook/ URL, not /webhook-test/.")
    if response.status_code in (401, 403):
        raise RuntimeError("n8n rejected the request. Verify the X-NovaTech-Key header name and secret in Streamlit and n8n.")
    if response.status_code >= 400:
        raise RuntimeError(f"n8n returned HTTP {response.status_code}. Open its Executions tab and inspect the failing node.")
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise RuntimeError("n8n returned an unexpected JSON response. Check Respond to Webhook.")
    if not payload.get("answer") and not payload.get("output"):
        raise RuntimeError("The response did not contain an answer. Check the n8n Build API Response node.")
    return payload


def show_sources(sources: list[dict], prefix: str):
    if not sources:
        return
    st.markdown("**Referenced documents**")
    for idx, source in enumerate(sources):
        raw_id = str(source.get("document_id", ""))
        doc_id = escape(raw_id)
        title = escape(str(source.get("title", "Untitled document")))
        meta = " · ".join(
            escape(str(source.get(key))) for key in ("department", "region", "status") if source.get(key)
        )
        st.markdown(
            f'<div class="source-card"><div class="source-title">{doc_id} · {title}</div>'
            f'<div class="source-meta">{meta}</div></div>',
            unsafe_allow_html=True,
        )
        a, b = st.columns([1,1.2])
        with a:
            if st.button("📖 Preview document", key=f"preview_{prefix}_{idx}", use_container_width=True):
                document_preview(raw_id)
        with b:
            doc, original_url = resolve_document(raw_id)
            if original_url:
                st.link_button("↗ View on GitHub", github_page_url(str(doc["file_path"])), use_container_width=True)
            else:
                st.caption("Source file unavailable")


with st.sidebar:
    st.markdown("### 🧠 NovaTech")
    st.caption("Enterprise knowledge workspace")
    if st.button("＋ New conversation", use_container_width=True, type="primary"):
        reset()
        st.rerun()
    st.divider()
    st.markdown("**Knowledge base**")
    st.caption(f"{document_count} GitHub-hosted synthetic documents · 8 departments")
    st.divider()
    st.markdown("**Conversation**")
    st.caption(f"{len([x for x in st.session_state.messages if x['role']=='user'])} questions in this session")
    st.download_button(
        "↓ Export chat JSON",
        data=json.dumps({"session_id":st.session_state.session_id,"messages":st.session_state.messages},indent=2),
        file_name="novatech_chat.json",
        mime="application/json",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    st.divider()
    st.caption("Prototype uses synthetic documents. Source previews load from GitHub. No employee-level permissions; do not use with confidential records.")

st.markdown(
    '<div class="hero"><div class="hero-tag">NovaTech · Enterprise AI</div>'
    '<h1>Knowledge Assistant</h1>'
    '<p>Ask a question across HR, IT, Finance, Product and Compliance. '
    'Get clear answers grounded in your enterprise document collection.</p></div>',
    unsafe_allow_html=True,
)

hdr_a, hdr_b = st.columns([1.2,1])
with hdr_a:
    st.metric("Knowledge base", f"{document_count} docs")
with hdr_b:
    st.metric("Departments", "8")

if not secrets():
    st.error("Webhook credentials are not configured. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and fill in your values.")
    st.stop()

if not st.session_state.messages:
    st.markdown("<div class='section-label'>EXPLORE BY DEPARTMENT</div>", unsafe_allow_html=True)
    group = st.selectbox("Topic", options=list(PROMPTS), label_visibility="collapsed")
    p1,p2 = st.columns(2)
    for idx,(column,q) in enumerate(zip((p1,p2),PROMPTS[group])):
        with column:
            if st.button(q, key=f"quick_{idx}_{group}",use_container_width=True):
                st.session_state.pending = q
    st.caption("Or type your own question below.")
else:
    st.markdown("<div class='section-label'>YOUR CONVERSATION</div>", unsafe_allow_html=True)

for i,msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
        if msg["role"] == "assistant" and not msg.get("error"):
            render_cited_answer(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and not msg.get("error"):
            sources = msg.get("sources") or []
            if sources:
                with st.expander(f"📄 Sources · {len(sources)}", expanded=False):
                    show_sources(sources, f"history_{i}")
            rate_a,rate_b,_=st.columns([1,1,5])
            if rate_a.button("👍",key=f"up_{i}",help="Helpful"):
                st.session_state.messages[i]["feedback"] = "helpful"
            if rate_b.button("👎",key=f"down_{i}",help="Needs work"):
                st.session_state.messages[i]["feedback"] = "needs_work"
            if msg.get("feedback"):
                st.caption("Feedback saved to this session's chat export.")

user_text = st.chat_input("Ask NovaTech a question…", max_chars=2000)
question = st.session_state.pending or user_text
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role":"user","content":question,"time":datetime.now(timezone.utc).isoformat()})
    with st.chat_message("user",avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant",avatar="🤖"):
        with st.spinner("Preparing your answer…"):
            try:
                payload=query_backend(question)
                answer=str(payload.get("answer") or payload.get("output"))
                sources=payload.get("sources") if isinstance(payload.get("sources"),list) else []
                render_cited_answer(answer)
                if sources:
                    with st.expander(f"📄 Sources · {len(sources)}"):
                        show_sources(sources, "latest")
                st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources,"time":datetime.now(timezone.utc).isoformat()})
            except requests.exceptions.Timeout:
                err="The request timed out. Review the latest n8n execution or try again."
                st.error(err)
                st.session_state.messages.append({"role":"assistant","content":err,"error":True})
            except requests.exceptions.RequestException as ex:
                err=f"Could not reach the n8n backend ({type(ex).__name__}). Confirm network connectivity and webhook URL."
                st.error(err)
                st.session_state.messages.append({"role":"assistant","content":err,"error":True})
            except (RuntimeError,ValueError) as ex:
                st.error(str(ex))
                st.session_state.messages.append({"role":"assistant","content":str(ex),"error":True})
