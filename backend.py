# backend.py
from gettext import translation
import os
import logging
from typing import List, Tuple, Optional

from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import google.generativeai as genai
import time


# ----- APIs -----
#  Log the presence/absence of all relevant API keys
log = logging.getLogger("startup")
def _mask(x): return f"{x[:4]}…{x[-4:]}" if x and len(x)>=8 else ("<empty>" if not x else "<short>")
log.info("ENV check: PINECONE_API_KEY=%s GEMINI_API_KEY=%s",
         bool(os.getenv("PINECONE_API_KEY")), bool(os.getenv("GEMINI_API_KEY")))

# Read API keys
PINECONE_API_KEY    = (os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY") or "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medrag").strip()
GEMINI_API_KEY  = (os.getenv("GEMINI_API_KEY") or "").strip()
EMBED_MODEL     = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")  # must match your index dims
TOP_K           = int(os.getenv("TOP_K", "5"))

# Configure Gemini (don’t crash if missing)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    log.warning("GEMINI_API_KEY is not set — responses will fall back to an error message.")

# Log what we see at boot (masked)
log.info("Config → index='%s', pinecone_key=%s, embed_model='%s'",
         PINECONE_INDEX_NAME, _mask(PINECONE_API_KEY), EMBED_MODEL)


# ----- PINECONE -----
# Check to ensure Pinecone vector DB index exists
_pc: Optional[Pinecone] = None
_idx = None

def _ensure_pinecone():
    """Create Pinecone client/index once; never raise at import/boot."""
    global _pc, _idx
    if _idx is not None:
        return _idx

    if not PINECONE_API_KEY:
        log.error("Missing Pinecone API key. Set PINECONE_API_KEY (or legacy PINECONE_KEY).")
        return None
    if not PINECONE_INDEX_NAME:
        log.error("Missing PINECONE_INDEX_NAME.")
        return None

    try:
        _pc = Pinecone(api_key=PINECONE_API_KEY)   # v5 client; environment not required
        _idx = _pc.Index(PINECONE_INDEX_NAME)      # host auto-discovery
        log.info("Pinecone client ready for index '%s'.", PINECONE_INDEX_NAME)
        return _idx
    except PineconeException:
        log.exception("Pinecone init failed.")
        return None


# ----- EMBEDDINGS -----
def embed(text: str) -> List[float]:
    """
    Create a retrieval-query embedding for `text` using Gemini.
    Returns an empty list on failure (callers should treat this as "no vector").
    """
    try:
        # task_type="retrieval_query" tells Gemini to optimize for search queries.
        res = genai.embed_content(model=EMBED_MODEL, content=text, task_type="retrieval_query")
        return res["embedding"]
    except Exception:
        log.exception("Gemini embed failed.")
        return []


# ----- VECTOR RETRIEVAL -----
def retrieve_chunks(query: str, k: int = TOP_K, namespace: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Embed `query`, search Pinecone for top-k nearest neighbors, and return
    a list of (chunk_text, score) pairs. If anything fails, returns [].
    """
    idx = _ensure_pinecone()
    if idx is None:
        return []

    vec = embed(query)
    if not vec:  # empty list → embedding failed
        return []

    try:
        out = idx.query(
            vector=vec,
            top_k=k,
            include_metadata=True,
            include_values=False,  # we only need metadata (text), not raw vectors
            namespace=(namespace or "")
        )
        # Filter to matches that actually carry text in metadata.
        return [
            (m.metadata["text"], m.score)
            for m in out.matches
            if m.metadata and "text" in m.metadata
        ]
    except PineconeException:
        log.exception("Pinecone query failed.")
        return []


# ----- RAG ANSWER PIPELINE -----
def answer_with_full_rag(
    question: str,
    mem: str,
    k: int = TOP_K,
    namespace: Optional[str] = None
) -> Tuple[str, List[Tuple[str, float]]]:

    """
    Full RAG flow:
      1) Retrieve top-k context chunks from Pinecone.
      2) Filter by a similarity score threshold.
      3) Multi-pass LLM pipeline using Gemini:
         - First pass: Respond, verify, safety-check (strictly checks context support or outputs 'Not enough information', tone down absolutes, keep simple language).
         - Second pass: Format and translate to question's original language (final donor-facing wording, add greetings, translate to user language).
    """

    high_model = genai.GenerativeModel("models/gemini-2.5-flash") # Previously was gemini-2.5-flash-lite
    low_model = genai.GenerativeModel("models/gemini-2.5-flash-lite") # Previously was gemini-2.5-flash-lite



    start = time.perf_counter()
  # First pass: summarize relevant context
    context_summarization_prompt = (
        "The following is a message history followed by the most recent question. \n"
        f"Context: {mem}\n\n"
        f"New question: {question}"
        "If the new question is dependent on information from the previous statements, rewrite the question so it is clear and can be sent to an answering agent.\n\n"
        "summarized question with detailed context:"
    )
    context_summary = high_model.generate_content(context_summarization_prompt).text.strip()

    elapsed_s = time.perf_counter() - start
    print(f"Elapsed: {elapsed_s:.6f} seconds")

    #print("CONTTXT",context_summary)

    start = time.perf_counter()


    # Retrieve chunks from Pinecone
    chunks = retrieve_chunks(question, k, namespace)
    if not chunks:
        return "Sorry, I couldn’t reach the knowledge base right now.", []

    # Filter by similarity to remove irrelevant chunks
    threshold = 0.4
    filtered = [(t, s) for t, s in chunks if s >= threshold]
    if not filtered:
        return "Sorry, I don't have enough information to answer your question. I can only see the most recent message, not the messsage history.", []

    context = "\n\n".join(t for t, _ in filtered)

    elapsed_s = time.perf_counter() - start
    print(f"Elapsed: {elapsed_s:.6f} seconds")
    
    #print("CONTEXTTT",context)

    start = time.perf_counter()

    # Second pass: obtain first chatbot answer + check validity of user query + ensure response is safe
    verify_prompt = (
        "You are a safe summary assistant for a blood donation chatbot pipeline.\n"
        f"Context:\n{context}\nQuestion: {context_summary}\n\n"
        "Respond to the question by drawing STRICTLY from the above context. If the answer cannot be found in the context, say 'Not enough information'. \n"
        "Write for a blood donor. Use straightforward language suitable for an uneducated audience. " 

        #"Self-Verification: "
        #"If the draft answer is not relevant to the question, output 'Not enough information'. DO NOT overextend yourself to report an answer if it does not clearly address the query. This is important. \n"
        #"If the draft answer DOES address the query, make any small edits to ensure that the answer is fully supported from the context alone \n\n"
        "Verified answer (be brief):"
    )
    verified_summary = high_model.generate_content(verify_prompt).text.strip()

    elapsed_s = time.perf_counter() - start
    print(f"Elapsed: {elapsed_s:.6f} seconds")

    #print(verified_summary)

    start = time.perf_counter()

    # Third pass: format model response + translate
    formatting_prompt = (
        "You are a formatting and language consistency assistant for a blood donor chatbot. \n"
        f"User's question: {context_summary} \n"
        f"Previous answer: {verified_summary}\n"
        "Formatting Guidelines: Take the previous answer and convert it into exactly what the blood donor should read, no internal notes,\n"
        "No references to the agent, just a clear and concise donor facing response.\n"
        "Use straightforward language for an uneducated audience; for niche, technical medical terms that a commoner might not understand, add parentheses with a few-word explanation / synonym. \n"
        f"Finally, look back at the question, and add any greetings/niceties such as 'Hello'/'Thank You'/'Good Afternoon/etc. If the user is JUST greeting you, thanking you etc., be sure to respond with the relevant greeting (e.g. Hello, you're welcome, etc.), even if the previous bot's answer says Not Enough Information." 
        "\nIf it is just an introductory greeting exchange like hi or hello, mention that you are a tool that can answer any general questions about blood donation, and that you can only see their most recent question, not the whole chat.  \n"
        "Formatted and translated answer (be brief):"
    )
    final_answer = low_model.generate_content(formatting_prompt).text.strip()
    elapsed_s = time.perf_counter() - start
    print(f"Elapsed: {elapsed_s:.6f} seconds")


    # Return the final answer plus the list of (chunk, score) pairs
    return final_answer, filtered
