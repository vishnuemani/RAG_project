# backend.py
from gettext import translation
import os
import logging
from typing import List, Tuple, Optional

from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import google.generativeai as genai


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

    model = genai.GenerativeModel("models/gemini-3-flash-preview") # Previously was gemini-2.5-flash-lite
    translated_question = question #model.generate_content(translation_prompt).text.strip()

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
    
    # First pass: summarize relevant context
    context_summarization_prompt = (
        "You are a safe and verified conversational assistant in a blood donor chat pipeline.\n"
        f"Context:\n{context}\nQuestion: {translated_question}\n\n"
        "No references to the agent, just a clear and concise donor facing response. Minimize wordiness.\n"
        "Try to provide a detailed, formatted summary for the given context that is relevant to the question.\n\n"
        "Formatted and summarized context:"
    )
    context_summary = model.generate_content(context_summarization_prompt).text.strip()


    # Second pass: obtain first chatbot answer + check validity of user query + ensure response is safe
    verify_prompt = (
        "You are a safe and verified conversational assistant in a blood donor chat pipeline.\n"
        f"Context:\n{context_summary}\nQuestion: {translated_question}\n\n"
        "Try to respond to the question using ONLY the above context, or say 'Not enough information'. \n"
        "Conversation Guidelines: Write for a blood donor. Use very straightforward, plain language suitable for an uneducated audience, unless the patient asks for a more technical answer. If it's a conversational comment or greeting, respond naturally. " 
        "\nIMPORTANT: you don't have the message history, but the user might not know this. If the query is not entirely a clear, standalone question about blood donation that you fully understand, (such as queries like 'I need help' or 'What does this mean?' or 'more information' or gibberish like 'Fidjfks'), respond that you don't understand AND that can only can see the most recent query and not the whole message history. \n\n"
        "Self-Verification: If the input explicitly signals insufficient info or a clarification request, output verbatim 'Not enough information' and DO NOT write anything more.\n"
        "If the draft answer is not relevant to the question, output 'Not enough information'. DO NOT overextend yourself to report an answer if the draft does not clearly address the query. This is important. \n"
        "If the draft answer DOES address the query, make minor edits to the draft answer to ensure that the answer is fully supported from the context alone; prefer straightforward wording for an uneducated audience, unless the query indicates the person is knowledgeable, in which case technical terms are acceptable. \n\n"
        "Safety Guidelines: If you see 'Not enough information' or an explicit clarification request, output 'Not enough information' verbatim and DO NOT GENERATE ANYTHING ELSE.\n"
        "Otherwise, make small adjustments to the draft by moderating absolutes and using reasonably cautious language. \n"
        "Keep wording straightforward for an uneducated audience \n\n"
        "Self-verified and safety-checked answer (be brief):"
    )
    verified_summary = model.generate_content(verify_prompt).text.strip()


    # Third pass: format model response + translate
    formatting_prompt = (
        "You are a formatting and language consistency assistant for a blood donor chatbot. \n"
        f"User's question: {question} \n"
        f"Previous answer: {verified_summary}\n"
        "Formatting Guidelines: Take the previous answer and convert it into exactly what the blood donor should read, no internal notes,\n"
        "No references to the agent, just a clear and concise donor facing response. Minimize wordiness.\n"
        "Use very straightforward language for an uneducated audience; for ALL technical terms that a villager might not understand, add parentheses with a few-word explanation / synonym. \n"
        f"Finally, look back at the question, and add any greetings/niceties such as 'Hello'/'Thank You'/'Good Afternoon/etc. If the user is JUST greeting you, thanking you etc., be sure to respond with the relevant greeting (e.g. Hello, you're welcome, etc.), even if the previous bot's answer says Not Enough Information." 
        "\nIf it is just an introductory greeting exchange like hi or hello, mention that you are a tool that can answer any general questions about blood donation, and that you can only see their most recent question, not the whole chat.  \n"
        "Translation Guidelines:"
        "Make sure the user's question is in the SAME language as our answer. \n"
        f"(Silently) figure out what language the bulk of this question (NOT just the greeting or a select few words) is most likely in: {question}. If it is in multiple langauges, take the most predominant language in the bulk of the question. It is most likely English. \n"
        f"Only if it is NOT English, translate my English answer back to the other language you identified. Make sure that your final answer is in the SAME LANGUAGE as the initial question, but exactly the same information. \n\n My English answer:\n{verified_summary}\n\nWhat is the answer for the patient? ONLY output the pure, patient-facing answer, no other words or extraneous output (such as anything about translation)."    
        "\nFormatted and translated answer (be brief):"
    )
    final_answer = model.generate_content(formatting_prompt).text.strip()


    # Return the final answer plus the list of (chunk, score) pairs
    return final_answer, filtered
