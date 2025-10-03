# backend.py
from gettext import translation
import os
import logging
from typing import List, Tuple, Optional

from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import google.generativeai as genai




log = logging.getLogger("startup")
def _mask(x): return f"{x[:4]}…{x[-4:]}" if x and len(x)>=8 else ("<empty>" if not x else "<short>")
log.info("ENV check: PINECONE_API_KEY=%s GEMINI_API_KEY=%s",
         bool(os.getenv("PINECONE_API_KEY")), bool(os.getenv("GEMINI_API_KEY")))



# --- env ---
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

def embed(text: str) -> List[float]:
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text, task_type="retrieval_query")
        return res["embedding"]
    except Exception:
        log.exception("Gemini embed failed.")
        return []

def retrieve_chunks(query: str, k: int = TOP_K, namespace: Optional[str] = None):
    idx = _ensure_pinecone()
    if idx is None:
        return []
    vec = embed(query)
    if not vec:
        return []
    try:
        out = idx.query(
            vector=vec,
            top_k=k,
            include_metadata=True,
            include_values=False,
            namespace=(namespace or "")
        )
        return [
            (m.metadata["text"], m.score)
            for m in out.matches
            if m.metadata and "text" in m.metadata
        ]
    except PineconeException:
        log.exception("Pinecone query failed.")
        return []

def answer_with_full_rag(
    question: str,
    k: int = TOP_K,
    namespace: Optional[str] = None
) -> Tuple[str, List[Tuple[str, float]]]:

    model = genai.GenerativeModel("models/gemini-2.0-flash")

    translation_prompt = f"Translate this to English. If it is already in English, REPEAT IT VERBATIM. \n Text to translate:\n{question}"

    translated_question = model.generate_content(translation_prompt).text.strip()

    chunks = retrieve_chunks(question, k, namespace)
    if not chunks:
        return "Sorry, I couldn’t reach the knowledge base right now.", []

    threshold = 0.57
    filtered = [(t, s) for t, s in chunks if s >= threshold]
    if not filtered:
        return "No relevant information found in your document store.", []

    context = "\n\n".join(t for t, _ in filtered)


        # 4. first pass summary
    summary_prompt = (
        "You are a first pass summary assistant for a blood donor question chat.\n"
        "Try to respond to the query using ONLY the context below, or say 'Not enough information' if there is uncertainty or vagueness.\n"
        "Write for a blood donor. Use very straightforward, plain language suitable for an uneducated audience. \n\n"
        f"Context:\n{context}\nQuery: {translated_question}\nAnswer (be brief):"
    )

    #If the query suggests the person is more knowledgeable, you may keep technical terms.
    first_summary = model.generate_content(summary_prompt).text.strip()
    print('SUMMARY:', first_summary)

    # 5. verification + question check pass
    verify_prompt = (
        "You are a strict verification agent in a blood donor chat pipeline.\n"
        "If the input explicitly signals insufficient info or a clarification request, output verbatim 'Not enough information' and DO NOT write anything more.\n"
        "DO NOT overextend yourself to report an answer if the draft does not address the query.\n"
        "If the answer does address the query, make minor edits to the draft answer to ensure that 1) the answer is fully supported from the context alone and 2) it is very relevant to the query; prefer straightforward wording for an uneducated audience, unless the query indicates the person is knowledgeable, in which case technical terms are acceptable.\n\n"
        f"Context:\n{context}\nQuestion: {translated_question}\n\n"
        f"Answer to modify:\n{first_summary}\nRevised answer (be brief):"
    )
    verified_summary = model.generate_content(verify_prompt).text.strip()
    print('VERIFIED:', verified_summary)

    # 6. safety check pass
    safety_prompt = (
        "You are a cautious safety check agent in a blood donor chat pipeline.\n"
        "If you see 'Not enough information' or an explicit clarification request, output the same words verbatim and DO NOT GENERATE ANYTHING ELSE.\n"
        "Otherwise, make small adjustments to the draft by moderating absolutes and using reasonably cautious language. \n"
        "Keep wording straightforward for an uneducated audience \n\n"
        f"Context:\n{context}\nAnswer to modify:\n{verified_summary}\nRevised answer (be brief):"
    )
    safe_summary = model.generate_content(safety_prompt).text.strip()
    print('SAFE:', safe_summary)

    # 7. final formatting pass
    formatting_prompt = (
        "You are a formatting assistant for a blood donor chatbot. Take the following answer and\n"
        "convert it into exactly what the blood donor should read, no internal notes,\n"
        "no references to the agent, just a clear and concise donor facing response. Minimize wordiness.\n"
        "Use very straightforward language for an uneducated audience; for ALL technical terms that a villager might not understand, add parentheses with a few-word explanation / synonym. \n"
        "If the answer says 'Not enough information', kindly tell the donor that you do not have enough information, and apologize, but do not request more info.\n"
        f"Finally, look back at the question and add any greetings/niceties such as 'Hello'/'Thank You'/'Good Afternoon/etc. : {translated_question} \n"
        f"Verified answer:\n{safe_summary}\n\nFormatted answer (be brief):"
    )

    formatted_answer = model.generate_content(formatting_prompt).text.strip()

    # 8. final translation pass
    final_translation = (
        "You are a language consistency agent for a blood donor chatbot. I need to make sure the user's question is in the SAME language as our answer."
        f"(Silently) figure out what language the bulk of this question (NOT just the greeting or a select few words) is most likely in: {question}. If it is in multiple langauges, take the most predominant language in the bulk of the question.  \n"
        f"It is probably English, but if not, it is potentially Swahili or French, based on our user base. Only if it is NOT English, translate my English answer back to the other language you identified. Make sure that your final answer is in the SAME LANGUAGE as the initial question, but exactly the same information. \n\n My English answer:\n{formatted_answer}\n\nWhat is the answer for the patient? ONLY output the pure, patient-facing answer, no other words or extraneous output (such as anything about translation). "
    )

    final_answer = model.generate_content(final_translation).text.strip()

    # 8. return the final answer plus the list of (chunk, score) pairs
    return final_answer, filtered
