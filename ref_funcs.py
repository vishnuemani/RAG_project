import google.generativeai as genai
from pinecone import Pinecone

# hard-coded keys (store securely in real use)
PINECONE_KEY = ""
GEMINI_KEY   = ""

INDEX_NAME  = "medrag"
ENVIRONMENT = "aped-4627-b74a"
EMBED_MODEL = "gemini-embedding-001"
TOP_K       = 5

# initialise clients
pc   = Pinecone(api_key=PINECONE_KEY, environment=ENVIRONMENT)
idx  = pc.Index(INDEX_NAME)
genai.configure(api_key=GEMINI_KEY)

def embed(text: str) -> list[float]:
    """Return the 3,072-dim Gemini embedding for the given text."""
    res = genai.embed_content(
        model=EMBED_MODEL,
        content=text,                 # singular “content”
        task_type="retrieval_query"   # optional but recommended
    )
    return res["embedding"]          # list[float]

def retrieve_chunks(query: str, k: int = TOP_K, namespace: str | None = None) -> list[tuple[str, float]]:
    """
    Find the top-k matching chunks for a user query, returning (text, score).
    Pass `namespace` to search within a specific namespace; defaults to the empty namespace "".
    """
    vec = embed(query)
    out = idx.query(
        vector=vec,
        top_k=k,
        include_metadata=True,
        include_values=False,
        namespace=(namespace or "")   # default namespace if None/empty
    )
    return [
        (m.metadata["text"], m.score)
        for m in out.matches
        if "text" in m.metadata
    ]


def answer_with_full_rag(question: str, k: int = TOP_K, namespace: str | None = None) -> tuple[str, list[tuple[str, float]]]:
    """RAG pipeline; returns (final_answer, filtered [(chunk, score), ...]) scoped to the given namespace."""
    # 1) retrieve context + scores (scoped by namespace)
    chunks_and_scores = retrieve_chunks(question, k, namespace=namespace)

    # 2. filter by similarity threshold
    threshold = 0.2
    filtered = [
        (text, score)
        for text, score in chunks_and_scores
        if score >= threshold
    ]
    if not filtered:
        return "No relevant information found in your document store.", []

    # optional: debug print of each retained chunk & score
    for i, (text, score) in enumerate(filtered, start=1):
        print(f"Chunk {i}: {text}")
        print(f"Score {i}: {score:.4f}\n")

    # 3. build LLM context from filtered chunks
    context = "\n\n".join(text for text, _ in filtered)

    model = genai.GenerativeModel("models/gemini-2.5-pro") #PRO MODEL INSTEAD

    # 4. first-pass summary
    summary_prompt = (

        "You are a first-pass summary assistant for a health-question chat.\n"
        f"Context:\n{context} \n"
        "Try to respond to the below query using ONLY the context above OR say 'Not enough information' if there is no relevant answer within the context. It is crucial that your answer comes from the context alone, not your own knowledge. \n"
        "Make sure your query also directly answers the question \n\n"
        f"Query: {question}\nAnswer:"
    )
    first_summary = model.generate_content(summary_prompt).text.strip()

    print("SUMMARY:", first_summary)



    # 7. final formatting pass
    formatting_prompt = (
        "You are a formatting assistant for a health chatbot. Take the following answer and\n"
        "convert it into exactly what the patient should read. no internal notes,\n"
        "no references to ‘the agent’, just the clear patient-facing response. If the answer is longer than 3-4 paragraphs, simplify it to 1-2 concise paragraphs / bullet point set. Be sure to keep the most important information relevant to the patient's query. \n"
        "If the draft answer says 'Not enough information', kindly tell the patient that you do not have enough information, and apologize, but do not request more info. You can use bullet points, but do not use formatting such as bolding or '**'."

        f"Patient's Query: {question}\n"
        f"Draft answer:\n{first_summary}\n\nFormatted answer:"
    )
    formatted_answer = model.generate_content(formatting_prompt).text.strip()

    # 8. return the final answer plus the list of (chunk, score) pairs
    return formatted_answer, filtered

