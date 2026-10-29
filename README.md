# Local PDF RAG (Gemini Free Tier + ChromaDB + Streamlit)

A real, end-to-end Retrieval-Augmented Generation (RAG) application that lets you
upload one or more PDFs, builds a local vector index of their content, and answers
your questions using **only** text retrieved from those PDFs — grounded, cited,
and inspectable. Runs entirely on the Gemini API **free tier**, with a fully
local vector database (no cloud account, no billing).

---

## 1. Project overview

**What RAG is.** Retrieval-Augmented Generation combines a search step
(retrieval) with a generation step (an LLM writing the answer). Instead of
asking an LLM to answer purely from what it memorized during training, you
first find the most relevant pieces of a document, then hand *only those
pieces* to the LLM as context and ask it to answer using them.

**What this app does.**

1. You upload a PDF.
2. The app extracts and cleans the text, page by page.
3. The text is split into overlapping chunks.
4. Each chunk is turned into a vector embedding via the Gemini embedding API.
5. Embeddings are stored in a local ChromaDB vector database.
6. When you ask a question, the app embeds the question, searches the vector
   database for the most similar chunks, and sends only those chunks (not the
   whole PDF) to Gemini.
7. Gemini answers **strictly from that retrieved context** and the app shows
   you the answer plus exactly which PDF pages it came from.

This is a genuine RAG pipeline — not "paste the whole PDF into the prompt."

---

## 2. Architecture

```
PDF
 ↓
Text Extraction         (pypdf, page by page)
 ↓
Cleaning                 (whitespace/noise normalization)
 ↓
Chunking                 (paragraph/sentence-aware, page-bounded, overlapping)
 ↓
Embeddings                (Gemini embedding model, batched + retried)
 ↓
ChromaDB                  (local, persistent vector store)
 ↓
Question Embedding
 ↓
Similarity Search          (cosine similarity, top-k, threshold filter)
 ↓
Top-K Context
 ↓
Gemini                    (grounded generation, low temperature)
 ↓
Grounded Answer + Sources
```

### Project structure

```
rag_pdf_app/
│
├── app.py              # Streamlit GUI — wires everything together
├── config.py           # All tunable settings in one place
├── rag_pipeline.py      # Orchestrates ingestion + querying, owns the cache
├── pdf_processor.py     # Stage A: extraction, cleaning, scanned-PDF detection
├── chunking.py           # Stage B: paragraph/sentence-aware chunking
├── embeddings.py          # Stage C: Gemini embeddings, batching, retry
├── vector_store.py         # Stage D: ChromaDB wrapper
├── llm.py                   # Stage E: grounded generation via Gemini
├── prompts.py                 # System prompt + context-building
├── utils.py                    # Logging, hashing, retry/backoff
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   ├── documents/        # Copies of indexed PDFs
│   ├── vector_db/         # ChromaDB's persistent files
│   └── cache/              # Document-hash cache (skip re-embedding)
│
└── tests/
    └── test_rag.py          # Offline unit tests (no API key required)
```

### Why these chunking numbers?

Chunk size defaults to **900 words** (~1100-1300 tokens for English), inside the
requested 800-1200 token range with headroom for tokenizer variance. Overlap
defaults to **150 words** (~15%): enough that a sentence near a chunk boundary
is very likely intact in at least one chunk, without meaningfully inflating the
embedding cost. Chunks never cross a page boundary, so every chunk's page
citation is exact — no guessing which page a chunk "really" belongs to.

---

## 3. Installation (Windows 10/11, Python 3.10+)

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

(On macOS/Linux, activate with `source venv/bin/activate` instead.)

---

## 4. Getting a Gemini API key (free, no billing)

1. Go to **Google AI Studio**: https://aistudio.google.com/
2. Sign in with a Google account.
3. Click **"Get API key"** / **"Create API key"**.
4. Copy the generated key. You do **not** need to enable billing or attach a
   credit card to use the free tier.

## 5. Configure your API key

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
   (macOS/Linux: `cp .env.example .env`)
2. Open `.env` and replace `YOUR_API_KEY_HERE` with your real key:
   ```env
   GEMINI_API_KEY=your-real-key-here
   ```
3. Never commit `.env` — it's already listed in `.gitignore`.

You can also override the model names, chunk size, top-k, etc. by uncommenting
the relevant lines in `.env` — see `config.py` for the full list of settings.

### About the model names

- **LLM**: defaults to `gemini-3.1-flash-lite` — a stable, free-tier-eligible
  model with generous free-tier request limits.
- **Embeddings**: defaults to `gemini-embedding-001` — the current free-tier
  embedding model.
- **Model names change over time.** Before relying on this in production,
  double-check current model availability and free-tier limits at
  https://ai.google.dev/gemini-api/docs/models and
  https://ai.google.dev/gemini-api/docs/rate-limits, and update `GEMINI_LLM_MODEL`
  / `GEMINI_EMBEDDING_MODEL` in `.env` if needed — you never have to edit
  Python source to change models.

---

## 6. Running the application

```bash
streamlit run app.py
```

This opens a browser tab (usually at `http://localhost:8501`). What happens
next:

1. **Upload a PDF** using the sidebar file uploader, then click **"Upload &
   Index"**. The app extracts text, chunks it, embeds every chunk, and stores
   the embeddings in the local `data/vector_db/` folder. Progress is shown
   live.
2. Once indexing finishes, the **Document(s)** section shows the filename and
   chunk count.
3. Type a question in **"Ask a question about your PDF..."** and click
   **Ask**. The app retrieves the most relevant chunks and asks Gemini to
   answer using only those chunks.
4. The **Answer** section shows the grounded answer and a **Sources** list
   (filename + page number for every page the answer drew from).
5. The **Retrieval debugging** section lets you expand each retrieved chunk
   to see its rank, page, similarity score, and full text — this is what
   proves the system is actually retrieving, not just chatting.
6. Use **"Clear Knowledge Base"** in the sidebar to wipe everything and start
   over with different PDFs.

Re-uploading the same unchanged PDF skips re-embedding automatically (the app
hashes file contents and reuses the existing index).

---

## 7. Handling Gemini Free Tier limits

The app is built to stay inside the free tier and to fail gracefully if you
hit it:

- Requests are batched and paced (a short pause between embedding batches).
- Every Gemini call is wrapped in retry-with-exponential-backoff logic.
- If the quota/rate limit is hit repeatedly, you'll see:

  > "Gemini Free Tier quota/rate limit reached. Please wait a minute and try
  > again, or reduce the number of requests (fewer/smaller PDFs, fewer
  > questions in quick succession)."

- The app **never** switches to a paid model or paid fallback automatically.
  If you outgrow the free tier, that's a decision you make explicitly (e.g.
  enabling billing in Google Cloud), never something this code does for you.

---

## 8. Test plan

Offline tests (`tests/test_rag.py`, no API key needed) cover chunking
behavior, scanned-PDF detection, and the document-hash cache. Run them with:

```bash
pip install pytest
pytest tests/ -v
```

Manual tests (require a real `GEMINI_API_KEY` and an uploaded PDF):

| # | Test | How to run | Expected result |
|---|------|-----------|------------------|
| 1 | Direct question | Ask something whose exact answer is stated in the PDF | Correct answer, with the right source page cited |
| 2 | Paraphrased question | Ask the same thing in different words | Same correct answer — proves retrieval is semantic, not keyword matching |
| 3 | Multi-step question | Ask something that needs info from two different sections/pages | Answer draws on multiple chunks; multiple pages appear in Sources |
| 4 | Out-of-scope question | Ask something not in the PDF at all | "I could not find sufficient information about this in the uploaded PDF." |
| 5 | Source verification | Open the Retrieval debugging section after any answer | The cited page/chunk text actually contains the information used in the answer |

---

## 9. RAG evaluation (lightweight, manual)

You don't need an expensive evaluation framework to sanity-check this system.
Four concepts to check by hand while testing:

- **Retrieval relevance** — Do the chunks shown in "Retrieval debugging"
  actually relate to the question? If not, try increasing Top-K or rephrasing.
- **Context relevance** — Of the retrieved chunks, how much of their content
  is actually useful for answering (vs. tangential)? Lower relevance suggests
  the chunk size/overlap may need tuning.
- **Faithfulness / groundedness** — Does every claim in the answer trace back
  to something in the retrieved chunks? Cross-check the answer against the
  expanded chunk text.
- **Answer correctness** — Is the answer actually right, given the source
  material?

**How to know it's really RAG and not just Gemini's own knowledge:** ask
something specific to your PDF that Gemini couldn't plausibly know on its
own (a made-up product name, an internal figure, a specific number from a
table). If the answer is correct and cites the right page, retrieval is
doing real work. If you clear the knowledge base and ask the same question,
you should get "I could not find sufficient information..." — proving the
answer really depended on the indexed PDF, not general knowledge.

---

## 10. Multiple PDFs

Upload additional PDFs from the sidebar the same way — each is chunked and
embedded independently, tagged with its own filename in every chunk's
metadata. Questions search across all indexed PDFs by default, and the
Sources list shows exactly which file (and page) each answer drew from.

---

## 11. Key concepts explained simply

- **RAG**: search first, then generate an answer using only what was found.
- **Why RAG is needed**: LLMs can't know about your private/new/large
  documents unless you show them the relevant parts; RAG is how you show them
  the relevant parts without exceeding context limits or paying to
  re-process the whole document on every question.
- **Embeddings**: a way of turning text into a list of numbers (a vector)
  such that texts with similar meaning end up with similar vectors.
- **Chunking**: splitting a long document into smaller pieces small enough to
  embed and retrieve individually, while trying not to cut sentences/ideas in
  half.
- **Vector database**: a specialized store that can quickly find the vectors
  most similar to a given query vector — this app uses ChromaDB, which runs
  entirely on your machine.
- **Semantic search**: finding text based on *meaning* similarity (via
  embeddings), not just exact keyword matches.
- **Top-K retrieval**: only take the K most similar chunks, instead of every
  chunk in the database, to keep the context focused and small.
- **What's sent to Gemini**: the question plus the text of the top-K
  retrieved chunks (with their source/page labels) — never the entire PDF.
- **Why this differs from a normal chatbot**: a plain chatbot answers from
  what it memorized during training; this app answers from text it just
  retrieved from your specific PDF, and is instructed to say "not found" when
  the PDF doesn't contain the answer.
- **Where hallucinations can still happen**: if retrieval returns weak or
  irrelevant chunks (e.g. Top-K too low, or the question isn't well covered
  by the PDF), the model may still try to answer loosely from those chunks —
  this is why the Retrieval debugging panel and the "not found" instruction
  in the prompt matter.
- **How retrieved chunks improve answer quality**: they narrow the model's
  attention to exactly the relevant material, which reduces the chance of it
  drifting into unrelated or memorized (and possibly wrong) information.
