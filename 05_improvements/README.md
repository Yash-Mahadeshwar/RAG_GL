# Step 05 - Improving RAG, with LangChain (~45 min)

Your step-04 pipeline works, but every stage is naive. Here we upgrade three
of them, one per subfolder, and switch to LangChain while doing it.

| Sub-step | Time | Upgrades | Key idea |
|---|---|---|---|
| [`5a_chunking`](5a_chunking/) | 15 min | **Chunk** | Cut at natural boundaries, not every 800 characters |
| [`5b_reranking`](5b_reranking/) | 15 min | **Retrieve** | Over-fetch with vectors, then re-score with a smarter model |
| [`5c_prompting`](5c_prompting/) | 15 min | **Prompt** | Strict rules, citations, and a reliable "I don't know" |
| [`5d_hybrid_search`](5d_hybrid_search/) | read / discuss | **Retrieve** | Theory only: combine keyword search with vector search |

Each sub-step is a standalone script. Do them in order, but you don't need
one finished to start the next.

## What LangChain gives you

You've already built every piece LangChain provides. The table below shows
where each of your step-04 functions went:

| You wrote in step 04 | LangChain equivalent | Package |
|---|---|---|
| `load_pdf_pages()` → list of strings | `Document(page_content=..., metadata={...})` | `langchain-core` |
| `chunk_pages()` | `RecursiveCharacterTextSplitter(...).split_documents(docs)` | `langchain-text-splitters` |
| `client.embeddings.create(...)` in `embed_texts()` | `OpenAIEmbeddings(model=...)` | `langchain-openai` |
| `chromadb.PersistentClient` + `fresh_collection()` + `index_chunks()` | `Chroma(...)` + `.add_documents(chunks)` | `langchain-chroma` |
| `retrieve()` (embed question + `collection.query`) | `vectorstore.similarity_search(q, k)` or `.as_retriever()` | `langchain-chroma` |
| `build_prompt()` f-string | `ChatPromptTemplate.from_messages([...])` | `langchain-core` |
| `client.chat.completions.create(...)` in `generate_answer()` | `ChatOpenAI(model=...)` | `langchain-openai` |
| `response.choices[0].message.content` | `StrOutputParser()` | `langchain-core` |
| Calling these functions one after another in `answer_question()` | `prompt \| llm \| parser` (LCEL pipes) | `langchain-core` |

**What you gain:** swap any piece (Chroma → Pinecone, OpenAI → Mistral) by
changing one line; batteries-included splitters; a uniform `.invoke()`,
`.batch()` and `.stream()` on every component; tracing via LangSmith.

**What you give up:** some transparency. When something breaks, the error can
come from three layers down. That's why you built it by hand first: when the
LangChain version misbehaves, you know what it's *trying* to do.

> We load PDFs with `pypdf` directly into `Document` objects, not with
> LangChain's `PyPDFLoader`. That loader lives in `langchain-community`, which is
> being sunset and prints deprecation warnings. Our version is 4 lines and shows
> that a `Document` is just text plus a metadata dict.

## Shared index

5b, 5c, 06 and 07 reuse a persistent Chroma index built with the recursive
splitter, so the PDF is only embedded once. If you change `CHUNK_SIZE` the
collection name changes and a new index is built automatically. Use `--rebuild`
to force a re-embed.
