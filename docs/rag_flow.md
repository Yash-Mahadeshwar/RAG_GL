# RAG pipeline diagrams

GitHub renders these Mermaid diagrams automatically. In VS Code, install the
"Markdown Preview Mermaid Support" extension.

Every RAG system has two phases:
- **Indexing** (offline, once per document): turn documents into searchable vectors.
- **Querying** (online, every question): find the relevant pieces and answer from them.

## 1. Naive RAG (step 04)

What you build by hand in plain Python.

```mermaid
flowchart LR
    subgraph IDX["Indexing (once per PDF)"]
        direction LR
        PDF[/"PDF"/] --> LOAD["load_pdf_pages()<br/>pypdf"]
        LOAD --> CHUNK["chunk_pages()<br/>fixed 800-char windows"]
        CHUNK --> EMB1["embed_texts()<br/>text-embedding-3-small"]
        EMB1 --> DB[("Chroma<br/>./chroma_db")]
    end

    subgraph QRY["Querying (every question)"]
        direction LR
        Q[/"Question"/] --> EMB2["embed question<br/>same model!"]
        EMB2 --> SEARCH["collection.query()<br/>top-4 by cosine distance"]
        DB -.-> SEARCH
        SEARCH --> PROMPT["build_prompt()<br/>instruction + context + question"]
        Q --> PROMPT
        PROMPT --> LLM["generate_answer()<br/>gpt-4o-mini"]
        LLM --> A[/"Answer"/]
    end
```

## 2. Improved RAG (after step 05)

Each upgraded stage is labelled with the sub-step that introduces it. The
dashed hybrid branch is the theory from 5d and isn't implemented in the main
scripts.

```mermaid
flowchart LR
    subgraph IDX["Indexing"]
        direction LR
        PDF[/"PDF"/] --> DOCS["Documents<br/>text + page metadata"]
        DOCS --> SPLIT["<b>5a</b> RecursiveCharacterTextSplitter<br/>cuts at paragraphs / lines / words"]
        SPLIT --> EMB1["OpenAIEmbeddings"]
        EMB1 --> DB[("Chroma")]
        SPLIT -.-> BM25[("<b>5d</b> BM25 index<br/>(optional)")]
    end

    subgraph QRY["Querying"]
        direction LR
        Q[/"Question"/] --> RET["Vector search<br/>over-fetch top 10-20"]
        DB -.-> RET
        Q -.-> KW["<b>5d</b> keyword search<br/>(optional)"]
        BM25 -.-> KW
        RET --> FUSE{{"<b>5d</b> RRF fusion<br/>(optional)"}}
        KW -.-> FUSE
        FUSE --> RR["<b>5b</b> Cross-encoder reranker<br/>FlashRank, keep top 3-4"]
        RR --> FMT["format_docs()<br/>[p. N] labels"]
        FMT --> PROMPT["<b>5c</b> Strict system prompt<br/>context only, cite pages,<br/>exact 'I don't know'"]
        Q --> PROMPT
        PROMPT --> LLM["ChatOpenAI<br/>gpt-4o-mini, temp 0"]
        LLM --> A[/"Answer + citations"/]
    end

    A -.-> EVAL["<b>06</b> Evaluation<br/>hit-rate@k + LLM judge"]
    RR -.-> EVAL
```

## How the steps map onto the pipeline

| Pipeline stage | Naive (04) | Improved (05+) | How we measure it (06) |
|---|---|---|---|
| Chunk | Fixed-size windows | Recursive splitter | Indirectly, via hit-rate |
| Retrieve | Top-k vector search | Over-fetch + rerank (+ hybrid) | **Hit-rate@k** |
| Prompt | "Use only the context" | Rules, citations, exact refusal | **Answer accuracy** |
| Generate | `gpt-4o-mini` | `gpt-4o-mini`, temperature 0 | **Answer accuracy** |
