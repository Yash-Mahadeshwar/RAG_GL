# Step 05c - Better prompting and "I don't know" (~15 min)

## What we're building
Two LCEL chains that share the same retriever and model and differ only in
their prompt. A weak prompt just pastes context and question. A strong
system prompt has explicit rules, page citations and an exact refusal phrase.
We run both on one answerable question and two the document *can't* answer.

## Why this step exists
Retrieval decides what the model *sees*; the prompt decides what it *does*
with it. With a lazy prompt, the model treats the context as a suggestion and
falls back on its own knowledge. Ask "What's the capital of Australia?" and it
says Canberra, even though your document is about Transformers. That's a
hallucination from RAG's point of view: an answer not grounded in your source.
A strict prompt fixes it, and a *fixed* refusal phrase means your code (and
step 06's evaluator) can detect "I don't know" reliably. Citations let a human
check the answer in seconds. This step also introduces **LCEL**, LangChain's
`|` pipe syntax: `prompt | llm | parser`.

## What to fill in
- **TODO 1** (`STRONG_SYSTEM_PROMPT`): write the system prompt: role, context-only rule, exact refusal phrase (`{IDK}`), citation format and length limit.
- **TODO 2** (`format_docs`): join retrieved Documents into one string, each labelled `[p. N]`.
- **TODO 3** (`main`): build `strong_chain` the same way as `weak_chain`.

## How to run it
```bash
python 05_improvements/5c_prompting/starter.py
python 05_improvements/5c_prompting/starter.py -q "Who are the authors of the paper?"
```

## What success looks like
```
Q: How many attention heads does the base model use?
  WEAK prompt:    The base model uses 8 attention heads.
  STRONG prompt:  The base model uses 8 attention heads (h = 8) [p. 5].

Q: What is the capital of Australia?
  WEAK prompt:    The capital of Australia is Canberra.          <- not from the doc!
  STRONG prompt:  I don't know based on the provided document.

Q: How many GPUs were used to train GPT-4?
  WEAK prompt:    The context does not mention GPT-4... / or a guess
  STRONG prompt:  I don't know based on the provided document.
```

**Try it:** delete rule 1 ("ONLY the context") from your prompt and rerun. Then
try a question that's *half* in the document ("What BLEU did the Transformer
get, and how does that compare to GPT-4?"). A good prompt answers the first
half and declines the second.

## Peek-ahead
`STRONG_PROMPT | llm | StrOutputParser()`. The pipe doesn't care which prompt it gets.
Still stuck? → [`solutions/05c_prompting.py`](../../solutions/05c_prompting.py)
