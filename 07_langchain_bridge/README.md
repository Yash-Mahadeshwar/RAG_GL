# Step 07 - The LangChain bridge (~10 min)

## What we're building
The complete RAG pipeline (load, split, embed, store, retrieve, prompt,
generate) as one LCEL chain in about 30 lines, with streaming output. Then
we point at what comes next.

## Why this step exists
To close the loop: **LangChain wraps what you built by hand.** Read
`build_rag_chain()` and every line has a comment naming the step-04 function
it replaces. The only genuinely new idea is the LCEL pipe, which turns separate
function calls into one composable object. That object gets `.invoke()`,
`.batch()`, `.stream()` and async versions for free, and plugs into tracing
tools. When you read LangChain code in tutorials or at work, you'll now see
the plain-Python pipeline underneath.

## What to fill in
- **TODO 1**: `vectorstore.as_retriever(search_kwargs={"k": 4})`.
- **TODO 2**: compose `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
- **TODO 3**: stream the answer with `rag_chain.stream(question)`.

## How to run it
```bash
python 07_langchain_bridge/starter.py
python 07_langchain_bridge/starter.py -q "Why is self-attention faster than recurrence?"
```

## What success looks like
The answer appears word by word as it's generated, with page citations:
```
> What optimizer did they use?
They used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ε = 10^-9, with a
learning rate that warms up over 4000 steps and then decays [p. 7].
```

## How the pipe works
```python
{"context": retriever | format_docs, "question": RunnablePassthrough()}
```
A dict in LCEL means "run every value on the same input, in parallel, and
collect the results". The question string goes into both branches:
- `retriever | format_docs`: question → list of Documents → one context string
- `RunnablePassthrough()`: question → question (unchanged)

The output `{"context": "...", "question": "..."}` is exactly what the prompt
template's `{context}` and `{question}` placeholders need. After that it's
`prompt | llm | StrOutputParser()`, which you wrote in 5c.

## Where to go next: LangGraph and agents
Everything today was a **fixed pipeline**: every question goes through the
same steps once. Real assistants need **decisions**:
- *Does this question even need retrieval?* ("Hi!" doesn't.)
- *The retrieved chunks look irrelevant, so should I rewrite the query and try again?*
- *This needs a calculator, a SQL query or a web search, not my PDF.*

That's an **agent**: an LLM in a loop that chooses its next action. **[LangGraph](https://langchain-ai.github.io/langgraph/)**
is LangChain's library for building them as explicit graphs of steps with
state, branches and loops. Your retriever becomes one *tool* the agent can
call. A good first project is "agentic RAG": an agent that decides whether to
retrieve, grades the retrieved chunks, and rewrites the query if they're bad.

Other directions are in the root README under "Where to go next".

## Peek-ahead
Wrap the whole chain in `return ( ... )` so it can span multiple lines.
Still stuck? → [`solutions/07_langchain_bridge.py`](../solutions/07_langchain_bridge.py)
