# Step 01 - Hello, LLM (~10 min)

## What we're building
A 40-line script that sends one question to `gpt-4o-mini` and prints the
answer and how many tokens it used. Nothing more.

## Why this step exists
Everything after this depends on one thing working: your code talking to a
model. Getting it out of the way first means key, billing and network problems
surface now, while there's time to fix them, rather than halfway through
building RAG. It's also your first look at the shape of an LLM call: a list of
**messages** with roles (`system`, `user`) goes in, and a **response** with
`choices` and `usage` comes out. You'll write the same call in every step
until LangChain wraps it in step 05.

## What to fill in
- **TODO 1** (`make_client`): create an `OpenAI()` client. It reads your key from the environment.
- **TODO 2** (`call_llm`): call `client.chat.completions.create(...)` with a system and a user message.
- **TODO 3** (`main`): pull the answer text out of the response object.

## How to run it
```bash
python 01_hello_llm/starter.py
python 01_hello_llm/starter.py "Explain embeddings to a 10 year old"
```

## What success looks like
```
Q: In one sentence, what is retrieval-augmented generation?

A: Retrieval-augmented generation (RAG) is a technique where a language model
   first retrieves relevant documents and then uses them to generate an answer...

[gpt-4o-mini] prompt tokens: 38, completion tokens: 41
```
The wording will differ every time. If you get a plain-English error about
your key, credits or network instead, see the Troubleshooting section of the
root README.

**Try it:** change the system message to "Answer like a pirate", or set
`temperature=1.5` and run the same question three times.

## Peek-ahead
The text is buried three levels deep: `response.choices[0].message.content`.
Still stuck? → [`solutions/01_hello_llm.py`](../solutions/01_hello_llm.py)
