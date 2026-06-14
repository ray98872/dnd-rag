"""
Prompt templates for the D&D RAG system.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── System persona ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are a precise and knowledgeable Dungeon Master's assistant, expert in the \
D&D 5e 2024 ruleset (SRD 5.2).

Rules:
- Answer ONLY using the provided context passages from the SRD.
- If the context does not contain enough information to answer fully, say so clearly \
  rather than guessing or drawing on knowledge outside the provided text.
- Quote relevant rule text verbatim when it strengthens the answer.
- Be concise and structured. Use bullet points or numbered lists for multi-part answers.
- Always cite which SRD section or page your answer comes from.
"""

# ── Main RAG prompt ────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        (
            "human",
            "Context from the D&D 5e 2024 SRD:\n\n{context}\n\n---\n\nQuestion: {question}",
        ),
    ]
)

# ── HyDE prompt (Hypothetical Document Embeddings) ─────────────────────────────
HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a D&D 5e 2024 rules expert. Write a short, factual passage "
            "(3-5 sentences) that WOULD appear in the D&D SRD and directly answers "
            "the following question. Write only the passage text, no preamble.",
        ),
        ("human", "{question}"),
    ]
)

# ── Condensation prompt (for multi-turn, optional) ─────────────────────────────
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        (
            "human",
            "Given the conversation above, rephrase the follow-up question into a "
            "standalone question that captures all necessary context.",
        ),
    ]
)
