"""Ancora - grounded answers, or an honest escalation. Never a guess.

A small, dependency-free verification layer that sits between a RAG
pipeline and your users. It checks every sentence a model produces against
the retrieved evidence and refuses to deliver anything it cannot trace back
to a source - with a hard guard on numbers, because an invented price or
deadline is the hallucination that actually costs money.

    from ancora import Ancora, AnthropicProvider

    kb = Ancora.from_directory("./knowledge-base")
    answer = kb.ask("Qual o prazo de entrega?", provider=AnthropicProvider())

    if answer.is_answerable:
        send(answer.text)          # every claim traced to a source
    else:
        handoff(answer.draft)      # a human sees what the model wanted to say
"""

from .chunking import chunk_document, chunk_documents
from .engine import Ancora
from .grounding import GroundingChecker, is_checkable_claim
from .policy import Policy, decide
from .providers import (
    AnthropicProvider,
    EchoProvider,
    OpenAIProvider,
    Provider,
    ScriptedProvider,
    build_prompt,
)
from .retrieval import BM25Index, Embedder, Retriever
from .types import Action, Answer, Chunk, ClaimCheck, Document, ScoredChunk, Verdict

__version__ = "0.1.1"

__all__ = [
    "Ancora",
    "Action",
    "Answer",
    "AnthropicProvider",
    "BM25Index",
    "Chunk",
    "ClaimCheck",
    "Document",
    "EchoProvider",
    "Embedder",
    "GroundingChecker",
    "OpenAIProvider",
    "Policy",
    "Provider",
    "Retriever",
    "ScoredChunk",
    "ScriptedProvider",
    "Verdict",
    "build_prompt",
    "chunk_document",
    "chunk_documents",
    "decide",
    "is_checkable_claim",
    "__version__",
]
