import math
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str):
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        vectors = [d.embedding for d in result.data]
        # Audit this outbound OpenAI call. The payload is chapter text (a real
        # PII surface); the response is vectors, so we log a shape descriptor
        # rather than the numbers. record_call is fail-open and never raises.
        from mash_core import record_call

        usage = getattr(result, "usage", None)
        record_call(
            prompt="\n".join(texts),
            response=f"[embeddings: {len(vectors)} vectors, dim {len(vectors[0]) if vectors else 0}]",
            model_id="openai:text-embedding-3-small",
            tokens_in=getattr(usage, "total_tokens", None),
            tokens_out=0,
            provider="openai",
            caller="book_mash.judges.embeddings.OpenAIEmbeddingClient.embed",
        )
        return vectors


def pick_embedding_client() -> EmbeddingClient:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return OpenAIEmbeddingClient(openai_key)
    raise RuntimeError("No embedding API key found (set OPENAI_API_KEY)")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
