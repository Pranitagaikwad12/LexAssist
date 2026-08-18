from typing import List, Dict, Any, Optional
from app.backbone.tokenizer import BPEClauseTokenizer


class ContextWindowPacker:
    """
    Ranks, selects, and fits retrieved legal context chunks into the LLM's context window
    under strict token budget constraints.
    """
    def __init__(self, tokenizer: Optional[BPEClauseTokenizer] = None):
        self.tokenizer = tokenizer or BPEClauseTokenizer()

    def pack_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        token_budget: int = 4000,
        system_prompt_tokens: int = 500,
        reserved_completion_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Packs retrieved chunks into context within max token limits.
        
        Args:
            retrieved_chunks: List of items with 'document', 'metadata', 'similarity_score', 'id'
            token_budget: Max overall model context limit (e.g. 4000)
            system_prompt_tokens: Reserved for system prompt
            reserved_completion_tokens: Reserved for model generation output
            
        Returns:
            Dict containing formatted context string, list of packed chunks, total tokens used, and dropped chunks count.
        """
        max_context_tokens = max(500, token_budget - system_prompt_tokens - reserved_completion_tokens)

        # Sort chunks by similarity score descending
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x.get("similarity_score", 0.0),
            reverse=True
        )

        packed_chunks = []
        formatted_blocks = []
        current_tokens = 0
        dropped_count = 0

        for chunk in sorted_chunks:
            doc_text = chunk.get("document", "")
            meta = chunk.get("metadata", {})
            
            # Format block header with citation identifier
            citation_id = meta.get("citation_id") or meta.get("clause_id") or meta.get("section") or chunk.get("id") or "Source"
            title = meta.get("title") or meta.get("heading") or "Legal Context"

            formatted_block = f"--- [CITATION: {citation_id}] ({title}) ---\n{doc_text}\n"
            block_tokens = self.tokenizer.count_tokens(formatted_block)

            if current_tokens + block_tokens <= max_context_tokens:
                packed_chunks.append({
                    **chunk,
                    "citation_id": citation_id,
                    "block_tokens": block_tokens
                })
                formatted_blocks.append(formatted_block)
                current_tokens += block_tokens
            else:
                dropped_count += 1

        packed_context_str = "\n".join(formatted_blocks)

        return {
            "formatted_context": packed_context_str,
            "packed_chunks": packed_chunks,
            "packed_chunks_count": len(packed_chunks),
            "dropped_chunks_count": dropped_count,
            "total_context_tokens": current_tokens,
            "max_context_tokens": max_context_tokens
        }
