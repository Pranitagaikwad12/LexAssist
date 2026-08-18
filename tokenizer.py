import re
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class BPEClauseTokenizer:
    """
    BPE-aware tokenizer & clause chunker for legal documents.
    Splits legal documents (contracts, statutes, judgments) into clause/section aware chunks
    respecting maximum token budgets without cutting mid-clause.
    """
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        if _TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Counts BPE tokens in text with fallback heuristic (~4 chars per token)."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Fallback approximation
        return max(1, len(text) // 4)

    def extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts structured clauses and sections from raw legal text using structural patterns.
        """
        # Patterns for legal clauses, sections, articles, and numbered items
        clause_pattern = re.compile(
            r'(?=\n(?:'
            r'SECTION\s+\d+|ARTICLE\s+[IVXLCDM\d]+|CLAUSE\s+\d+(?:\.\d+)*|'
            r'\d+\.\d+\s+|'
            r'(?:[A-Z][A-Za-z0-9\s,\-]{2,40}):\n|'
            r'\b(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF|GOVERNING LAW|TERMINATION|INDEMNIFICATION|LIABILITY|CONFIDENTIALITY|FORCE MAJEURE)\b'
            r'))',
            re.IGNORECASE
        )

        raw_parts = clause_pattern.split(text)
        clauses = []

        for idx, part in enumerate(raw_parts):
            clean_part = part.strip()
            if not clean_part:
                continue

            # Extract title / heading if present
            lines = clean_part.split('\n')
            first_line = lines[0].strip()
            
            # Simple title heuristic
            title = first_line[:80] if len(first_line) > 0 else f"Clause {idx + 1}"

            clauses.append({
                "clause_id": f"clause_{idx + 1}",
                "title": title,
                "text": clean_part,
                "token_count": self.count_tokens(clean_part)
            })

        # Fallback if no distinct clause pattern matched (e.g. raw unstructured text)
        if not clauses and text.strip():
            # Paragraph based split
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs):
                clauses.append({
                    "clause_id": f"clause_{idx + 1}",
                    "title": f"Paragraph {idx + 1}",
                    "text": p,
                    "token_count": self.count_tokens(p)
                })

        return clauses

    def chunk_by_token_budget(
        self, text: str, max_tokens: int = 500, overlap_tokens: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks respecting clause boundaries and max token budget.
        Never cuts mid-clause unless a single clause exceeds max_tokens.
        """
        clauses = self.extract_clauses(text)
        chunks = []
        current_chunk_clauses = []
        current_token_count = 0
        chunk_index = 0

        for clause in clauses:
            c_tokens = clause["token_count"]

            # If a single clause is bigger than max_tokens, split it by sentence boundary
            if c_tokens > max_tokens:
                # Flush existing buffer first
                if current_chunk_clauses:
                    chunk_text = "\n\n".join([c["text"] for c in current_chunk_clauses])
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_index}",
                        "title": current_chunk_clauses[0]["title"],
                        "text": chunk_text,
                        "token_count": self.count_tokens(chunk_text),
                        "clause_ids": [c["clause_id"] for c in current_chunk_clauses]
                    })
                    chunk_index += 1
                    current_chunk_clauses = []
                    current_token_count = 0

                # Sentence split long clause
                sentences = re.split(r'(?<=[.!?])\s+', clause["text"])
                sub_text = ""
                sub_tokens = 0
                sub_ids = []

                for s in sentences:
                    st_tokens = self.count_tokens(s)
                    if sub_tokens + st_tokens > max_tokens and sub_text:
                        chunks.append({
                            "chunk_id": f"chunk_{chunk_index}",
                            "title": clause["title"] + " (Part)",
                            "text": sub_text.strip(),
                            "token_count": self.count_tokens(sub_text),
                            "clause_ids": [clause["clause_id"]]
                        })
                        chunk_index += 1
                        sub_text = s + " "
                        sub_tokens = st_tokens
                    else:
                        sub_text += s + " "
                        sub_tokens += st_tokens

                if sub_text.strip():
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_index}",
                        "title": clause["title"] + " (End)",
                        "text": sub_text.strip(),
                        "token_count": self.count_tokens(sub_text),
                        "clause_ids": [clause["clause_id"]]
                    })
                    chunk_index += 1

            elif current_token_count + c_tokens > max_tokens:
                # Emit current buffer and start new chunk
                chunk_text = "\n\n".join([c["text"] for c in current_chunk_clauses])
                chunks.append({
                    "chunk_id": f"chunk_{chunk_index}",
                    "title": current_chunk_clauses[0]["title"],
                    "text": chunk_text,
                    "token_count": self.count_tokens(chunk_text),
                    "clause_ids": [c["clause_id"] for c in current_chunk_clauses]
                })
                chunk_index += 1
                current_chunk_clauses = [clause]
                current_token_count = c_tokens
            else:
                current_chunk_clauses.append(clause)
                current_token_count += c_tokens

        if current_chunk_clauses:
            chunk_text = "\n\n".join([c["text"] for c in current_chunk_clauses])
            chunks.append({
                "chunk_id": f"chunk_{chunk_index}",
                "title": current_chunk_clauses[0]["title"],
                "text": chunk_text,
                "token_count": self.count_tokens(chunk_text),
                "clause_ids": [c["clause_id"] for c in current_chunk_clauses]
            })

        return chunks
