"""Android-compatible Snowflake Arctic Embed inference using ONNX Runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from pypdf import PdfReader

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 128
DIM = 384
CHUNK_WORDS = 400
OVERLAP_WORDS = 50
PAGE_WINDOW = 20
PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@dataclass(frozen=True)
class Encoding:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray


class AndroidWordPieceTokenizer:
    """Port of android-mnn's minimal WordPieceTokenizer."""

    def __init__(self, vocab_path: str | Path):
        tokens = Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.cls_id = self._required("[CLS]")
        self.sep_id = self._required("[SEP]")
        self.unk_id = self._required("[UNK]")
        self.pad_id = self._required("[PAD]")

    def _required(self, token: str) -> int:
        if token not in self.vocab:
            raise ValueError(f"Vocabulary is missing {token}")
        return self.vocab[token]

    def encode(self, text: str, max_len: int = MAX_LEN) -> Encoding:
        token_ids = self._word_piece(text.lower().strip())[: max_len - 2]
        ids = np.full(max_len, self.pad_id, dtype=np.int64)
        mask = np.zeros(max_len, dtype=np.int64)
        token_types = np.zeros(max_len, dtype=np.int64)
        ids[0] = self.cls_id
        mask[0] = 1
        if token_ids:
            ids[1 : len(token_ids) + 1] = token_ids
            mask[1 : len(token_ids) + 1] = 1
        sep_position = len(token_ids) + 1
        ids[sep_position] = self.sep_id
        mask[sep_position] = 1
        return Encoding(ids, mask, token_types)

    def _word_piece(self, text: str) -> list[int]:
        result: list[int] = []
        for word in self._split_on_punctuation(text):
            if not word:
                continue
            if len(word) > 100:
                result.append(self.unk_id)
                continue
            start = 0
            pieces: list[int] = []
            while start < len(word):
                found_id: int | None = None
                found_end = start
                for end in range(len(word), start, -1):
                    piece = word[start:end] if start == 0 else f"##{word[start:end]}"
                    if piece in self.vocab:
                        found_id = self.vocab[piece]
                        found_end = end
                        break
                if found_id is None:
                    pieces = [self.unk_id]
                    break
                pieces.append(found_id)
                start = found_end
            result.extend(pieces)
        return result

    @staticmethod
    def _split_on_punctuation(text: str) -> list[str]:
        tokens: list[str] = []
        buffer: list[str] = []
        for char in text:
            if char.isspace():
                if buffer:
                    tokens.append("".join(buffer))
                    buffer.clear()
            elif char in PUNCTUATION:
                if buffer:
                    tokens.append("".join(buffer))
                    buffer.clear()
                tokens.append(char)
            else:
                buffer.append(char)
        if buffer:
            tokens.append("".join(buffer))
        return tokens


class ArcticOnnxEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        intra_op_threads: int | None = None,
    ):
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AndroidWordPieceTokenizer(vocab_path)
        inputs = {item.name for item in self.session.get_inputs()}
        expected = {"input_ids", "attention_mask", "token_type_ids"}
        if inputs != expected:
            raise ValueError(f"Unexpected ONNX inputs: {sorted(inputs)}")

    def embed(self, text: str, *, is_query: bool = False) -> np.ndarray:
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(
        self,
        texts: Iterable[str],
        *,
        is_query: bool = False,
    ) -> np.ndarray:
        values = list(texts)
        if not values:
            raise ValueError("texts must not be empty")
        if is_query:
            values = [QUERY_PREFIX + value for value in values]
        encodings = [self.tokenizer.encode(value) for value in values]
        ids = np.stack([value.input_ids for value in encodings])
        masks = np.stack([value.attention_mask for value in encodings])
        types = np.stack([value.token_type_ids for value in encodings])
        hidden = self.session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": masks,
                "token_type_ids": types,
            },
        )[0]
        if hidden.shape != (len(values), MAX_LEN, DIM):
            raise ValueError(f"Unexpected ONNX output shape: {hidden.shape}")
        weights = masks[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-9)).astype(np.float32)


@dataclass(frozen=True)
class Chunk:
    index: int
    start_page: int
    end_page: int
    text: str


def chunk_words(text: str, *, start_index: int, start_page: int, end_page: int) -> list[Chunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    words: list[str] = []
    for paragraph in paragraphs:
        words.extend(re.split(r"\s+", paragraph))
        words.append("\0")
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end]).replace(" \0 ", "\n\n").replace("\0", "").strip()
        if chunk_text:
            chunks.append(Chunk(start_index + len(chunks), start_page, end_page, chunk_text))
        if end >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def extract_android_style_chunks(pdf_path: str | Path) -> list[Chunk]:
    """Approximate Android's 20-page PDFBox windows using local pypdf text."""
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    for zero_start in range(0, len(reader.pages), PAGE_WINDOW):
        zero_end = min(zero_start + PAGE_WINDOW, len(reader.pages))
        text = "\n".join(reader.pages[index].extract_text() or "" for index in range(zero_start, zero_end))
        chunks.extend(
            chunk_words(
                text,
                start_index=len(chunks),
                start_page=zero_start + 1,
                end_page=zero_end,
            )
        )
    return chunks


def cosine_top_k(query: np.ndarray, passages: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    scores = passages @ query
    order = np.argsort(-scores, kind="stable")[:k]
    return order, scores[order]
