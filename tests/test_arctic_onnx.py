from pathlib import Path

import numpy as np

from src.rag.arctic_onnx import (
    ArcticOnnxEmbedder,
    AndroidWordPieceTokenizer,
    DIM,
    MAX_LEN,
    chunk_words,
)

MODEL = Path("models/arctic.onnx")
VOCAB = Path("models/vocab.txt")


def test_tokenizer_emits_android_shaped_int64_inputs():
    encoding = AndroidWordPieceTokenizer(VOCAB).encode("Differentiate x^2.")

    assert encoding.input_ids.shape == (MAX_LEN,)
    assert encoding.attention_mask.shape == (MAX_LEN,)
    assert encoding.token_type_ids.shape == (MAX_LEN,)
    assert encoding.input_ids.dtype == np.int64
    assert encoding.attention_mask.sum() > 2
    assert not encoding.token_type_ids.any()


def test_android_style_chunking_is_deterministic_and_overlapping():
    text = " ".join(f"word{index}" for index in range(900))
    chunks = chunk_words(text, start_index=10, start_page=1, end_page=20)

    assert [chunk.index for chunk in chunks] == [10, 11, 12]
    assert len(chunks[0].text.split()) == 400
    assert chunks[0].text.split()[-50:] == chunks[1].text.split()[:50]


def test_onnx_batch_and_single_embeddings_match_and_are_normalized():
    embedder = ArcticOnnxEmbedder(MODEL, VOCAB, intra_op_threads=2)
    texts = ["What is a derivative?", "Integration adds small quantities."]

    batch = embedder.embed_batch(texts)
    singles = np.stack([embedder.embed(text) for text in texts])

    assert batch.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(singles, axis=1), 1.0, atol=1e-5)
    # Dynamic INT8 quantization produces small batch-dependent drift.
    assert np.all(np.sum(batch * singles, axis=1) > 0.995)


def test_query_prefix_changes_embedding():
    embedder = ArcticOnnxEmbedder(MODEL, VOCAB, intra_op_threads=2)
    text = "What is differentiation?"

    passage = embedder.embed(text)
    query = embedder.embed(text, is_query=True)

    assert not np.allclose(query, passage, atol=1e-5)
