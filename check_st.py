try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    m = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"SentenceTransformer OK — {m.__class__.__name__}")
    c = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"CrossEncoder OK — {c.__class__.__name__}")
except Exception as e:
    print(f"FAIL: {e}")
