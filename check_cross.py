try:
    from sentence_transformers import CrossEncoder
    print("CrossEncoder available")
except Exception as e:
    print(f"NOT available: {e}")
