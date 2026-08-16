pkgs = ["google.generativeai", "ragas", "openai"]
for p in pkgs:
    try:
        __import__(p.split(".")[0])
        print(f"OK  {p}")
    except ImportError:
        print(f"MISSING  {p}")
