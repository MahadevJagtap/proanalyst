"""Check what text the PDF actually contains around rate limits and OAuth tokens."""
import sys
sys.path.insert(0, ".")
from src.config import config
from langchain_community.document_loaders import PyPDFLoader

pdf_path = config.data_dir / "API Documentation Partial.pdf"
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

full_text = " ".join(p.page_content for p in pages).lower()

# Search for rate limit info
print("=== RATE LIMIT SEARCH ===")
keywords = ["rate limit", "request per second", "per key", "per ip", "requests/second", "rps"]
for kw in keywords:
    idx = full_text.find(kw)
    if idx != -1:
        snippet = full_text[max(0, idx-100):idx+200]
        print(f"\nFound '{kw}' at index {idx}:")
        print(snippet)
        print("-" * 60)
    else:
        print(f"NOT FOUND: '{kw}'")

print("\n=== OAUTH TOKEN SEARCH ===")
keywords2 = ["access token", "oauth", "expires", "valid for", "token expir"]
for kw in keywords2:
    idx = full_text.find(kw)
    if idx != -1:
        snippet = full_text[max(0, idx-100):idx+200]
        print(f"\nFound '{kw}' at index {idx}:")
        print(snippet)
        print("-" * 60)

print("\n=== CLIENT CREDENTIALS SEARCH ===")
keywords3 = ["client credential", "private contract", "grant type"]
for kw in keywords3:
    idx = full_text.find(kw)
    if idx != -1:
        snippet = full_text[max(0, idx-100):idx+200]
        print(f"\nFound '{kw}' at index {idx}:")
        print(snippet)
        print("-" * 60)
    else:
        print(f"NOT FOUND: '{kw}'")
