# src/document.py

class Document:
    def __init__(self, text, source_type, metadata=None):
        self.text = text
        self.source_type = source_type   # pdf / image / web
        self.metadata = metadata or {}