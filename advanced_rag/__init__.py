"""Advanced RAG — a production-shaped Retrieval-Augmented Generation pipeline on Google Gemini.

The package is organized as a stack of independently-useful layers:

    loaders  -> chunking -> store (hybrid) -> retriever -> generate
                                   \\-> graph (GraphRAG)
                                        \\-> agent (plan/route/act/verify/stop)

See the docs/ folder for a guided tour that maps each technique to the code here.
"""

from .config import Settings, get_settings
from .pipeline import RAGPipeline

__all__ = ["Settings", "get_settings", "RAGPipeline"]
__version__ = "0.1.0"
