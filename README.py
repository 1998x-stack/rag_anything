"""
RAG-ANYTHING: Complete Usage Example
====================================

This file demonstrates end-to-end usage of the RAG-Anything framework.
"""

# =============================================================================
# SETUP AND IMPORTS
# =============================================================================

from pathlib import Path
import json

# Core components
from rag_anything.config.settings import RAGConfig
from rag_anything.utils.logger import setup_logger
from rag_anything.pipeline.indexing_pipeline import IndexingPipeline
from rag_anything.pipeline.query_pipeline import QueryPipeline

# =============================================================================
# BASIC USAGE EXAMPLE
# =============================================================================

def basic_example():
    """Basic document indexing and querying."""
    
    # 1. Setup logging
    setup_logger(
        level="INFO",
        log_file=Path("logs/rag_anything.log")
    )
    
    # 2. Load configuration
    config = RAGConfig()
    
    # Or load from YAML
    # config = RAGConfig.from_yaml(Path("config.yaml"))
    
    # 3. Create indexing pipeline
    indexer = IndexingPipeline(config)
    
    # 4. Index a document
    doc_path = Path("documents/research_paper.pdf")
    index = indexer.index_document(doc_path)
    
    print(f"✓ Indexed document with {len(index.graph.entities)} entities")
    
    # 5. Save index for later use
    index.save(Path("indexes/research_paper"))
    
    # 6. Create query pipeline
    query_pipeline = QueryPipeline(config, index)
    
    # 7. Query the document
    question = "What were the main experimental results shown in Figure 3?"
    result = query_pipeline.query(question)
    
    print(f"\n Question: {question}")
    print(f"Answer: {result.answer}")
    print(f"\nSources used: {len(result.retrieved_units)}")
    for i, retrieved in enumerate(result.retrieved_units[:3], 1):
        print(f"  {i}. {retrieved.content_unit.id} (score: {retrieved.score:.3f})")

# =============================================================================
# ADVANCED USAGE: MULTIPLE DOCUMENTS
# =============================================================================

def advanced_example():
    """Index multiple documents and perform cross-document queries."""
    
    config = RAGConfig()
    indexer = IndexingPipeline(config)
    
    # Index multiple documents
    doc_paths = [
        Path("documents/paper1.pdf"),
        Path("documents/paper2.pdf"),
        Path("documents/financial_report.pdf")
    ]
    
    index = indexer.index_multiple(doc_paths)
    
    print(f"✓ Indexed {len(doc_paths)} documents")
    print(f"  Total entities: {len(index.graph.entities)}")
    print(f"  Total relations: {len(index.graph.relations)}")
    
    # Cross-document query
    query_pipeline = QueryPipeline(config, index)
    
    questions = [
        "Compare the methodologies used in paper1 and paper2",
        "What are the financial metrics in Q3 2024?",
        "Show me all tables related to performance evaluation"
    ]
    
    for question in questions:
        result = query_pipeline.query(question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer[:200]}...")

# =============================================================================
# CONFIGURATION CUSTOMIZATION
# =============================================================================

def custom_config_example():
    """Customize configuration for specific use cases."""
    
    # Create custom configuration
    config = RAGConfig()
    
    # Modify graph settings
    config.graph.context_window_size = 5  # Larger context window
    config.graph.max_hops = 3  # Deeper graph traversal
    
    # Adjust retrieval settings
    config.retrieval.top_k_semantic = 15
    config.retrieval.top_k_structural = 10
    config.retrieval.fusion_weights = {
        "structural": 0.5,  # Prefer structural knowledge
        "semantic": 0.3,
        "modality": 0.2
    }
    
    # Configure parsers
    config.parser.image_resolution = 300  # Higher resolution
    config.parser.max_pages_per_doc = 100  # Process more pages
    
    # Save custom configuration
    config.to_yaml(Path("custom_config.yaml"))
    
    # Use custom configuration
    indexer = IndexingPipeline(config)
    # ... continue with indexing

# =============================================================================
# EVALUATION ON BENCHMARKS
# =============================================================================

def evaluate_on_docbench():
    """Evaluate on DocBench dataset."""
    
    from rag_anything.utils.metrics import compute_accuracy
    
    config = RAGConfig()
    
    # Load DocBench dataset
    docbench_path = Path("datasets/docbench")
    documents = list(docbench_path.glob("*.pdf"))
    
    # Load ground truth Q&A pairs
    with open(docbench_path / "questions.json") as f:
        qa_pairs = json.load(f)
    
    # Index all documents
    indexer = IndexingPipeline(config)
    index = indexer.index_multiple(documents)
    
    # Query pipeline
    query_pipeline = QueryPipeline(config, index)
    
    # Evaluate
    predictions = []
    ground_truths = []
    
    for qa in qa_pairs:
        question = qa["question"]
        expected_answer = qa["answer"]
        
        result = query_pipeline.query(question)
        predictions.append(result.answer)
        ground_truths.append(expected_answer)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    print(f"DocBench Accuracy: {accuracy:.2%}")

# =============================================================================
# MODALITY-SPECIFIC QUERIES
# =============================================================================

def modality_specific_examples():
    """Examples of queries targeting specific modalities."""
    
    config = RAGConfig()
    indexer = IndexingPipeline(config)
    
    doc_path = Path("documents/multimodal_paper.pdf")
    index = indexer.index_document(doc_path)
    
    query_pipeline = QueryPipeline(config, index)
    
    # Image-focused query
    question1 = "Describe the architecture diagram in Figure 2"
    result1 = query_pipeline.query(question1)
    print(f"Image Query: {result1.answer}")
    
    # Table-focused query
    question2 = "What were the accuracy values in Table 3?"
    result2 = query_pipeline.query(question2)
    print(f"Table Query: {result2.answer}")
    
    # Equation-focused query
    question3 = "Explain the loss function in Equation 5"
    result3 = query_pipeline.query(question3)
    print(f"Equation Query: {result3.answer}")

# =============================================================================
# STREAMING RESPONSES
# =============================================================================

def streaming_example():
    """Stream responses for long-running queries."""
    
    config = RAGConfig()
    # ... setup index ...
    
    query_pipeline = QueryPipeline(config, index)
    
    # Enable streaming in config
    config.vlm.streaming = True
    
    question = "Provide a comprehensive summary of all experimental results"
    
    print("Generating response...")
    for chunk in query_pipeline.query_stream(question):
        print(chunk, end="", flush=True)
    print()

# =============================================================================
# README CONTENT
# =============================================================================

README_CONTENT = """
# RAG-Anything: All-in-One RAG Framework

A unified framework for multimodal retrieval-augmented generation, enabling comprehensive knowledge retrieval across text, images, tables, and equations.

## 🌟 Features

- **Multimodal Knowledge Unification**: Process text, images, tables, and equations uniformly
- **Dual-Graph Construction**: Separate graphs for cross-modal relationships and textual semantics
- **Hybrid Retrieval**: Combine structural navigation with semantic matching
- **Superior Long-Document Performance**: Particularly effective on 100+ page documents
- **Extensible Architecture**: Easy to add new modalities and retrieval strategies

## 📦 Installation

```bash
# Basic installation
pip install rag-anything

# Or install from source
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `loguru` - Logging
- `pydantic` - Configuration management
- `openai` or `anthropic` - VLM/LLM APIs
- `networkx` - Graph operations
- `numpy` - Numerical operations
- `Pillow` - Image processing
- `faiss-cpu` or `chromadb` - Vector storage

## 🚀 Quick Start

```python
from pathlib import Path
from rag_anything import RAGConfig, IndexingPipeline, QueryPipeline
from rag_anything.utils.logger import setup_logger

# Setup
setup_logger(level="INFO")
config = RAGConfig()

# Index a document
indexer = IndexingPipeline(config)
index = indexer.index_document(Path("paper.pdf"))

# Query
query_pipeline = QueryPipeline(config, index)
result = query_pipeline.query("What are the main findings in Figure 3?")

print(result.answer)
```

## 📖 Documentation

### Configuration

Customize behavior through YAML configuration:

```yaml
models:
  vlm:
    provider: openai
    model: gpt-4o-mini
  embedding:
    model: text-embedding-3-large
    dimensions: 3072

graphs:
  context_window_size: 3
  max_hops: 2

retrieval:
  top_k_semantic: 10
  fusion_weights:
    structural: 0.4
    semantic: 0.4
    modality: 0.2
```

### Supported Modalities

- **Text**: Paragraphs, headings, sections
- **Images**: Figures, diagrams, charts, photos
- **Tables**: Data tables, comparison tables, financial reports
- **Equations**: Mathematical formulas, LaTeX expressions

### Architecture Components

1. **Knowledge Unification Layer**
   - Document parsing
   - Modality detection
   - Context extraction

2. **Dual-Graph Construction**
   - Cross-modal knowledge graph
   - Text-based knowledge graph
   - Entity alignment

3. **Hybrid Retrieval**
   - Structural navigation
   - Semantic matching
   - Multi-signal fusion

4. **Knowledge Synthesis**
   - Context building
   - Visual content recovery
   - VLM-based generation

## 📊 Performance

Results on multimodal benchmarks:

| Dataset | RAG-Anything | MMGraphRAG | GPT-4o-mini |
|---------|--------------|------------|-------------|
| DocBench | **63.4%** | 61.0% | 51.2% |
| MMLongBench | **42.8%** | 37.7% | 33.5% |

Performance on long documents (100+ pages):
- **13+ point improvement** over baselines
- Consistently better with increasing document length

## 🔬 Advanced Usage

### Custom Parsers

Add support for new document formats:

```python
from rag_anything.parsers import BaseParser

class CustomParser(BaseParser):
    def _load_document(self, file_path):
        # Custom loading logic
        pass
    
    def _extract_content_units(self, doc, document_id):
        # Custom extraction logic
        pass
```

### Custom Retrieval Strategies

Extend retrieval with domain-specific logic:

```python
from rag_anything.retrieval import FusionRanker

class DomainRanker(FusionRanker):
    def _compute_domain_score(self, unit, query):
        # Domain-specific scoring
        pass
```

## 🛠️ Development

### Project Structure

```
rag_anything/
├── config/          # Configuration management
├── core/            # Core components
├── parsers/         # Document parsers
├── graphs/          # Knowledge graph builders
├── retrieval/       # Retrieval components
├── embeddings/      # Embedding generation
├── models/          # VLM/LLM interfaces
├── pipeline/        # End-to-end pipelines
└── utils/           # Utilities
```

### Running Tests

```bash
pytest tests/
```

### Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📝 Citation

If you use RAG-Anything in your research, please cite:

```bibtex
@article{guo2025raganything,
  title={RAG-Anything: All-in-One RAG Framework},
  author={Guo, Zirui and Ren, Xubin and Xu, Lingrui and Zhang, Jiahao and Huang, Chao},
  journal={arXiv preprint arXiv:2510.12323},
  year={2025}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Acknowledgments

- Based on research from The University of Hong Kong
- Built on top of GraphRAG and LightRAG concepts
- Uses OpenAI GPT-4V and Anthropic Claude APIs

## 🔗 Links

- [Paper](https://arxiv.org/abs/2510.12323)
- [GitHub](https://github.com/HKUDS/RAG-Anything)
- [Documentation](https://rag-anything.readthedocs.io)
- [Examples](https://github.com/HKUDS/RAG-Anything/tree/main/examples)

## 💬 Support

- GitHub Issues: [Report bugs](https://github.com/HKUDS/RAG-Anything/issues)
- Discussions: [Ask questions](https://github.com/HKUDS/RAG-Anything/discussions)
- Email: chaohuang75@gmail.com
"""

# Save README
if __name__ == "__main__":
    with open("README.md", "w") as f:
        f.write(README_CONTENT)
    print("✓ README.md created")
    
    # Run example
    print("\nRunning basic example...")
    # basic_example()  # Uncomment to run