# RAG-Anything Python Code Architecture

## Project Structure

```
rag_anything/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configuration management
│   └── prompts.py               # Prompt templates
├── core/
│   ├── __init__.py
│   ├── knowledge_unification.py # Multimodal knowledge unification
│   ├── dual_graph_builder.py   # Dual-graph construction
│   ├── hybrid_retriever.py     # Cross-modal hybrid retrieval
│   └── synthesizer.py          # Knowledge-enhanced generation
├── parsers/
│   ├── __init__.py
│   ├── base_parser.py          # Abstract base parser
│   ├── text_parser.py          # Text extraction
│   ├── image_parser.py         # Image + caption extraction
│   ├── table_parser.py         # Table structure parsing
│   └── equation_parser.py      # LaTeX equation recognition
├── graphs/
│   ├── __init__.py
│   ├── cross_modal_graph.py    # Cross-modal KG builder
│   ├── text_graph.py           # Text-based KG builder
│   ├── graph_fusion.py         # Entity alignment & fusion
│   └── graph_utils.py          # Graph operations utilities
├── retrieval/
│   ├── __init__.py
│   ├── structural_navigator.py # Graph-based navigation
│   ├── semantic_matcher.py     # Vector similarity search
│   └── fusion_ranker.py        # Multi-signal fusion scoring
├── embeddings/
│   ├── __init__.py
│   ├── encoder.py              # Unified embedding encoder
│   └── vector_store.py         # Vector database interface
├── models/
│   ├── __init__.py
│   ├── vlm_interface.py        # VLM API wrapper
│   └── llm_interface.py        # LLM API wrapper
├── utils/
│   ├── __init__.py
│   ├── logger.py               # Loguru setup
│   ├── exceptions.py           # Custom exceptions
│   └── metrics.py              # Evaluation metrics
└── pipeline/
    ├── __init__.py
    ├── indexing_pipeline.py    # End-to-end indexing
    └── query_pipeline.py       # End-to-end querying

tests/
├── __init__.py
├── test_parsers.py
├── test_graphs.py
├── test_retrieval.py
└── test_pipeline.py

examples/
├── basic_usage.py
├── long_document_qa.py
└── multimodal_retrieval.py

requirements.txt
README.md
setup.py
```

---

## Module Specifications

### 1. config/settings.py
**Purpose:** Central configuration management  
**Key Classes:**
- `RAGConfig`: Main configuration dataclass
  - `model_config`: Model settings (VLM, LLM, embedding)
  - `graph_config`: Graph construction parameters
  - `retrieval_config`: Retrieval hyperparameters
  - `parser_config`: Document parsing settings

**Dependencies:** `pydantic`, `yaml`

---

### 2. config/prompts.py
**Purpose:** Manage all prompt templates  
**Key Constants:**
- `VISION_ANALYSIS_PROMPT`: Image interpretation template
- `TABLE_ANALYSIS_PROMPT`: Table parsing template
- `EQUATION_ANALYSIS_PROMPT`: Math expression template
- `ENTITY_EXTRACTION_PROMPT`: Entity/relation extraction
- `SYNTHESIS_PROMPT`: Final answer generation

---

### 3. core/knowledge_unification.py
**Purpose:** Decompose documents into atomic knowledge units  
**Key Classes:**
- `ContentUnit`: Dataclass for atomic units
  - `modality_type`: Enum (text, image, table, equation)
  - `content`: Raw content
  - `metadata`: Position, caption, etc.
  
- `KnowledgeUnifier`:
  - `decompose(document: Document) -> List[ContentUnit]`
  - `_extract_hierarchy(content_units) -> Dict`
  
**Dependencies:** `parsers.*`

---

### 4. parsers/base_parser.py
**Purpose:** Abstract interface for all parsers  
**Key Classes:**
- `BaseParser(ABC)`:
  - `parse(file_path: Path) -> List[ContentUnit]`
  - `extract_metadata(unit) -> Dict`

**Dependencies:** `abc`, `pathlib`

---

### 5. parsers/text_parser.py
**Purpose:** Extract hierarchical text (paragraphs, lists)  
**Key Classes:**
- `TextParser(BaseParser)`:
  - `parse(file_path) -> List[ContentUnit]`
  - `_segment_paragraphs(text) -> List[str]`

**Dependencies:** `nltk` or `spacy`

---

### 6. parsers/image_parser.py
**Purpose:** Extract images with captions and metadata  
**Key Classes:**
- `ImageParser(BaseParser)`:
  - `parse(file_path) -> List[ContentUnit]`
  - `_extract_caption(image_context) -> str`
  - `_generate_description(image, context) -> str` (via VLM)

**Dependencies:** `PIL`, `models.vlm_interface`

---

### 7. parsers/table_parser.py
**Purpose:** Parse table structure (headers, cells, values)  
**Key Classes:**
- `TableParser(BaseParser)`:
  - `parse(file_path) -> List[ContentUnit]`
  - `_extract_structure(table) -> Dict`
  - `_parse_cells(table) -> List[Cell]`

**Dependencies:** `pandas`, `camelot` or `tabula`

---

### 8. parsers/equation_parser.py
**Purpose:** Recognize LaTeX equations  
**Key Classes:**
- `EquationParser(BaseParser)`:
  - `parse(file_path) -> List[ContentUnit]`
  - `_ocr_equation(image) -> str` (LaTeX format)

**Dependencies:** `pix2tex` or similar OCR

---

### 9. graphs/cross_modal_graph.py
**Purpose:** Build cross-modal knowledge graph  
**Key Classes:**
- `CrossModalGraphBuilder`:
  - `build(content_units: List[ContentUnit]) -> Tuple[Set[Node], Set[Edge]]`
  - `_generate_descriptions(unit, context) -> Tuple[str, str]`
  - `_extract_intra_entities(description) -> Tuple[Set, Set]`
  - `_create_multimodal_anchor(unit) -> Node`

**Dependencies:** `networkx`, `models.vlm_interface`

---

### 10. graphs/text_graph.py
**Purpose:** Build text-based knowledge graph  
**Key Classes:**
- `TextGraphBuilder`:
  - `build(text_units: List[ContentUnit]) -> Tuple[Set[Node], Set[Edge]]`
  - `_extract_entities(text) -> List[Entity]`
  - `_extract_relations(text, entities) -> List[Relation]`

**Dependencies:** `networkx`, `spacy` or `models.llm_interface`

---

### 11. graphs/graph_fusion.py
**Purpose:** Merge cross-modal and text graphs via entity alignment  
**Key Classes:**
- `GraphFusion`:
  - `merge(graph1, graph2) -> Tuple[Set[Node], Set[Edge]]`
  - `_align_entities(nodes1, nodes2) -> Dict[Node, Node]`
  - `_consolidate_nodes(aligned_pairs) -> Set[Node]`

**Dependencies:** `networkx`

---

### 12. graphs/graph_utils.py
**Purpose:** Graph operations (traversal, subgraph extraction)  
**Key Functions:**
- `get_neighborhood(graph, node, hops) -> Set[Node]`
- `find_paths(graph, start, end, max_hops) -> List[Path]`
- `compute_centrality(graph) -> Dict[Node, float]`

**Dependencies:** `networkx`

---

### 13. embeddings/encoder.py
**Purpose:** Unified embedding generation  
**Key Classes:**
- `UnifiedEncoder`:
  - `encode_text(text: str) -> np.ndarray`
  - `encode_entity(entity: Entity) -> np.ndarray`
  - `encode_chunk(chunk: ContentUnit) -> np.ndarray`
  - `batch_encode(items: List) -> np.ndarray`

**Dependencies:** `openai`, `sentence-transformers`

---

### 14. embeddings/vector_store.py
**Purpose:** Vector database interface  
**Key Classes:**
- `VectorStore(ABC)`:
  - `add(embeddings, metadata) -> None`
  - `search(query_embedding, top_k) -> List[Result]`
  
- `FAISSStore(VectorStore)`: FAISS implementation
- `ChromaStore(VectorStore)`: ChromaDB implementation

**Dependencies:** `faiss`, `chromadb`

---

### 15. core/dual_graph_builder.py
**Purpose:** Orchestrate dual-graph construction  
**Key Classes:**
- `DualGraphBuilder`:
  - `build(content_units) -> Tuple[Graph, EmbeddingTable]`
  - `_build_cross_modal_graph(units) -> Graph`
  - `_build_text_graph(units) -> Graph`
  - `_fuse_graphs(g1, g2) -> Graph`
  - `_create_embedding_table(graph, units) -> Dict`

**Dependencies:** `graphs.*`, `embeddings.*`

---

### 16. retrieval/structural_navigator.py
**Purpose:** Graph-based retrieval  
**Key Classes:**
- `StructuralNavigator`:
  - `navigate(query, graph) -> Set[Node]`
  - `_match_entities(query_terms, graph) -> Set[Node]`
  - `_expand_neighborhood(nodes, graph, hops) -> Set[Node]`
  - `_retrieve_content(nodes) -> List[ContentUnit]`

**Dependencies:** `graphs.graph_utils`

---

### 17. retrieval/semantic_matcher.py
**Purpose:** Vector similarity search  
**Key Classes:**
- `SemanticMatcher`:
  - `match(query_embedding, vector_store, top_k) -> List[Match]`
  - `_compute_similarity(emb1, emb2) -> float`
  - `_rank_by_score(matches) -> List[Match]`

**Dependencies:** `embeddings.vector_store`

---

### 18. retrieval/fusion_ranker.py
**Purpose:** Multi-signal fusion scoring  
**Key Classes:**
- `FusionRanker`:
  - `rank(structural_results, semantic_results, query) -> List[Result]`
  - `_compute_structural_score(node, graph) -> float`
  - `_compute_semantic_score(match) -> float`
  - `_infer_modality_preference(query) -> Dict[str, float]`
  - `_fuse_scores(candidates) -> List[ScoredResult]`

**Dependencies:** None

---

### 19. core/hybrid_retriever.py
**Purpose:** Orchestrate cross-modal hybrid retrieval  
**Key Classes:**
- `HybridRetriever`:
  - `retrieve(query, index) -> List[ContentUnit]`
  - `_encode_query(query) -> Tuple[str, np.ndarray]`
  - `_structural_retrieval(query, graph) -> Set`
  - `_semantic_retrieval(query_emb, vector_store) -> Set`
  - `_unify_and_rank(candidates) -> List`

**Dependencies:** `retrieval.*`

---

### 20. core/synthesizer.py
**Purpose:** Generate answers from retrieved knowledge  
**Key Classes:**
- `KnowledgeSynthesizer`:
  - `synthesize(query, retrieved_units, vlm) -> str`
  - `_build_textual_context(units) -> str`
  - `_recover_visual_content(units) -> List[Image]`
  - `_generate_response(query, context, visuals) -> str`

**Dependencies:** `models.vlm_interface`

---

### 21. models/vlm_interface.py
**Purpose:** Vision-Language Model API wrapper  
**Key Classes:**
- `VLMInterface`:
  - `generate(prompt, images, max_tokens) -> str`
  - `describe_image(image, context) -> str`
  - `extract_entities(description) -> Dict`

**Dependencies:** `openai` (GPT-4V) or `anthropic` (Claude)

---

### 22. models/llm_interface.py
**Purpose:** Large Language Model API wrapper  
**Key Classes:**
- `LLMInterface`:
  - `generate(prompt, max_tokens) -> str`
  - `extract_entities(text) -> List[Entity]`
  - `extract_relations(text) -> List[Relation]`

**Dependencies:** `openai` or `anthropic`

---

### 23. pipeline/indexing_pipeline.py
**Purpose:** End-to-end document indexing  
**Key Classes:**
- `IndexingPipeline`:
  - `index_document(file_path) -> Index`
  - `_parse_document(file_path) -> List[ContentUnit]`
  - `_build_graphs(units) -> Graph`
  - `_create_embeddings(graph, units) -> EmbeddingTable`

**Dependencies:** `core.*`, `parsers.*`

---

### 24. pipeline/query_pipeline.py
**Purpose:** End-to-end query processing  
**Key Classes:**
- `QueryPipeline`:
  - `query(question, index) -> str`
  - `_retrieve(question, index) -> List[ContentUnit]`
  - `_synthesize(question, retrieved) -> str`

**Dependencies:** `core.hybrid_retriever`, `core.synthesizer`

---

### 25. utils/logger.py
**Purpose:** Centralized logging with loguru  
**Key Functions:**
- `setup_logger(level, log_file) -> None`
- `log_exception(exc) -> None`

**Dependencies:** `loguru`

---

### 26. utils/exceptions.py
**Purpose:** Custom exception classes  
**Key Classes:**
- `RAGAnythingException`: Base exception
- `ParsingError`: Document parsing failures
- `GraphBuildError`: Graph construction failures
- `RetrievalError`: Retrieval failures
- `SynthesisError`: Generation failures

---

### 27. utils/metrics.py
**Purpose:** Evaluation metrics  
**Key Functions:**
- `compute_accuracy(predictions, ground_truth) -> float`
- `evaluate_retrieval(retrieved, relevant) -> Dict`

---

## Data Flow Diagram

```
┌─────────────────┐
│  Input Document │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Knowledge Unification       │
│ (parsers/*)                 │
│ → List[ContentUnit]         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Dual Graph Construction     │
│ (graphs/*)                  │
│ → Cross-Modal Graph         │
│ → Text-Based Graph          │
│ → Fused Graph G             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Index Creation              │
│ (embeddings/*)              │
│ → Embedding Table T         │
│ → Index I = (G, T)          │
└────────┬────────────────────┘
         │
    [Storage]
         │
         ▼
┌─────────────────────────────┐
│ Query Input                 │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Hybrid Retrieval            │
│ (retrieval/*)               │
│ → Structural Navigation     │
│ → Semantic Matching         │
│ → Fusion Ranking            │
│ → Retrieved Units C*(q)     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Knowledge Synthesis         │
│ (core/synthesizer.py)       │
│ → Build Context P(q)        │
│ → Recover Visuals V*(q)     │
│ → VLM Generation            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│ Final Answer    │
└─────────────────┘
```

---

## Key Design Patterns

1. **Strategy Pattern**: Different parsers for different modalities
2. **Builder Pattern**: Dual-graph construction
3. **Facade Pattern**: Pipeline classes abstract complexity
4. **Adapter Pattern**: VLM/LLM interfaces
5. **Template Method**: Base parser defines extraction flow

---

## Configuration Example (YAML)

```yaml
models:
  vlm:
    provider: openai
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
  llm:
    provider: openai
    model: gpt-4o-mini
  embedding:
    provider: openai
    model: text-embedding-3-large
    dimensions: 3072

graphs:
  context_window_size: 3  # δ parameter
  max_hops: 2
  entity_token_limit: 20000
  chunk_token_limit: 12000

retrieval:
  top_k_semantic: 10
  top_k_structural: 5
  fusion_weights:
    structural: 0.4
    semantic: 0.4
    modality: 0.2

parsers:
  image:
    resolution: 144  # dpi
    max_size: 2048
  table:
    engine: camelot
  equation:
    engine: pix2tex
```

---

## Dependencies (requirements.txt)

```
# Core
loguru>=0.7.0
pydantic>=2.0.0
python-dotenv>=1.0.0

# Document Processing
PyPDF2>=3.0.0
pdf2image>=1.16.0
Pillow>=10.0.0
camelot-py>=0.11.0
pix2tex>=0.1.0

# NLP & Graphs
spacy>=3.7.0
networkx>=3.2.0
nltk>=3.8.0

# Embeddings & Vector Stores
sentence-transformers>=2.3.0
faiss-cpu>=1.7.4
chromadb>=0.4.0

# API Clients
openai>=1.10.0
anthropic>=0.18.0

# Utilities
numpy>=1.24.0
pandas>=2.1.0
pyyaml>=6.0.0
tqdm>=4.66.0
```

---

## Testing Strategy

- **Unit Tests**: Each parser, graph builder, retriever
- **Integration Tests**: Pipeline end-to-end
- **Benchmark Tests**: DocBench, MMLongBench datasets
- **Performance Tests**: Memory and latency profiling

---

## Next Steps: Implementation Priority

1. **Phase 1**: Core infrastructure
   - Config management
   - Logger setup
   - Base classes (parsers, exceptions)

2. **Phase 2**: Parsing layer
   - Text, image, table, equation parsers
   - Knowledge unification

3. **Phase 3**: Graph construction
   - Cross-modal graph builder
   - Text graph builder
   - Graph fusion

4. **Phase 4**: Embeddings & indexing
   - Encoder implementation
   - Vector store integration

5. **Phase 5**: Retrieval layer
   - Structural navigator
   - Semantic matcher
   - Fusion ranker

6. **Phase 6**: Synthesis & pipelines
   - Synthesizer
   - End-to-end pipelines

7. **Phase 7**: Testing & optimization
   - Unit/integration tests
   - Benchmark evaluation
   - Performance tuning
