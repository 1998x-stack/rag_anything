class IndexingPipeline:
    '''End-to-end document indexing pipeline.'''
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.parsers = self._initialize_parsers()
        self.vlm = VLMInterface(config.vlm)
        self.encoder = UnifiedEncoder(config.embedding)
        self.graph_builder = DualGraphBuilder(config, self.vlm, self.encoder)
    
    def index_document(self, file_path: Path) -> Index:
        '''
        Index a single document.
        
        Steps:
            1. Parse document -> List[ContentUnit]
            2. Build dual graph -> KnowledgeGraph
            3. Generate embeddings -> EmbeddingTable
            4. Create Index I = (G, T)
        '''
        pass
    
    def index_multiple(self, file_paths: List[Path]) -> Index:
        '''
        Index multiple documents into single unified index.
        '''
        pass