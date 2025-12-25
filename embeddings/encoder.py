
class UnifiedEncoder:
    '''Generate embeddings for all components: entities, relations, chunks.'''
    
    def __init__(self, config: EmbeddingConfig):
        '''Initialize encoder with OpenAI or other embedding model.'''
        if config.provider == ModelProvider.OPENAI:
            import openai
            self.client = openai.OpenAI(api_key=config.api_key)
            self.model = config.model  # e.g., text-embedding-3-large
        pass
    
    def encode_text(self, text: str) -> np.ndarray:
        '''
        Encode text to dense vector.
        
        Input:
            - text: String to encode
        
        Output:
            - numpy array of shape (dimensions,)
        
        Implementation (OpenAI):
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding)
        '''
        pass
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        '''
        Encode multiple texts in batch.
        
        Output:
            - numpy array of shape (len(texts), dimensions)
        '''
        pass
    
    def encode_entity(self, entity: Entity) -> np.ndarray:
        '''Encode entity using its name and attributes.'''
        text = f"{entity.name}: {entity.attributes.get('description', '')}"
        return self.encode_text(text)
    
    def encode_chunk(self, chunk: ContentUnit) -> np.ndarray:
        '''Encode content unit.'''
        if chunk.description:
            text = chunk.description  # Use generated description for non-text
        else:
            text = str(chunk.content)  # Use raw content for text
        return self.encode_text(text)