
class KnowledgeSynthesizer:
    '''Generate final answer from retrieved knowledge.'''
    
    def synthesize(
        self,
        query: str,
        retrieved_units: List[RetrievalResult],
        vlm: VLMInterface,
        max_context_tokens: int = 12000
    ) -> str:
        '''
        Generate answer using VLM.
        
        Input:
            - query: User question
            - retrieved_units: Retrieved content units with scores
            - vlm: Vision-language model
        
        Output:
            - Generated answer string
        
        Algorithm:
            1. Build textual context from all units
            2. Recover visual content (images) for multimodal units
            3. Call VLM with query + context + visuals
            4. Return response
        '''
        pass
    
    def _build_textual_context(
        self,
        units: List[RetrievalResult]
    ) -> str:
        '''
        Concatenate textual representations.
        
        Format:
            [Source: {unit_id}]
            {content/description}
            ---
        '''
        pass
    
    def _recover_visual_content(
        self,
        units: List[RetrievalResult]
    ) -> List[Image.Image]:
        '''
        Load actual images for visual units.
        '''
        pass