
class TextGraphBuilder:
    '''Build text-based knowledge graph from textual content.'''
    
    def build(self, text_units: List[ContentUnit]) -> KnowledgeGraph:
        '''
        Build text KG using NER + relation extraction.
        
        Input:
            - text_units: List of ContentUnit with modality=TEXT
        
        Output:
            - KnowledgeGraph with text entities and relations
        
        Algorithm:
            1. Concatenate text content
            2. Run NER to identify entities
            3. Extract relations between entities
            4. Build graph structure
        
        Options:
            - Use spaCy for NER
            - Use LLM for relation extraction
        '''
        pass
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        '''Extract entities using spaCy NER.'''
        # TODO: Requires spaCy model
        # import spacy
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(text)
        # return [Entity(...) for ent in doc.ents]
        pass
    
    def _extract_relations_llm(
        self,
        text: str,
        entities: List[Entity],
        llm: LLMInterface
    ) -> List[Relation]:
        '''Extract relations using LLM.'''
        pass