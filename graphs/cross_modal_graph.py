class CrossModalGraphBuilder:
    '''Build cross-modal knowledge graph from non-text content units.'''
    
    def build(self, content_units: List[ContentUnit]) -> KnowledgeGraph:
        '''
        Build cross-modal KG from non-text units.
        
        Input:
            - content_units: List of ContentUnit (images, tables, equations)
        
        Output:
            - KnowledgeGraph with:
                * Multimodal anchor nodes (v_mm_j)
                * Intra-chunk entities (Vj)
                * belongs_to edges
        
        Algorithm:
            1. For each non-text unit cj:
                a. Generate descriptions using VLM: (d_chunk_j, e_entity_j)
                b. Extract entities/relations: R(d_chunk_j) -> (Vj, Ej)
                c. Create multimodal anchor node v_mm_j
                d. Add belongs_to edges: u -> v_mm_j for all u in Vj
            2. Return merged graph
        '''
        pass
    
    def _generate_descriptions(
        self,
        unit: ContentUnit,
        context_units: List[ContentUnit],
        vlm: VLMInterface
    ) -> Tuple[str, Dict]:
        '''
        Generate textual representations for non-text content.
        
        Input:
            - unit: Non-text content unit
            - context_units: Neighborhood units (C_j with |k-j| <= δ)
            - vlm: Vision-language model interface
        
        Output:
            - d_chunk_j: Detailed description for retrieval
            - e_entity_j: Entity summary for graph construction
        
        Implementation:
            - For IMAGE: vlm.describe_image(unit.content, context)
            - For TABLE: vlm.analyze_table(unit.content, context)
            - For EQUATION: vlm.analyze_equation(unit.content, context)
        '''
        pass
    
    def _extract_intra_entities(
        self,
        description: str,
        vlm: VLMInterface
    ) -> Tuple[Set[Entity], Set[Relation]]:
        '''
        Extract fine-grained entities from description: R(d_chunk_j) -> (Vj, Ej).
        
        Uses VLM to extract entities and relations.
        '''
        pass