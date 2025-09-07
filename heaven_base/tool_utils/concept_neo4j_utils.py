"""Neo4j utilities for concept ingestion from the hierarchical concept aggregation system.

This module provides functions specifically for storing canonical concepts and their
relationships in Neo4j from the concept resolution pipeline.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from .neo4j_utils import KnowledgeGraphBuilder


class ConceptGraphBuilder:
    """Specialized graph builder for concept ingestion."""
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize the concept graph builder.
        
        Args:
            uri: Neo4j server URI (optional, uses env or default)
            user: Neo4j username (optional, uses env or default)  
            password: Neo4j password (optional, uses env or default)
        """
        self.graph_builder = KnowledgeGraphBuilder(uri, user, password)
    
    def close(self):
        """Close the Neo4j connection."""
        self.graph_builder.close()
    
    def create_concept_indexes(self):
        """Create indexes for temporal conversation knowledge graph."""
        queries = [
            "CREATE INDEX IF NOT EXISTS FOR (c:ConceptTag:Conversations) ON (c.canonical_form)",
            "CREATE INDEX IF NOT EXISTS FOR (c:ConceptTag:Conversations) ON (c.keyword)",
            "CREATE INDEX IF NOT EXISTS FOR (conv:Conversation:Conversations) ON (conv.conversation_id)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Phase:Conversations) ON (p.phase_number)",
            "CREATE INDEX IF NOT EXISTS FOR (i:Iteration:Conversations) ON (i.iteration_number)",
            "CREATE INDEX IF NOT EXISTS FOR (s:TotalSummary:Conversations) ON (s.conversation_id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:PhaseSummary:Conversations) ON (s.phase_id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:IterationSummary:Conversations) ON (s.iteration_number)"
        ]
        
        for query in queries:
            self.graph_builder.execute_query(query)
    
    def ingest_conversation(self, conversation_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create or merge a conversation node.
        
        Args:
            conversation_id: Unique conversation identifier
            metadata: Optional metadata about the conversation
            
        Returns:
            Node ID of the created/merged conversation
        """
        query = """
        MERGE (conv:Conversation:Conversations {conversation_id: $conversation_id})
        SET conv.last_updated = datetime($timestamp)
        SET conv += $metadata
        RETURN conv.conversation_id as node_id
        """
        
        params = {
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        result = self.graph_builder.execute_query(query, params)
        return result[0]['node_id'] if result else conversation_id
    
    def ingest_concept_tag(
        self, 
        keyword: str,
        description: str,
        canonical_form: str,
        conversation_id: str,
        iteration_number: int = None,
        phase_number: int = None
    ) -> str:
        """Ingest a concept tag from HEAVEN analysis.
        
        Args:
            keyword: The concept keyword
            description: Description of what this concept represents
            canonical_form: The canonical form of the concept
            conversation_id: Conversation this concept came from
            iteration_number: Specific iteration where concept appeared
            phase_number: Phase where concept appeared
            
        Returns:
            Node ID of the created/merged concept
        """
        query = """
        // Create or merge the concept tag
        MERGE (c:ConceptTag:Conversations {canonical_form: $canonical_form})
        SET c.keyword = $keyword
        SET c.description = $description
        SET c.last_updated = datetime($timestamp)
        
        // Link to conversation
        WITH c
        MATCH (conv:Conversation:Conversations {conversation_id: $conversation_id})
        MERGE (c)-[r:MENTIONED_IN]->(conv)
        SET r.last_updated = datetime($timestamp)
        
        // Link to iteration if specified
        WITH c
        OPTIONAL MATCH (i:Iteration:Conversations {iteration_number: $iteration_number, conversation_id: $conversation_id})
        FOREACH (_ IN CASE WHEN i IS NOT NULL THEN [1] ELSE [] END |
            MERGE (c)-[r2:MENTIONED_IN_ITERATION]->(i)
            SET r2.last_updated = datetime($timestamp)
        )
        
        RETURN c.canonical_form as node_id
        """
        
        params = {
            'canonical_form': canonical_form,
            'keyword': keyword,
            'description': description,
            'conversation_id': conversation_id,
            'iteration_number': iteration_number,
            'phase_number': phase_number,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.graph_builder.execute_query(query, params)
        return result[0]['node_id'] if result else canonical_form
    
    def ingest_temporal_concept_relationship(
        self,
        from_concept: str,
        to_concept: str,
        relationship_type: str,
        valid_at: str,
        invalid_at: str = None,
        strength: float = 1.0
    ):
        """Create a temporal relationship between two concepts.
        
        Args:
            from_concept: Canonical form of the source concept
            to_concept: Canonical form of the target concept
            relationship_type: Type of relationship (e.g., 'resolved_by', 'caused_by')
            valid_at: When this relationship became valid (ISO timestamp)
            invalid_at: When this relationship became invalid (ISO timestamp, optional)
            strength: Strength of the relationship (0.0 to 1.0)
        """
        query = """
        MATCH (c1:ConceptTag:Conversations {canonical_form: $from_concept})
        MATCH (c2:ConceptTag:Conversations {canonical_form: $to_concept})
        MERGE (c1)-[r:RELATES_TO]->(c2)
        SET r.type = $relationship_type
        SET r.valid_at = datetime($valid_at)
        SET r.invalid_at = CASE WHEN $invalid_at IS NOT NULL THEN datetime($invalid_at) ELSE NULL END
        SET r.strength = $strength
        SET r.last_updated = datetime($timestamp)
        """
        
        params = {
            'from_concept': from_concept,
            'to_concept': to_concept,
            'relationship_type': relationship_type,
            'valid_at': valid_at,
            'invalid_at': invalid_at,
            'strength': strength,
            'timestamp': datetime.now().isoformat()
        }
        
        self.graph_builder.execute_query(query, params)
    
    def ingest_phase(
        self,
        conversation_id: str,
        phase_number: int,
        iteration_range: str,
        description: str
    ) -> str:
        """Ingest a phase and link it to the conversation.
        
        Args:
            conversation_id: Conversation this phase belongs to
            phase_number: Phase number within the conversation
            iteration_range: Range of iterations (e.g., "1-15")
            description: Description of this phase
            
        Returns:
            Phase ID
        """
        phase_id = f"{conversation_id}_phase_{phase_number}"
        
        query = """
        MERGE (p:Phase:Conversations {phase_id: $phase_id})
        SET p.phase_number = $phase_number
        SET p.iteration_range = $iteration_range
        SET p.description = $description
        SET p.last_updated = datetime($timestamp)
        
        WITH p
        MATCH (conv:Conversation:Conversations {conversation_id: $conversation_id})
        MERGE (p)-[:PART_OF]->(conv)
        
        RETURN p.phase_id as node_id
        """
        
        params = {
            'phase_id': phase_id,
            'phase_number': phase_number,
            'iteration_range': iteration_range,
            'description': description,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.graph_builder.execute_query(query, params)
        return result[0]['node_id'] if result else phase_id
    
    def ingest_iteration(
        self,
        conversation_id: str,
        iteration_number: int,
        phase_number: int,
        timestamp: str,
        summary: str
    ) -> str:
        """Ingest an iteration and link it to phase and conversation.
        
        Args:
            conversation_id: Conversation this iteration belongs to
            iteration_number: Iteration number within the conversation
            phase_number: Phase this iteration belongs to
            timestamp: When this iteration occurred
            summary: Summary of this iteration
            
        Returns:
            Iteration ID
        """
        query = """
        MERGE (i:Iteration:Conversations {iteration_number: $iteration_number, conversation_id: $conversation_id})
        SET i.timestamp = datetime($timestamp)
        SET i.summary = $summary
        SET i.last_updated = datetime($current_timestamp)
        
        // Link to conversation
        WITH i
        MATCH (conv:Conversation:Conversations {conversation_id: $conversation_id})
        MERGE (i)-[:PART_OF]->(conv)
        
        // Link to phase
        WITH i
        MATCH (p:Phase:Conversations {phase_number: $phase_number, conversation_id: $conversation_id})
        MERGE (i)-[:PART_OF]->(p)
        
        RETURN i.iteration_number as node_id
        """
        
        params = {
            'conversation_id': conversation_id,
            'iteration_number': iteration_number,
            'phase_number': phase_number,
            'timestamp': timestamp,
            'summary': summary,
            'current_timestamp': datetime.now().isoformat()
        }
        
        result = self.graph_builder.execute_query(query, params)
        return result[0]['node_id'] if result else iteration_number
    
    def ingest_total_summary(
        self,
        conversation_id: str,
        executive_summary: str,
        outcomes: str,
        challenges: str,
        total_iterations: int
    ) -> str:
        """Ingest total summary for a conversation.
        
        Args:
            conversation_id: Conversation this summary belongs to
            executive_summary: Executive summary content
            outcomes: Key outcomes
            challenges: Major challenges
            total_iterations: Total number of iterations
            
        Returns:
            Summary ID
        """
        query = """
        MERGE (s:TotalSummary:Conversations {conversation_id: $conversation_id})
        SET s.executive_summary = $executive_summary
        SET s.outcomes = $outcomes
        SET s.challenges = $challenges
        SET s.total_iterations = $total_iterations
        SET s.last_updated = datetime($timestamp)
        
        // Link to conversation
        WITH s
        MATCH (conv:Conversation:Conversations {conversation_id: $conversation_id})
        MERGE (conv)-[:HAS_SUMMARY]->(s)
        
        RETURN s.conversation_id as node_id
        """
        
        params = {
            'conversation_id': conversation_id,
            'executive_summary': executive_summary,
            'outcomes': outcomes,
            'challenges': challenges,
            'total_iterations': total_iterations,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.graph_builder.execute_query(query, params)
        return result[0]['node_id'] if result else conversation_id
    
    def ingest_iteration_summary(
        self,
        conversation_id: str,
        iteration_number: int,
        actions_taken: str,
        outcomes: str,
        challenges: str,
        tools_used: str
    ) -> str:
        """Ingest iteration summary.
        
        Args:
            conversation_id: Conversation this iteration belongs to
            iteration_number: Iteration number
            actions_taken: Actions taken in this iteration
            outcomes: Outcomes achieved
            challenges: Challenges encountered
            tools_used: Tools used
            
        Returns:
            Summary ID
        """
        query = """
        MERGE (s:IterationSummary:Conversations {iteration_number: $iteration_number, conversation_id: $conversation_id})
        SET s.actions_taken = $actions_taken
        SET s.outcomes = $outcomes
        SET s.challenges = $challenges
        SET s.tools_used = $tools_used
        SET s.last_updated = datetime($timestamp)
        
        // Link to iteration
        WITH s
        MATCH (i:Iteration:Conversations {iteration_number: $iteration_number, conversation_id: $conversation_id})
        MERGE (i)-[:HAS_SUMMARY]->(s)
        
        RETURN s.iteration_number as node_id
        """
        
        params = {
            'conversation_id': conversation_id,
            'iteration_number': iteration_number,
            'actions_taken': actions_taken,
            'outcomes': outcomes,
            'challenges': challenges,
            'tools_used': tools_used,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.graph_builder.execute_query(query, params)
        return result[0]['node_id'] if result else iteration_number
    
    def query_concepts_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Query concepts that match a keyword.
        
        Args:
            keyword: Keyword to search for in concept names and descriptions
            
        Returns:
            List of matching concepts with their details
        """
        query = """
        MATCH (c:ConceptTag:Conversations)
        WHERE toLower(c.canonical_form) CONTAINS toLower($keyword)
           OR toLower(c.keyword) CONTAINS toLower($keyword)
           OR toLower(c.description) CONTAINS toLower($keyword)
        OPTIONAL MATCH (c)-[r:MENTIONED_IN]->(conv:Conversation)
        RETURN c.canonical_form as concept,
               c.keyword as keyword,
               c.description as description,
               collect(DISTINCT conv.conversation_id) as conversations,
               count(r) as total_mentions
        ORDER BY total_mentions DESC
        """
        
        params = {'keyword': keyword}
        return self.graph_builder.execute_query(query, params)
    
    def get_concept_graph(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Get the concept graph around a specific concept.
        
        Args:
            concept: Canonical form of the concept to center on
            depth: How many hops to traverse from the concept
            
        Returns:
            Dictionary with nodes and relationships
        """
        query = """
        MATCH path = (c:ConceptTag {canonical_form: $concept})-[*0..""" + str(depth) + """]-()
        WITH nodes(path) as nodes, relationships(path) as rels
        UNWIND nodes as node
        WITH collect(DISTINCT node) as all_nodes, rels
        UNWIND rels as rel
        WITH all_nodes, collect(DISTINCT rel) as all_rels
        RETURN all_nodes as nodes, all_rels as relationships
        """
        
        params = {'concept': concept}
        result = self.graph_builder.execute_query(query, params)
        
        if result:
            return {
                'nodes': result[0]['nodes'],
                'relationships': result[0]['relationships']
            }
        return {'nodes': [], 'relationships': []}
    
    def generate_daily_report(self, date: str = None) -> str:
        """Generate a daily report of concepts and activities.
        
        Args:
            date: Date to generate report for (ISO format, default: today)
            
        Returns:
            Formatted report string
        """
        if not date:
            date = datetime.now().date().isoformat()
        
        query = """
        // Find all concepts updated on the given date
        MATCH (c:ConceptTag:Conversations)
        WHERE date(c.last_updated) = date($date)
        
        // Get their conversation context
        OPTIONAL MATCH (c)-[r:MENTIONED_IN]->(conv:Conversation)
        WHERE date(r.last_updated) = date($date)
        
        WITH c, collect(DISTINCT conv.conversation_id) as convs
        
        RETURN 'ConceptTag' as concept_type,
               count(DISTINCT c) as concept_count,
               collect(c.canonical_form)[0..5] as sample_concepts,
               convs[0..3] as sample_conversations
        ORDER BY concept_count DESC
        """
        
        params = {'date': date}
        results = self.graph_builder.execute_query(query, params)
        
        # Format report
        report = f"# Daily Concept Report for {date}\n\n"
        
        if results:
            total_concepts = sum(r['concept_count'] for r in results)
            report += f"**Total Concepts Processed:** {total_concepts}\n\n"
            
            report += "## Concept Breakdown by Type:\n\n"
            for row in results:
                report += f"### {row['concept_type']} ({row['concept_count']} concepts)\n"
                report += f"- Sample concepts: {', '.join(row['sample_concepts'])}\n"
                if row['sample_conversations']:
                    report += f"- From conversations: {', '.join(row['sample_conversations'][:3])}\n"
                report += "\n"
        else:
            report += "No concepts were processed on this date.\n"
        
        return report


# Convenience functions for direct use

def load_concepts_to_neo4j(
    concept_resolution_output: Dict[str, Any],
    conversation_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Load resolved concepts from ConceptResolverTool output to Neo4j.
    
    Args:
        concept_resolution_output: Output from ConceptResolverTool
        conversation_metadata: Metadata about the conversation
        
    Returns:
        Summary of what was loaded
    """
    graph = ConceptGraphBuilder()
    
    try:
        # Create indexes
        graph.create_concept_indexes()
        
        # Ingest conversation
        conv_id = conversation_metadata.get('conversation_id')
        graph.ingest_conversation(conv_id, conversation_metadata)
        
        # Ingest canonical concepts
        concepts_loaded = 0
        for original, data in concept_resolution_output.get('canonical_concepts', {}).items():
            graph.ingest_canonical_concept(
                canonical_form=data['canonical_form'],
                original_form=original,
                concept_type=data['type'],
                description=data['description'],
                conversation_id=conv_id,
                phases_mentioned=data.get('phases_mentioned', [])
            )
            concepts_loaded += 1
        
        # Ingest relationships
        relationships_loaded = 0
        for rel in concept_resolution_output.get('concept_relationships', []):
            graph.ingest_concept_relationship(
                from_concept=rel['from_concept'],
                to_concept=rel['to_concept'],
                relationship_type=rel['relationship_type'],
                shared_phases=rel['shared_phases'],
                strength=rel['strength']
            )
            relationships_loaded += 1
        
        return {
            'status': 'success',
            'concepts_loaded': concepts_loaded,
            'relationships_loaded': relationships_loaded,
            'conversation_id': conv_id
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
        
    finally:
        graph.close()


def query_concept(keyword: str) -> List[Dict[str, Any]]:
    """Query for concepts matching a keyword.
    
    Args:
        keyword: Keyword to search for
        
    Returns:
        List of matching concepts
    """
    graph = ConceptGraphBuilder()
    try:
        return graph.query_concepts_by_keyword(keyword)
    finally:
        graph.close()


def get_daily_report(date: str = None) -> str:
    """Get the daily concept processing report.
    
    Args:
        date: Date to get report for (ISO format, default: today)
        
    Returns:
        Formatted report string
    """
    graph = ConceptGraphBuilder()
    try:
        return graph.generate_daily_report(date)
    finally:
        graph.close()