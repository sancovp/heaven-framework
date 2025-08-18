"""Tool for interacting with Neo4j graph database.

This module provides a tool for executing Cypher queries against a Neo4j database
and visualizing the results in the Neo4j Browser.
"""

from typing import Dict, Any, List, Optional
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..tool_utils.neo4j_utils import KnowledgeGraphBuilder


class Neo4jToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'query': {
            'name': 'query',
            'type': 'str',
            'description': 'Cypher query to execute',
            'required': True
        },
        'params': {
            'name': 'params',
            'type': 'dict',
            'description': 'Parameters for the Cypher query',
            'required': False
        },
        'visualize': {
            'name': 'visualize',
            'type': 'bool',
            'description': 'Whether to open Neo4j Browser with this query',
            'required': False
        }
    }


def neo4j_tool_func(query: str, params: Optional[Dict[str, Any]] = None, visualize: bool = False) -> Dict[str, Any]:
    """Execute a Cypher query against Neo4j and optionally visualize the results.
    
    Args:
        query: Cypher query to execute
        params: Optional parameters for the query
        visualize: Whether to open Neo4j Browser with this query
        
    Returns:
        Dictionary containing query results and browser URL if visualized
    """
    try:
        # Initialize graph builder
        graph_builder = KnowledgeGraphBuilder()
        
        # Execute query
        result = graph_builder.execute_query(query, params)
        
        # Visualize if requested
        browser_url = None
        if visualize:
            browser_url = graph_builder.visualize_query(query)
        
        # Construct response
        response = {
            'status': 'success',
            'results': result,
            'result_count': len(result)
        }
        
        if browser_url:
            response['browser_url'] = browser_url
        
        return response
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        if 'graph_builder' in locals():
            graph_builder.close()


class Neo4jTool(BaseHeavenTool):
    name = "Neo4jTool"
    description = "Executes Cypher queries against Neo4j database and can visualize results in browser"
    func = neo4j_tool_func
    args_schema = Neo4jToolArgsSchema
    is_async = False