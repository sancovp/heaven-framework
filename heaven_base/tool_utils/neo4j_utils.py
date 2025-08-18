"""Utility functions for Neo4j integration.

This module provides functions for interacting with Neo4j graph database,
including converting tools and agents to Cypher queries and visualizing
the knowledge graph in the Neo4j Browser.
"""

import urllib.parse
import os
from typing import Dict, Any, List, Tuple, Optional


class KnowledgeGraphBuilder:
    """Class for building and managing a Neo4j knowledge graph of the system.
    
    This class provides methods to add tools, agents, and their relationships
    to a Neo4j database, as well as querying and visualizing the graph.
    """
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize the graph builder with Neo4j connection details.
        
        Args:
            uri: Neo4j server URI (default: from env NEO4J_URI or bolt://host.docker.internal:7687)
            user: Neo4j username (default: from env NEO4J_USER or neo4j)
            password: Neo4j password (default: from env NEO4J_PASSWORD or password)
        """
        # Get connection params from environment or use HEAVEN defaults
        self.uri = uri or os.environ.get('NEO4J_URI', 'bolt://host.docker.internal:7687')
        self.user = user or os.environ.get('NEO4J_USER', 'neo4j')
        self.password = password or os.environ.get('NEO4J_PASSWORD', 'password')
        self.driver = None  # Lazy connection - only connect when needed
    
    def _ensure_connection(self):
        """Ensure Neo4j connection is established (lazy initialization)."""
        if self.driver is None:
            try:
                from neo4j import GraphDatabase
                # Add connection timeout to prevent hanging
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password),
                    connection_timeout=5.0,  # 5 second timeout
                    max_connection_lifetime=30  # 30 second max lifetime
                )
                self._test_connection()
            except ImportError:
                raise ImportError("Neo4j Python driver not installed. Install with: pip install neo4j")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Neo4j: {str(e)}")
    
    def _test_connection(self):
        """Test the connection to Neo4j."""
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            if not record or record.get("test") != 1:
                raise ConnectionError("Neo4j connection test failed")
    
    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver') and self.driver is not None:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        self._ensure_connection()
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_indexes(self):
        """Create indexes for faster lookups."""
        self._ensure_connection()
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Tool) ON (t.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Agent) ON (a.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (f:Function) ON (f.name)")

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return the results.
        
        Args:
            query: Cypher query string
            params: Optional parameters for the query
            
        Returns:
            List of records as dictionaries
        """
        self._ensure_connection()
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def visualize_query(self, query: str) -> str:
        """Open the Neo4j Browser with the specified query.
        
        Args:
            query: Cypher query to visualize
            
        Returns:
            The browser URL that was opened
        """
        encoded_query = urllib.parse.quote(query)
        browser_url = f"http://localhost:7474/browser/?cmd={encoded_query}"
        
        # Open Firefox to the URL
        os.system(f"firefox '{browser_url}' &")
        
        return browser_url
    
    def get_browser_url(self, query: str) -> str:
        """Get the Neo4j Browser URL for a query without opening it.
        
        Args:
            query: Cypher query to visualize
            
        Returns:
            The browser URL for the query
        """
        encoded_query = urllib.parse.quote(query)
        return f"http://localhost:7474/browser/?cmd={encoded_query}"

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in the knowledge graph.
        
        Returns:
            List of tool records
        """
        query = "MATCH (t:Tool) RETURN t"
        return self.execute_query(query)
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents in the knowledge graph.
        
        Returns:
            List of agent records
        """
        query = "MATCH (a:Agent) RETURN a"
        return self.execute_query(query)
    
    def get_agent_tools(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all tools used by an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of tool records
        """
        query = "MATCH (a:Agent {name: $name})-[:USES]->(t:Tool) RETURN t"
        return self.execute_query(query, {"name": agent_name})
    
    def get_tool_dependencies(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get all dependencies of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of dependency records
        """
        query = """
        MATCH (t:Tool {name: $name})-[r]->(dep)
        RETURN type(r) as relationship_type, dep
        """
        return self.execute_query(query, {"name": tool_name})
    
    def visualize_system_overview(self) -> str:
        """Visualize an overview of the entire system.
        
        Returns:
            The browser URL that was opened
        """
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 100
        """
        return self.visualize_query(query)


def create_sample_data(graph_builder: KnowledgeGraphBuilder) -> None:
    """Create sample data in the Neo4j database for testing.
    
    Args:
        graph_builder: KnowledgeGraphBuilder instance
    """
    # Clear existing data
    graph_builder.clear_database()
    
    # Create indexes
    graph_builder.create_indexes()
    
    # Create sample tools
    tools_query = """
    CREATE (edit:Tool {name: 'EditTool', description: 'Tool for editing files', is_async: false})
    CREATE (bash:Tool {name: 'BashTool', description: 'Tool for running bash commands', is_async: false})
    CREATE (network:Tool {name: 'NetworkFileViewerTool', description: 'Tool for viewing files in network containers', is_async: false})
    CREATE (hermes:Tool {name: 'HermesTool', description: 'Tool for sending goals to agents', is_async: true})
    
    // Create tool arguments
    CREATE (edit_path:Argument {name: 'path', type: 'str', required: true})
    CREATE (edit_cmd:Argument {name: 'command', type: 'str', required: true})
    CREATE (bash_cmd:Argument {name: 'command', type: 'str', required: true})
    
    // Create relationships
    CREATE (edit)-[:HAS_ARGUMENT]->(edit_path)
    CREATE (edit)-[:HAS_ARGUMENT]->(edit_cmd)
    CREATE (bash)-[:HAS_ARGUMENT]->(bash_cmd)
    CREATE (bash)-[:DEPENDS_ON]->(edit)
    """
    
    # Create sample agents
    agents_query = """
    CREATE (file:Agent {name: 'FileAgent', model: 'claude-3-sonnet-20240229', temperature: 0.2})
    CREATE (code:Agent {name: 'CodeAgent', model: 'claude-3-opus-20240229', temperature: 0.1})
    
    // Create relationships between agents and tools
    MATCH (file:Agent {name: 'FileAgent'})
    MATCH (edit:Tool {name: 'EditTool'})
    MATCH (bash:Tool {name: 'BashTool'})
    CREATE (file)-[:USES]->(edit)
    CREATE (file)-[:USES]->(bash)
    
    MATCH (code:Agent {name: 'CodeAgent'})
    MATCH (edit:Tool {name: 'EditTool'})
    MATCH (network:Tool {name: 'NetworkFileViewerTool'})
    CREATE (code)-[:USES]->(edit)
    CREATE (code)-[:USES]->(network)
    """
    
    # Execute queries
    graph_builder.execute_query(tools_query)
    graph_builder.execute_query(agents_query)
    
    print("Sample data created successfully.")


def main():
    """Main function for testing the Neo4j utilities."""
    try:
        print("Initializing Neo4j knowledge graph...")
        graph_builder = KnowledgeGraphBuilder()
        
        print("Creating sample data...")
        create_sample_data(graph_builder)
        
        print("Visualizing system overview...")
        browser_url = graph_builder.visualize_system_overview()
        print(f"Browser opened at: {browser_url}")
        
        # Example queries
        print("\nQuerying all tools:")
        tools = graph_builder.get_all_tools()
        for tool in tools:
            print(f"- {tool['t']['name']}: {tool['t']['description']}")
        
        print("\nQuerying tools used by FileAgent:")
        agent_tools = graph_builder.get_agent_tools("FileAgent")
        for tool in agent_tools:
            print(f"- {tool['t']['name']}")
        
        # Example of a custom query and visualization
        print("\nVisualizing tools with their arguments:")
        query = """
        MATCH (t:Tool)-[:HAS_ARGUMENT]->(a:Argument)
        RETURN t, a
        """
        browser_url = graph_builder.visualize_query(query)
        print(f"Browser opened at: {browser_url}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'graph_builder' in locals():
            graph_builder.close()


if __name__ == "__main__":
    main()