import networkx as nx
import random


#Step 1 Creating a graph (bidirectional)
def create_bidirectional_connected_graph(num_nodes, edge_prob):
    """
    Create a strictly bidirectional directed graph (for each edge (u->v), 
    also v->u), ensuring the underlying undirected version is connected.
    """
    while True:
        # A) Generate a directed graph with the specified number of edges
        directed_graph = nx.gnm_random_graph(
            n=num_nodes,
            m=int(edge_prob * num_nodes * (num_nodes - 1)),
            directed=True
        )
        # B) Make it strictly bidirectional by adding reverse edges
        for u, v in list(directed_graph.edges()):
            directed_graph.add_edge(v, u)

        # C) Convert to undirected to check connectivity
        undirected_view = directed_graph.to_undirected()

        if nx.is_connected(undirected_view):
            print("\nBidirectional Graph Edges (Direct View):")
            print(list(directed_graph.edges()))
            print("→ Underlying undirected version is connected.")
            return directed_graph  # Return the strictly bidirectional directed graph

        # If not connected, loop again to generate a new graph

# Help function (to convert the bidirectional graph into  unidrected graph)
def convert_to_undirected(graph):
    """
    Converts the bidirectional graph to an undirected graph,
    keeping only one copy of each edge. Verifies connectivity.
    """
    if graph.is_directed():
        graph = graph.to_undirected()

    print("\n[DEBUG] Graph After Converting to Undirected:", list(graph.edges()))
    print("[DEBUG] Is Graph Connected After Conversion?", nx.is_connected(graph))

    if not nx.is_connected(graph):
        raise ValueError("Graph must be connected for edge coloring.")
    return graph




#help function to get dict  from of graph
def build_graph_dict(graph_nx):
    """
    Create a dictionary adjacency representation from a NetworkX graph.
    Example: {node: {neighbor1, neighbor2, ...}, ...}
    """
    graph_dict = {}
    for u, v in graph_nx.edges():
        # Ensure each endpoint is in the dict
        if u not in graph_dict:
            graph_dict[u] = set()
        if v not in graph_dict:
            graph_dict[v] = set()

        # Add each node to the other's neighbor set
        graph_dict[u].add(v)
        graph_dict[v].add(u)

    return graph_dict


#Help function to check wheather result is correct or not
def validate_node_incident_colors(graph, edge_colors, max_allowed=2):
    print("====================Start of Validate node incident colors=======================")
    """
    Recalculate the incident colors for all nodes based on the edge_colors dictionary.
    Print the incident colors for each node.
    If any node has more than 'max_allowed' distinct incident colors,
    print that node's incident colors and raise a ValueError.
    
    Parameters:
        graph: A NetworkX graph whose nodes we are validating.
        edge_colors: Dictionary mapping each edge (u,v) to its color.
                     It is assumed that for each edge (u,v), a corresponding (v,u) exists.
        max_allowed: The maximum allowed number of distinct incident colors per node (default 2).
    """
    # Initialize a temporary dictionary for incident colors.
    temp_node_colors = {node: set() for node in graph.nodes()}
    
    # To avoid processing both (u,v) and (v,u) twice, iterate over sorted edge tuples.
    temp_processed_edges = set()
    for (u, v) in edge_colors.keys():
        edge_tuple = tuple(sorted((u, v)))
        if edge_tuple in temp_processed_edges:
            continue
        temp_processed_edges.add(edge_tuple)
        
        # Get the color for this edge.
        color = edge_colors[(u, v)]
        temp_node_colors[u].add(color)
        temp_node_colors[v].add(color)
    
    # Print the incident colors for each node.
    print("\n=== Node Incident Colors (Recalculated) ===")
    for node, colors in temp_node_colors.items():
        print(f"Node {node}: {colors}")
    
    # Validate that each node has at most max_allowed colors.
    for node, colors in temp_node_colors.items():
        if len(colors) > max_allowed:
            print(f"\n❌ ERROR: Node {node} has {len(colors)} incident colors (exceeds {max_allowed}).")
            raise ValueError(f"Node {node} incident colors {colors} exceed allowed limit ({max_allowed}).")
    
    print("\nAll nodes have at most", max_allowed, "incident colors.")
    print("=== End of Validate node incident colors ===\n")
# Example usage:
# validate_node_incident_colors(graph_nx, edge_colors, max_allowed=2)


#help function to check if both graphs are equivalent 
def are_graphs_equivalent(graph_a, graph_b):
    """
    Check whether two graphs are equivalent in an undirected format.

    Parameters:
        graph_a (networkx.Graph): First input graph.
        graph_b (networkx.Graph): Second input graph.

    Returns:
        bool: True if both graphs are equivalent, False otherwise.
    """
    # Convert both graphs to undirected form (if not already)
    graph_a = graph_a.to_undirected()
    graph_b = graph_b.to_undirected()

    # Get sorted edge sets (treat (a, b) the same as (b, a))
    edges_a = {tuple(sorted(edge)) for edge in graph_a.edges()}
    edges_b = {tuple(sorted(edge)) for edge in graph_b.edges()}

    # Check if both graphs have the same edges
    return edges_a == edges_b










# Example Usage
# #Creating a new graph 
# NUM_NODES = 10
# EDGE_PROB = 0.2

# # bidirectional_graph = create_bidirectional_graph(NUM_NODES, EDGE_PROB) 
# bidirectional_graph = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
# print("bidirectional graph", bidirectional_graph.edges(),"\n number of edges",bidirectional_graph.number_of_edges())
# undirected_graph = convert_to_undirected(bidirectional_graph)
# print("undirected graph", undirected_graph.edges(),"\n number of edges",undirected_graph.number_of_edges())
# gedges = undirected_graph.edges()
# print(gedges())
# graph_nx = nx.Graph()
# graph_nx.add_edges_from(gedges)

# # Execute restructuring process
# # outward_edges, root_node = execute_graph_restructuring(undirected_graph)

# # ✅ Convert outward edges into a graph and print it
# outward_graph = nx.DiGraph()
# outward_graph.add_edges_from(outward_edges)




# #Loading graph file
# # with open("/home/aravindb/Documents/a_Mtech/out_restructureed_graph.edgelist", "r") as f:
# #     loaded_edges = [tuple(map(int, line.strip().split())) for line in f]  # Read edges exactly as stored

# # # Create a new graph from the loaded edges
# # loaded_graph = nx.DiGraph()
# # loaded_graph.add_edges_from(loaded_edges)

# # print(f"🚀 The saved and loaded graph: {loaded_graph.edges()}")

# # print("the original graph",loaded_graph.edges())



# # undirected_graph = convert_to_undirected(loaded_graph)
# # print("undirected graph", undirected_graph.edges(),"\n number of edges",undirected_graph.number_of_edges())
# # gedges = undirected_graph.edges()
# # print(gedges())
# # graph_nx = nx.Graph()
# # graph_nx.add_edges_from(gedges)




# # Step X: Save the final 'graph_nx' to an edge list file for reference purposes
# # save_path = "/home/aravindb/Documents/a_Mtech/final_graph.edgelist"
# # nx.write_edgelist(graph_nx, save_path, data=False)

# # print(f"Graph saved to {save_path}")


# # Read the edge list back while preserving order
# # with open("/home/aravindb/Documents/a_Mtech/graph.edgelist", "r") as f:
# #     loaded_edges = [tuple(map(int, line.strip().split())) for line in f]  # Read edges exactly as stored

# # # Create a new graph from the loaded edges
# # graph_nx = nx.DiGraph()
# # graph_nx.add_edges_from(loaded_edges)

# print(f"🚀 The saved and loaded graph: {outward_graph.edges()}")

# print("the original graph", outward_graph.edges())
# # Run the multi-level coloring algorithm

# validate_node_incident_colors(outward_graph, edge_colors, max_allowed=2)

