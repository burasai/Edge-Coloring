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









#Functions to restrucutre graph outwards with respect to root node(highest degree node)
# ✅  restrucutre outwards  : Select Root Node
def select_root_node(graph):
    """
    Selects the highest-degree node as the root.
    """
    max_degree = max(graph.degree(), key=lambda x: x[1])[1]
    candidate_nodes = [node for node, degree in graph.degree() if degree == max_degree]
    root_node = random.choice(candidate_nodes)
    
    print(f"\n🔹 **Selected Root Node:** {root_node} (Degree: {max_degree})")
    
    return root_node

# ✅ restrucutre outwards  Step : Process d_i → d_{i+1} and d_{i+1} → d_{i+1} Edges
def process_next_level_edges(graph, current_level_nodes, unprocessed_edges, processed_edges, outward_edges):
    """
    Processes d_i → d_{i+1} edges, restructures them, and adds them to outward edges.
    Also identifies and processes d_{i+1} → d_{i+1} edges.
    """
    next_level_nodes = set()
    next_level_edges = set()

    # ✅ Find d_i → d_{i+1} edges
    for u in current_level_nodes:
        for v in graph.neighbors(u):
            edge = (u, v) if (u, v) in unprocessed_edges else (v, u)
            if edge in unprocessed_edges:
                next_level_edges.add(edge)

    # ✅ Process d_i → d_{i+1} edges
    for edge in next_level_edges:
        src, dst = edge
        structured_edge = (src, dst) if src in current_level_nodes else (dst, src)
        
        outward_edges.add(structured_edge)
        processed_edges.add(edge)
        unprocessed_edges.remove(edge)
        next_level_nodes.add(dst if src in current_level_nodes else src)

        print(f"   ✅ Edge Added: {structured_edge}")  # ✅ FIXED ERROR

    # ✅ Find and process d_{i+1} → d_{i+1} edges
    d1_d1_edges = {edge for edge in unprocessed_edges if edge[0] in next_level_nodes and edge[1] in next_level_nodes}
    for edge in d1_d1_edges:
        outward_edges.add(edge)
        processed_edges.add(edge)
        unprocessed_edges.remove(edge)
        print(f"   ✅ Edge Added (d_{len(processed_edges) // 2} → d_{len(processed_edges) // 2}): {edge}")

    return next_level_nodes, outward_edges, processed_edges, unprocessed_edges

# ✅ Step : Execute Graph Restructuring main 
def execute_graph_restructuring(graph):
    """
    Main function to perform outward restructuring of the graph.
    """
    # ✅ Initialize tracking lists
    unprocessed_edges = set(graph.edges())
    processed_edges = set()
    outward_edges = set()

    # ✅ Select root node
    root_node = select_root_node(graph)

    # ✅ Step 1: Process Root Node d0 → d1
    current_level_nodes = {root_node}
    while unprocessed_edges:
        print(f"\n🟢 **Processing Level {len(processed_edges) // 2 + 1}**")
        print(f"   → Current Level Nodes: {list(current_level_nodes)}")
        print(f"   → Queue Before Clearing: {list(current_level_nodes)}")
        print(f"   → Unprocessed Edges: {unprocessed_edges}")
        print(f"   → Processed Edges: {processed_edges}")

        # ✅ Process d_i → d_{i+1} edges and d_{i+1} → d_{i+1} edges
        next_level_nodes, outward_edges, processed_edges, unprocessed_edges = process_next_level_edges(
            graph, current_level_nodes, unprocessed_edges, processed_edges, outward_edges
        )

        print(f"   → Next-Level Nodes Added to Queue: {list(next_level_nodes)}")
        print(f"   → Queue After Clearing: {list(next_level_nodes)}")

        # ✅ Move to next level
        current_level_nodes = next_level_nodes

        if not next_level_nodes:
            print("\n✅ **Graph Processing Complete!** No more levels to process.")
            break

    return outward_edges, root_node





#Method for segregating edges of a  root node (Used in Greedy Multi level coloring)
def classify_edges(root_nodes, edge_list):
    """
    Classifies edges into linked edges (d_i → d_{i+1}) and interlinked edges (d_i → d_i+1 and d_i+1 → d_i+1).
    
    Parameters:
        root_nodes: The root nodes from which edges are considered.
        edge_list: List of edges (tuples) representing the graph.
        
    Returns:
        linked_edges: List of direct connections from root nodes.
        interlinked_edges: List of edges between root nodes' neighbors.
        level_level_edges: Remaining edges connecting d_{i+1} → d_{i+1}.
    """
    linked_edges = []      # Direct edges from root nodes to next level (only d_i → d_{i+1})
    interlinked_edges = [] # Edges among d_i+1 nodes or involving d_i → d_{i+1} with further connections
    temp_list = []         # Temporary list for d_i → d_{i+1} edges
    node_list = set()      # Nodes in d_{i+1}
    level_level_edges = [] # Remaining edges that are not connected to root_nodes

    # Step 1: Identify all edges connected to `root_nodes`
    for u, v in edge_list:
        if u in root_nodes or v in root_nodes:  # If either node is a root node
            temp_list.append((u, v))  # Store the edge for classification
            node_list.add(v if u in root_nodes else u)  # Extract d_{i+1} nodes
        else:
            level_level_edges.append((u, v))  # Remaining edges

    # Step 2: Classify edges in temp_list
    for x, y in temp_list:
        is_interlinked = any(y in edge for edge in level_level_edges)  # Check if y has connections

        if is_interlinked:
            interlinked_edges.append((x, y))
        else:
            linked_edges.append((x, y))

    

    
    print("==============Final++++++++++ segregation====================")
    print(f"Linked Edges: {linked_edges}")
    print(f"Interlinked Edges: {interlinked_edges}")
    print(f"level_level_edges: {level_level_edges}")

    return linked_edges, interlinked_edges, level_level_edges



#main greedy method for coloring
def greedy_multi_level_coloring(graph_nx, root_node):

    graph_dict = build_graph_dict(graph_nx)
    """
    Multi-level edge coloring ensuring:
      - Generalized for all levels (d0-d1, d1-d2, d2-d3, ... until all edges are colored).
      - Strictly enforces the 2-incident color rule.
      - Same coloring rules apply for d_i-d_{i+1} and d_{i+1}-d_{i+1} edges.
      - Stops only when all edges are colored.
      - Prints debugging information at every level.
    
    Parameters:
        graph_nx: Undirected graph (NetworkX format)

    Returns:
        edge_colors: Dictionary mapping edges to assigned colors.
        node_colors: Dictionary mapping nodes to their incident colors.
        uncolored_edges: List of edges that couldn't be assigned a valid color.
    """

    # Step 1: Initialize data structures
    edge_colors = {}  # Stores assigned colors for each edge
    # node_colors = {v: set() for v in graph_nx.nodes()}  # Tracks colors used at each node
    node_colors = {v: set() for v in sorted(graph_nx.nodes())}

    available_colors = iter(range(1, len(graph_nx.edges()) + 1))  # Assign colors sequentially
    uncolored_edges = []  # Stores edges that couldn't be assigned a color
    level = 1  # Track levels of processing
    processed_edges = set()  # Set to track edges that have been colored
    unprocessed_edges = set(graph_nx.edges())  # Initially, all edges are unprocessed
    removed_colors = set()

    print("\n===========================================")
    print(f"📌 **Total Number of Edges in the Graph:** {len(graph_nx.edges())}")
    print("===========================================\n")





    # Step 2: Select the highest-degree node as the primary root (d0)
    # root_nodes = {max(graph_nx.nodes(), key=lambda v: graph_nx.degree(v))}
    # root_nodes = {6}
    root_nodes = {root_node}  #input parameter 
    print(f"root_node {root_node}")
    #input("check root node intially")
    print("root_nodes", root_nodes, "type", type(root_nodes))
    # input("please check then remove fixed root node if present ")

    global_next_level_nodes = set()  # <--- (1) NEW variable outside the "while root_nodes" loop
    
    level = 1



    print("\n================GLOBAL VARIABLES===========================")
    # print(f"processed_edges type : {processed_edges.type()}")
    # print(f"unprocessed_edges type : {unprocessed_edges.type()}")
    # print(f"node colors type {node_colors.type()}")
    # print(f"edge_colors type : {edge_colors.type()}")
    print("===========================================\n")
    while root_nodes:
        print(f"\n🟢 **Processing Level {level}** (d{level-1} → d{level})")
        print(f"   → **Root Nodes (d{level-1}):** {root_nodes}")

        # For THIS level, we will iterate over each current root separately,
        # and gather new next-level nodes into global_next_level_nodes.
        global_next_level_nodes.clear()  # Ensure empty at start of the level

        # Process EACH root node one by one
        for single_root in list(root_nodes):
            print(f"\n   🔸 **Now Processing Single Root Node:** {single_root}")

            # Step 3: Identify all (single_root - d_{i+1}) edges
            level_edges = [
                (u, v) for (u, v) in unprocessed_edges
                if (u == single_root or v == single_root)
            ]

            if not level_edges:
                print(f"      → No uncolored edges for this single root ({single_root}).")
                continue

            print(f"      → **Uncolored Edges Connected to {single_root}:** {level_edges}")

            # Identify d_{i+1} nodes from this single root
            next_level_nodes = set()
            for src, dst in level_edges:
                # Whichever endpoint is *not* the single_root is the next-level node
                if src == single_root:
                    next_level_nodes.add(dst)
                else:
                    next_level_nodes.add(src)

            # Identify d_{i+1}-d_{i+1} edges (preserve original graph order)
            next_level_edges = []
            for (u, v) in unprocessed_edges:
                if u in next_level_nodes and v in next_level_nodes \
                   and (u, v) not in edge_colors and (v, u) not in edge_colors:
                    next_level_edges.append((u, v))

            print(f"      → **D_{level} Nodes Identified from {single_root}:** {next_level_nodes}")
            print(f"      → **D_{level} → D_{level} Edges Identified:** {next_level_edges}")

            # Combine single_root → d_{i+1} and d_{i+1} → d_{i+1} edges
            all_edges = list(level_edges) + list(next_level_edges)

            # Classify edges
            linked_edges, interlinked_edges, level_level_edges = classify_edges(
                {single_root}, all_edges
            )
            
            print("      ===========================================================")
            print(f"         → **Linked Edges ({single_root} → d_{level}):** {linked_edges}")
            print(f"         → **Interlinked Edges (d_{level} → d_{level}):** {interlinked_edges}")
            print(f"         → **level_level Edges (d_{level} → d_{level}):** {level_level_edges}")
            print("      ===========================================================")

            # Step 4: Color function (unchanged):
            
            def color_edge(src, dst, edge_type):
                print(f"\n🟡 **Processing {edge_type} Edge:** {src} → {dst}")
                print(f"   - Node {src} Incident Colors BEFORE: {node_colors[src]}")
                print(f"   - Node {dst} Incident Colors BEFORE: {node_colors[dst]}")

                if len(node_colors[src]) < 2 and len(node_colors[dst]) < 2:
                    color = next(available_colors)
                    decision = f"✅ Introducing NEW Color {color}"
                elif len(node_colors[src]) == 2 and len(node_colors[dst]) == 2:
                    common_colors = node_colors[src] & node_colors[dst]
                    print(f"common colors  {common_colors}")
                    if common_colors:
                        color = next(iter(common_colors))
                        decision = f"✅ Using Common Color {color}"
                    else:
                        print(f"\n❌❌❌❌❌❌❌ Main ERROR in src has 2 incident colors and  dst has 2 incident colors : No common color found for edge {src} → {dst}!")
                        #raise ValueError(f"Edge {src} → {dst} has no common color! Stopping execution.")
                        reconcile_incident_colors(node_colors[src],node_colors[dst])
                        return

                    
                elif len(node_colors[src]) == 2:
                    possible_colors = node_colors[src] - node_colors[dst]
                    print(f" only src colors =2 \n possible colors  {possible_colors}")
                    if possible_colors:
                        color = next(iter(possible_colors))
                        decision = f"✅ Using Color {color} from src {src}"
                    else:
                        print(f"\n❌ ERROR in src has 2 incident colors but not dst : No valid color available from src {src} for edge {src} → {dst}!")
                        uncolored_edges.append((src, dst))
                        raise ValueError(f"Edge {src} → {dst} has no common color! Stopping execution.")
                        # return
                elif len(node_colors[dst]) == 2:
                    possible_colors = node_colors[dst] - node_colors[src]
                    print(f" only dst colors =2 \n possible colors  {possible_colors}")
                    if possible_colors:
                        color = next(iter(possible_colors))
                        decision = f"✅ Using Color {color} from dst {dst}"
                    else:
                        print(f"\n❌ ERROR in dst has 2 incident colors but not src: No valid color available from dst {dst} for edge {src} → {dst}!")
                        uncolored_edges.append((src, dst))
                        raise ValueError(f"Edge {src} → {dst} has no common color! Stopping execution.")
                        # return

                edge_colors[(src, dst)] = color
                edge_colors[(dst, src)] = color
                node_colors[src].add(color)
                node_colors[dst].add(color)
                print(f"   {decision}")



            def reconcile_incident_colors(src_colors, dst_colors):
                """
                Given two sets of incident colors (src_colors and dst_colors) for two nodes,
                choose one color from each (here we choose the larger element from each set, e.g. b and d).
                Then, determine which of these is greater (higher_color) and which is lower (lower_color).
                
                Next, for every edge in processed_edges that has the higher_color assigned, 
                replace that color with the lower_color in the edge_colors dictionary and update node_colors for both endpoints.
                
                Finally, since the higher_color is no longer in use, prepend it to available_colors for future use.
                
                Global variables assumed:
                - processed_edges: set of edges (tuples) already colored
                - edge_colors: dict mapping edge (u,v) to its color (also stored for (v,u))
                - node_colors: dict mapping each node to a set of incident colors
                - available_colors: a list of available colors (so we can insert at index 0)
                """

                print("\n=== Reconcile Incident Colors ===")
                print("Source colors:", src_colors)
                print("Destination colors:", dst_colors)

                # Convert to sorted lists (assumes each has exactly 2 elements)
                src_sorted = sorted(src_colors)
                dst_sorted = sorted(dst_colors)
                
                # For our procedure, we choose the second element (largest) from each set.
                chosen_src = src_sorted[-1]  # e.g. b from {a, b}
                chosen_dst = dst_sorted[-1]  # e.g. d from {c, d}
                print(f"Chosen colors: from src = {chosen_src}, from dst = {chosen_dst}")

                # Determine which is higher and which is lower
                if chosen_src > chosen_dst:
                    higher_color = chosen_src
                    lower_color = chosen_dst
                else:
                    higher_color = chosen_dst
                    lower_color = chosen_src
                print(f"Determined higher_color = {higher_color}, lower_color = {lower_color}")

                # Now, for every processed edge, if its assigned color is higher_color,
                # replace it with lower_color and update node_colors for the endpoints.
                for edge in list(processed_edges):  # iterate over a copy since we may update things
                    # Check if this edge has been colored with higher_color.
                    if edge in edge_colors and edge_colors[edge] == higher_color:
                        print(f"Updating edge {edge}: replacing {higher_color} with {lower_color}")
                        # Update edge color for both (u,v) and its reverse (v,u)
                        edge_colors[edge] = lower_color
                        rev_edge = (edge[1], edge[0])
                        if rev_edge in edge_colors and edge_colors[rev_edge] == higher_color:
                            edge_colors[rev_edge] = lower_color

                        # Update the node_colors for both endpoints, if higher_color is present
                        u, v = edge
                        if higher_color in node_colors[u]:
                            node_colors[u].remove(higher_color)
                            node_colors[u].add(lower_color)
                            print(f"   Node {u} incident colors updated to: {node_colors[u]}")
                        if higher_color in node_colors[v]:
                            node_colors[v].remove(higher_color)
                            node_colors[v].add(lower_color)
                            print(f"   Node {v} incident colors updated to: {node_colors[v]}")

                # Finally, the higher_color is no longer in use, so add it to the beginning of available_colors.
                # available_colors.insert(0, higher_color)
                try:
                    available_colors.insert(0, higher_color)
                    
                except AttributeError:
                    # available_colors is an iterator and does not support insert;
                    # simply ignore recycling the higher_color.
                    pass
                removed_colors.add(higher_color)
                print(f"Recycled higher_color {higher_color} to available_colors at the beginning.")
                print("Updated available_colors:", available_colors)
                print("====================After changing the colors (removing a color and adjusting ) validating =======================")
                validate_node_incident_colors(graph_nx, edge_colors, max_allowed=2)
                print("=== End of Reconciliation ===\n")





            # Step 5: Process linked edges
            # for (src, dst) in linked_edges:
            #     if (src, dst) in unprocessed_edges or (dst, src) in unprocessed_edges:
            #         color_edge(src, dst, edge_type=f"d{level-1}-d{level} (Linked)")
            #         unprocessed_edges.discard((src, dst))
            #         unprocessed_edges.discard((dst, src))
            #         processed_edges.add((src, dst))
            #         processed_edges.add((dst, src))

            # # Step 6: Process interlinked edges and related level_level_edges
            # for (src, dst) in interlinked_edges:
            #     if (src, dst) in unprocessed_edges or (dst, src) in unprocessed_edges:
            #         color_edge(src, dst, edge_type=f"d{level}-d{level} (Interlinked)")
            #         unprocessed_edges.discard((src, dst))
            #         unprocessed_edges.discard((dst, src))
            #         processed_edges.add((src, dst))
            #         processed_edges.add((dst, src))

            #     # Check for related edges in level_level_edges
            #     for (u, v) in level_level_edges:
            #         if dst in (u, v):
            #             if (u, v) in unprocessed_edges or (v, u) in unprocessed_edges:
            #                 color_edge(u, v, edge_type=f"d{level}-d{level} (Related Interlinked)")
            #                 unprocessed_edges.discard((u, v))
            #                 unprocessed_edges.discard((v, u))
            #                 processed_edges.add((u, v))
            #                 processed_edges.add((v, u))
            
        
            # ===============================================================================
            # ---- Step A: Color all linked edges first
            
            for (src, dst) in linked_edges:
                if (src, dst) in unprocessed_edges or (dst, src) in unprocessed_edges:
                    color_edge(src, dst, edge_type=f"linked edge d{level-1}-d{level}")
                    unprocessed_edges.discard((src, dst))
                    unprocessed_edges.discard((dst, src))
                    processed_edges.add((src, dst))
                    processed_edges.add((dst, src))

            # ---- Step B: Process interlinked edges, one by one
            # We'll build a new list because we might color them out of order
            remaining_interlinked = list(interlinked_edges)

            for (src, dst) in remaining_interlinked:
                if (src, dst) not in unprocessed_edges and (dst, src) not in unprocessed_edges:
                    # Already colored or removed
                    continue

                # ---- B1: Identify which node is x_node if single_root is in (src, dst)
                #     Otherwise, we color as normal but won't do the x_node logic.
                if single_root == src:
                    x_node = dst
                elif single_root == dst:
                    x_node = src
                else:
                    x_node = None  # single_root is not part of this edge

                # ---- B2: Color this interlinked edge
                if (src, dst) in unprocessed_edges or (dst, src) in unprocessed_edges:
                    color_edge(src, dst, edge_type=f"Interlinked edge d{level-1}-d{level}")
                    unprocessed_edges.discard((src, dst))
                    unprocessed_edges.discard((dst, src))
                    processed_edges.add((src, dst))
                    processed_edges.add((dst, src))

                # ---- B3: If x_node is defined, handle level_level edges of x_node
                if x_node is not None:
                    # (i) Identify edges from level_level_edges that contain x_node
                    temp_edge_list = [
                        (u, v) for (u, v) in level_level_edges
                        if x_node in (u, v)
                    ]

                    # (ii) We "give more importance" to neighbors in interlinked_edges
                    #     i.e., see if x_node forms interlinked edges with any other node
                    priority_edges = [
                        (u, v) for (u, v) in interlinked_edges
                        if x_node in (u, v) and
                        ((u, v) in unprocessed_edges or (v, u) in unprocessed_edges)
                    ]
                    # Example approach: color these priority_edges first
                    for (u, v) in priority_edges:
                        if (u, v) in unprocessed_edges or (v, u) in unprocessed_edges:
                            color_edge(src, dst, edge_type=f"Priority interlinked edge d{level-1}-d{level}")
                            unprocessed_edges.discard((u, v))
                            unprocessed_edges.discard((v, u))
                            processed_edges.add((u, v))
                            processed_edges.add((v, u))

                    # (iii) Now color the temp_edge_list
                    for (u, v) in temp_edge_list:
                        if (u, v) in unprocessed_edges or (v, u) in unprocessed_edges:
                            color_edge(src, dst, edge_type=f"Related level level edge d{level}-d{level}")
                            unprocessed_edges.discard((u, v))
                            unprocessed_edges.discard((v, u))
                            processed_edges.add((u, v))
                            processed_edges.add((v, u))







            # (7) Add next-level nodes from this single root to the global set
            global_next_level_nodes.update(next_level_nodes)

        # After processing **all** root nodes at this level:
        print("\n🔢 **Node Incident Colors After Level Completion:**")
        for node, colors in node_colors.items():
            print(f"   🔹 Node {node} → {len(colors)} incident colors: {colors}")

        # Now update root_nodes for the next iteration
        print(f"\n🔀 Updating root_nodes from global_next_level_nodes.")
        print(f"global_next_level_nodes at level {level}  :{global_next_level_nodes}")
        root_nodes = global_next_level_nodes.copy()
        global_next_level_nodes.clear()  # ready for the next level
        level += 1
    print(f"removed Colors :{removed_colors}")

    return edge_colors, node_colors, uncolored_edges, removed_colors



# # Step 1: Create a Strictly Bidirectional Graph`
# def create_bidirectional_graph(num_nodes, edge_prob):
#     """
#     Create a strictly bidirectional graph (directed edges in both directions).
#     """
#     graph_nx = nx.gnm_random_graph(num_nodes, int(edge_prob * num_nodes * (num_nodes - 1)), directed=True)
#     for u, v in list(graph_nx.edges):
#         graph_nx.add_edge(v, u)  # Ensure bidirectional edges
#     print("\nBidirectional Graph Edges:")
#     print(list(graph_nx.edges))
#     return graph_nx








#Creating a new graph 
NUM_NODES = 10
EDGE_PROB = 0.2

# bidirectional_graph = create_bidirectional_graph(NUM_NODES, EDGE_PROB) 
bidirectional_graph = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
print("bidirectional graph", bidirectional_graph.edges(),"\n number of edges",bidirectional_graph.number_of_edges())
undirected_graph = convert_to_undirected(bidirectional_graph)
print("undirected graph", undirected_graph.edges(),"\n number of edges",undirected_graph.number_of_edges())
gedges = undirected_graph.edges()
print(gedges())
graph_nx = nx.Graph()
graph_nx.add_edges_from(gedges)

# Execute restructuring process
outward_edges, root_node = execute_graph_restructuring(undirected_graph)

# ✅ Convert outward edges into a graph and print it
outward_graph = nx.DiGraph()
outward_graph.add_edges_from(outward_edges)




#Loading graph file
# with open("/home/aravindb/Documents/a_Mtech/out_restructureed_graph.edgelist", "r") as f:
#     loaded_edges = [tuple(map(int, line.strip().split())) for line in f]  # Read edges exactly as stored

# # Create a new graph from the loaded edges
# loaded_graph = nx.DiGraph()
# loaded_graph.add_edges_from(loaded_edges)

# print(f"🚀 The saved and loaded graph: {loaded_graph.edges()}")

# print("the original graph",loaded_graph.edges())



# undirected_graph = convert_to_undirected(loaded_graph)
# print("undirected graph", undirected_graph.edges(),"\n number of edges",undirected_graph.number_of_edges())
# gedges = undirected_graph.edges()
# print(gedges())
# graph_nx = nx.Graph()
# graph_nx.add_edges_from(gedges)




# Step X: Save the final 'graph_nx' to an edge list file for reference purposes
# save_path = "/home/aravindb/Documents/a_Mtech/final_graph.edgelist"
# nx.write_edgelist(graph_nx, save_path, data=False)

# print(f"Graph saved to {save_path}")


# Read the edge list back while preserving order
# with open("/home/aravindb/Documents/a_Mtech/graph.edgelist", "r") as f:
#     loaded_edges = [tuple(map(int, line.strip().split())) for line in f]  # Read edges exactly as stored

# # Create a new graph from the loaded edges
# graph_nx = nx.DiGraph()
# graph_nx.add_edges_from(loaded_edges)

print(f"🚀 The saved and loaded graph: {outward_graph.edges()}")

print("the original graph", outward_graph.edges())
# Run the multi-level coloring algorithm
edge_colors, node_colors, uncolored_edges, removed_colors = greedy_multi_level_coloring(outward_graph, root_node)

print("\n================================ FINAL RESULTS ================================")
print(f"Edge Colors: {edge_colors}")
for i, ((u, v), color) in enumerate(edge_colors.items(), start=1):
    print(f"{i}. ({u}, {v}): {color}")

print("Node incident colors")
for node, colors in node_colors.items():
            print(f"   🔹 Node {node} → {len(colors)} incident colors: {colors}")

print(f"==============Final Validation Check==============================================")

validate_node_incident_colors(outward_graph, edge_colors, max_allowed=2)

i=1
print("Uncolered Edges")
for edge in uncolored_edges:
    print(f"{i}.  {edge}")
    i = i+1

# Collect unique colors from all nodes
unique_colors = set()
for colors in node_colors.values():
    unique_colors.update(colors)

# Print the unique colors and their count
print("Unique colors used:", unique_colors)
print("Total number of unique colors:", len(unique_colors))
print(f"removed_colors {removed_colors}")



print("================================ The END ======================================")
