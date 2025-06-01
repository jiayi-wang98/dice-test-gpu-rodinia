import random

def generate_random_graph(n_nodes, edge_probability=0.1, output_file="random_graph.txt"):
    """
    Generate a random graph with n nodes and write it to a file.
    edge_probability: probability of an edge between any two nodes (0 to 1)
    """
    # Create adjacency list
    adj_list = [[] for _ in range(n_nodes)]
    
    # Generate random edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Start from i+1 to avoid self-loops and duplicates
            if random.random() < edge_probability:
                adj_list[i].append(j)
                adj_list[j].append(i)  # Undirected graph

    # Calculate edge list and offsets
    edge_list = []
    offsets = []
    current_offset = 0
    
    for i in range(n_nodes):
        num_edges = len(adj_list[i])
        offsets.append((current_offset, num_edges))
        edge_list.extend([(j, 1) for j in sorted(adj_list[i])]) # Cost of 1 for each edge
        current_offset += num_edges

    total_edges = len(edge_list)

    # Write to file
    with open(output_file, 'w') as f:
        # Number of nodes
        f.write(f"{n_nodes}\n")
        
        # Node offsets and edge counts
        for offset, num_edges in offsets:
            f.write(f"{offset} {num_edges}\n")
        
        f.write("\n")
        # Source node
        f.write("0\n\n")
        
        # Total edge list size
        f.write(f"{total_edges}\n")
        
        # Edge list with costs
        for dest, cost in edge_list:
            f.write(f"{dest} {cost}\n")

    print(f"Generated graph with {n_nodes} nodes and {total_edges} edges")
    print(f"Written to {output_file}")

# Example usage
if __name__ == "__main__":
    # Generate a 512-node graph with 10% edge probability
    generate_random_graph(30720, edge_probability=0.0002)