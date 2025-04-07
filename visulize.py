import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Load the DTDG .pkl file
with open("output_dtdg_2graphs.pkl", "rb") as f:
    dtdg_graphs = pickle.load(f)

# Unpack: if each item is (timestamp, graph)
_, G = dtdg_graphs[0]

# Try to use real positions if available
positions = {}
sample_node = list(G.nodes(data=True))[0][1]
if 'position' in sample_node:
    for nid, attr in G.nodes(data=True):
        positions[nid] = tuple(attr['position'])[:2]  # x, y for 2D
else:
    positions = nx.spring_layout(G)

# Draw it
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, positions, node_color='skyblue', node_size=300)
nx.draw_networkx_edges(G, positions, edge_color='gray')
nx.draw_networkx_labels(G, positions, font_size=8, font_color='black')

plt.title("DTDG Snapshot: Timestamp 0")
plt.axis("off")
plt.tight_layout()
plt.show()
print("Number of edges in this snapshot:", G.number_of_edges())
print("Edges:", list(G.edges()))
