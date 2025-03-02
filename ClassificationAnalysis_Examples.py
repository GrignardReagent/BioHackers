import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import networkx as nx
import json

from scipy.ndimage import label

def correlatedNoise(n):
    """
    Generate artificial gaussianly correlated data
    
    input: side of the square
    output: artificial data asa numpy (n x n) matrix
    """
    # Compute filter kernel with radius correlation_scale
    correlation_scale = 20
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))
    
    # Generate n-by-n grid of spatially correlated noise
    noise = np.random.randn(n, n)
    noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
    
    return noise + noise.min()

# %% LOAD AND CLUSTER DATA
data_type = int(input("Enter the data: synthetic (1), colour clas. (2), GRAF clas. (3): "))

if data_type == 1:
    attributes = ["residential area", "industrial area", "water", "farm", "forest"]
    patched_data = np.empty([100, 100, 5])
    
    # Artifiical data
    for i in range(patched_data.shape[2]):
        patched_data[:, :, i] = correlatedNoise(patched_data.shape[0])
    
# Real data (Color)
if data_type == 2:
    attributes = ["residential area", "water", "farm"]
    data = np.loadtxt("label_matrix.csv", delimiter=",")
    
    patched_data = np.zeros([data.shape[0], data.shape[1], 3])
    
    human_indices = np.where(data==0)
    patched_data[human_indices[0],human_indices[1], 0] = 1
    human_indices = np.where(data==1)
    patched_data[human_indices[0],human_indices[1], 1] = 1
    human_indices = np.where(data==2)
    patched_data[human_indices[0],human_indices[1], 2] = 1
    
    # new shape
    reduction = 2
    if data.shape[0]%4!=0:
        print("Use a divisible number for the reduction")
    else:
        new_shape = (patched_data.shape[0] // 4, patched_data.shape[1] // 4, 3)

    # Reshape to (new_rows, 4, new_cols, 4, 3) and compute the mean over (4,4) blocks
    patched_data = patched_data.reshape(new_shape[0], 4, new_shape[1], 4, 3).mean(axis=(1, 3))

# Ral data (GRAF)
if data_type == 3:
    with open("ClassificationGRAF.json", "r") as file:
        data = json.load(file)  # data is now a Python dictionary
    
    side = int(np.sqrt(len(data)))
    images =  list(data.keys()) 
    attributes = list(data["image_1"]["scores"].keys()) 
    patched_data = np.zeros([side, side, len(attributes)])
    
    for i in range(len(images)):
        for j in range(len(attributes)):
            patched_data[i%side, i//side, j] = data[images[i]]["scores"][attributes[j]]

# Identify the main attrbute and compute similary with other attributes
number_modes = np.zeros_like(patched_data, dtype=float)

for i in range(patched_data.shape[0]):
    for j in range(patched_data.shape[1]):
        ordered_attributes = np.argsort(patched_data[i, j, :])[::-1]
        
        # Similarity with i-th attribute
        for k in range(patched_data.shape[2]):
            number_modes[i, j, k] = patched_data[i, j, ordered_attributes[0]] - patched_data[i, j, ordered_attributes[k]]
        
        # Main attribute
        patched_data[i, j, :] = 0
        patched_data[i, j, ordered_attributes[0]] = 1

# Identify the clusters of main attributes
clustered_data = np.zeros_like(patched_data, dtype=int)
num_clusters = np.empty(patched_data.shape[2], dtype=int)

for i in range(patched_data.shape[2]): 
    clustered_data[:, :, i], _ = label(patched_data[:, :, i]) 
    if i>0:
        cluster_indices = np.where(clustered_data[:, :, i]!=0)
        clustered_data[cluster_indices[0], cluster_indices[1], i]+= clustered_data[:, :, i-1].max()

aggregated_clusters = np.sum(clustered_data, axis=2)

# %% PLOTTING

# First Figure: Binary classification and clustering results
fig0, axes0 = plt.subplots(2, patched_data.shape[2], figsize=(12, 6))

for i in range(patched_data.shape[2]):
    # Binary classification
    axes0[0, i].imshow(patched_data[:, :, i], cmap='viridis', vmin=0, vmax=1)
    axes0[0, i].set_title(attributes[i])
    axes0[0, i].axis('off')

    # Clustered result
    axes0[1, i].imshow(clustered_data[:, :, i], cmap='tab20')
    
    # Loop through each cluster in the current attribute and display its ID at the center of mass
    labeled_clusters = clustered_data[:, :, i]
    flattened_labels = labeled_clusters.flatten()
    sorted_labels = np.unique(np.sort(flattened_labels))
    for cluster_id in range(sorted_labels[1], sorted_labels[-1] + 1):
        # Find all the pixel positions for the current cluster
        positions = np.argwhere(labeled_clusters == cluster_id)
        
        # Calculate the average x and y positions (center of mass)
        avg_position = np.mean(positions, axis=0)
        avg_x, avg_y = avg_position[0], avg_position[1]
        
        # Annotate the cluster ID at the center of mass
        axes0[1, i].text(avg_y, avg_x, str(cluster_id), color='white', ha='center', va='center', fontsize=8)
    
    axes0[1, i].axis('off')

fig0.suptitle("Binary Classification (Top) & Clustering (Bottom)", fontsize=14)
plt.tight_layout()

# Second Figure: Difference in Modes
fig1, axes1 = plt.subplots(1, patched_data.shape[2], figsize=(12, 3))

for i in range(patched_data.shape[2]):
    axes1[i].imshow(-number_modes[:, :, i], cmap='Reds', vmin=-number_modes.max(), vmax=0) 
    axes1[i].set_title(f"Mode {i+1}")
    axes1[i].axis('off')

fig1.suptitle("Multimodal classification", fontsize=14)
plt.tight_layout()



# %% TRANSFORM INTO NETWORK

# Initialize the graph
G = nx.Graph()

# Define a color map for the attributes
attribute_colors = {
    "residential area": "tab:cyan",
    "industrial area": "tab:grey",
    "water": "tab:blue",
    "forest": "tab:green",
    "farm": "tab:olive"
}


# Loop through each attribute's clustered data
for attr_index in range(patched_data.shape[2]):
    # Get the labeled clusters for the current attribute
    labeled_clusters = clustered_data[:, :, attr_index]
    flattened_labels = labeled_clusters.flatten()
    sorted_labels = np.unique(np.sort(flattened_labels))
    for cluster_id in range(sorted_labels[1], sorted_labels[-1] + 1):
        # Find all the pixel positions for the current cluster
        positions = np.argwhere(labeled_clusters == cluster_id)
        
        # Calculate the average x and y positions (center of mass)
        avg_position = np.mean(positions, axis=0)
        avg_x, avg_y = avg_position[0], avg_position[1]
        
        # Calculate the area (number of pixels in the cluster)
        area = positions.shape[0]

        # Add node to the graph with attributes: position, area, and attribute
        G.add_node(cluster_id, 
                    pos=(avg_x, avg_y), 
                    weight=area, 
                    attribute=attributes[attr_index],
                    color=attribute_colors[attributes[attr_index]])

        
# Function to find the boundary pixels of a cluster
def get_contacting_pixels(cluster_id1, cluster_id2, aggregated_clusters):
    """
    Get the number of contacting pixels between two clusters.
    This function checks if the two clusters share any boundary pixels.
    """
    contacting_pixels = 0

    # Iterate over all positions of the first cluster
    positions1 = np.argwhere(aggregated_clusters == cluster_id1)
    for x, y in positions1:
        # Check all 8 neighbors (N, NE, E, SE, S, SW, W, NW)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the center pixel
                nx, ny = x + dx, y + dy
                # Check if neighbor is within bounds
                if 0 <= nx < aggregated_clusters.shape[0] and 0 <= ny < aggregated_clusters.shape[1]:
                    # If the neighbor belongs to the other cluster, we have a contacting pixel
                    if aggregated_clusters[nx, ny] == cluster_id2:
                        contacting_pixels += 1
                        break  # Once we find a neighbor, we don't need to check further neighbors for this pixel
    
    return contacting_pixels


# Add edges between adjacent clusters of different attributes
for cluster_id1 in range(1,aggregated_clusters.max()+1):
    for cluster_id2 in range(cluster_id1 + 1, aggregated_clusters.max()+1):  # Only check clusters of different attributes
        # Get the number of contacting pixels between the two clusters
        contacting_pixels = get_contacting_pixels(cluster_id1, cluster_id2, aggregated_clusters)
        if contacting_pixels > 0:
            node_id1 = cluster_id1
            node_id2 = cluster_id2
            # Create an edge between the two clusters with the contacting pixel count as the weight
            G.add_edge(node_id1, node_id2, weight=contacting_pixels)

# Get the node weights from the graph
node_weights = nx.get_node_attributes(G, 'weight')
max_weight = max(node_weights.values())
min_weight = min(node_weights.values())

# Scale nodes size
def scale_node_size(weight, min_weight, max_weight, min_size=200, max_size=2000):
    return min_size + (weight - min_weight) * (max_size - min_size) / (max_weight - min_weight)

node_sizes = [scale_node_size(node_weights[node], min_weight, max_weight) for node in G.nodes]

edge_weights = [G[u][v]['weight'] for u, v in G.edges]
min_edge_weight = min(edge_weights)
max_edge_weight = max(edge_weights)

# Scale edge sizes
def scale_edge_width(weight, min_weight, max_weight, min_width=1, max_width=5):
    return min_width + (weight - min_weight) * (max_width - min_width) / (max_weight - min_weight)

edge_widths = [scale_edge_width(weight, min_edge_weight, max_edge_weight) for weight in edge_weights]


# Compute layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)  # Adjust k and iterations for tighter spacing

# If needed, apply additional constraints to pull disconnected nodes closer
for node in G.nodes:
    if G.degree(node) == 0:  # If node is disconnected
        pos[node] = np.mean([pos[n] for n in G.nodes if G.degree(n) > 0], axis=0) + np.random.rand(2) * 0.1

# Draw the network
plt.figure(figsize=(5, 5))
nx.draw(G, 
        pos=nx.get_node_attributes(G, 'pos'),  # Use the center of mass for node positions
        with_labels=True, 
        labels={node: f"{node}" for node in G.nodes},  # Display cluster ID
        node_size=node_sizes,  # Use the calculated node sizes based on cluster area
        node_color=[G.nodes[node]['color'] for node in G.nodes],  # Color by attribute classification
        font_size=10, 
        font_weight='bold', 
        edge_color='black', 
        width=edge_widths,  # Edge widths based on number of contacting pixels
        node_shape='o', 
        alpha=1.0)  # Set transparency to avoid overlapping nodes


# Show the plot
plt.show()

graph_data = nx.node_link_data(G)  # Convert graph to dictionary
with open("graph.json", "w") as f:
    json.dump(graph_data, f)

