import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
from networkx.readwrite import json_graph

# Load graph from the JSON file
with open("graph.json", "r") as f:
    G = json_graph.node_link_graph(json.load(f))  # Correctly load the graph

money_consumption_factor = 10
money_production_factor = 30
water_consumption_factor = 5
water_production_factor = 10
food_consumption_factor = 10
food_production_factor = 100
pollution_factor = 10
purification_factor = 10

Pollution = 0
Food = 0
Money = 0
Water = 0

# Iterate over each node in the graph
for node in G.nodes:
    attribute = G.nodes[node].get("attribute", "")
    weight = G.nodes[node].get("weight", 1)  # Default weight to 1 if not set

    if attribute == "residential area":
        G.nodes[node]["density"] = 10  
        G.nodes[node]["money consumption"] = G.nodes[node]["density"] * weight * money_consumption_factor
        G.nodes[node]["water consumption"] = G.nodes[node]["density"] * weight * water_consumption_factor
        G.nodes[node]["food consumption"] = G.nodes[node]["density"] * weight * food_consumption_factor
        
        Food-= G.nodes[node]["food consumption"]
        Money-= G.nodes[node]["money consumption"]
        Water-= G.nodes[node]["water consumption"]
        
    elif attribute == "industrial area":
        G.nodes[node]["pollution"] = weight * pollution_factor
        G.nodes[node]["money production"] = weight * money_production_factor
        
        Pollution+= G.nodes[node]["pollution"]
        Money+= G.nodes[node]["money production"]
        
    elif attribute == "water":
        G.nodes[node]["water production"] = weight * water_production_factor
        
        Water+= G.nodes[node]["water production"]
        
    elif attribute == "forest":
        G.nodes[node]["purification"] = purification_factor * weight
        
        Pollution-= G.nodes[node]["purification"]
        
    elif attribute == "farm":
        G.nodes[node]["food production"] = weight * food_production_factor
        
        Food+= G.nodes[node]["food production"]
        
        
plt.bar(["Food", "Water", "Money", "Pollution"], [Food, Water, Money, Pollution])
        
        
New_Pollution = 0
New_Food = 0
New_Money = 0
New_Water = 0   
             
        
