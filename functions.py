import pandas as pd
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import heapq
import folium
from folium import PolyLine
import random

### 2.1 ###

def analyze_centrality(flight_network, airport):
    """
    Analyzes the centrality of a given airport in a flight network.

    Args:
        flight_network (graph): A graph representing the flight network, where nodes are airports and edges are lists of neighboring airports.
        airport (str): The airport to analyze.

    Returns:
        dict: A dictionary containing the following centrality measures:
            - Betweenness Centrality
            - Closeness Centrality
            - Degree Centrality
            - PageRank
    """

    def bfs_shortest_paths(source):
        """
        Calculates the shortest path distances and number of shortest paths from a source node to all other nodes in the network.

        Args:
            source (str): The source node.

        Returns:
            tuple: A tuple containing two dictionaries:
                - distances: A dictionary mapping nodes to their shortest path distances from the source.
                - num_paths: A dictionary mapping nodes to the number of shortest paths from the source.
        """
        queue = deque([source])
        distances = {node: float('inf') for node in flight_network}
        distances[source] = 0
        num_paths = {node: 0 for node in flight_network}
        num_paths[source] = 1

        while queue:
            current = queue.popleft()
            for neighbor in flight_network[current]:
                # If the shortest path has been found
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
                # Count the shortest paths
                if distances[neighbor] == distances[current] + 1:
                    num_paths[neighbor] += num_paths[current]

        return distances, num_paths

    def compute_betweenness():
        """
        Calculates the betweenness centrality of the given airport.

        Returns:
            float: The betweenness centrality value.
        """
        N = len(flight_network)
        betweenness = 0
        for s in flight_network:
            if s == airport: 
                continue
            distances, num_paths = bfs_shortest_paths(s)
            for t in flight_network:
                if t == s or t == airport: 
                    continue
                if distances[t] < float('inf') and distances[airport] < distances[t]:
                    betweenness += num_paths[airport] / num_paths[t]
        normalization_factor = (N - 1) * (N - 2) / 2 if N > 2 else 1
        return betweenness / normalization_factor

    def compute_closeness():
        """
        Calculates the closeness centrality of the given airport.

        Returns:
            float: The closeness centrality value.
        """
        distances, _ = bfs_shortest_paths(airport)
        total_distance = sum(d for d in distances.values() if d < float('inf'))
        N = len(flight_network)
        if total_distance > 0:
            return (N - 1) / total_distance
        return 0

    def compute_degree():
        """
        Calculates the degree centrality of the given airport.

        Returns:
            float: The degree centrality value.
        """
        N = len(flight_network)
        degree = len(flight_network[airport])
        return degree / (N - 1) if N > 1 else 0

    def compute_pagerank_websurfer(flight_network, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Calculates the PageRank of each node in the network using the Web Surfer model with teleportation.

        Args:
            flight_network (dict): The flight network.
            alpha (float, optional): The teleportation probability. Defaults to 0.85.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

        Returns:
            dict: A dictionary mapping nodes to their PageRank values.
        """
        N = len(flight_network)
        pagerank = {node: 1 / N for node in flight_network} 
        teleport = (1 - alpha) / N 

        for _ in range(max_iter):
            new_pagerank = {}
            for node in flight_network:
                rank_sum = 0
                for neighbor in flight_network:
                    if node in flight_network[neighbor] and len(flight_network[neighbor]) > 0:
                        rank_sum += pagerank[neighbor] / len(flight_network[neighbor])
                new_pagerank[node] = teleport + alpha * rank_sum

            # Check for convergence
            if all(abs(new_pagerank[node] - pagerank[node]) < tol for node in pagerank):
                pagerank = new_pagerank
                break
            pagerank = new_pagerank

        return pagerank

    # Calculate the centrality measures
    betweenness = compute_betweenness()
    closeness = compute_closeness()
    degree = compute_degree()
    pagerank = compute_pagerank_websurfer(flight_network)
    pagerank_value = pagerank[airport]

    # Return the results
    return {
        "Betweenness Centrality": betweenness,
        "Closeness Centrality": closeness,
        "Degree Centrality": degree,
        "PageRank": pagerank_value
    }

### 2.2 ###

def compute_pagerank(flight_network, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Calculates the PageRank for all nodes in the network.

    Args:
        flight_network (graph): A graph representing the flight network.
        alpha (float, optional): The teleportation probability. Defaults to 0.85.
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Returns:
        dict: A dictionary mapping nodes to their PageRank values.
    """
    N = len(flight_network)
    pagerank = {node: 1 / N for node in flight_network}  # Initialize uniformly
    out_degree = {node: len(flight_network[node]) for node in flight_network}

    # Normalize to avoid division by zero
    for node, deg in out_degree.items():
        if deg == 0:
            out_degree[node] = 1  # Avoid division by zero for nodes with no outgoing edges

    for _ in range(max_iter):
        new_pagerank = {node: 0 for node in flight_network} 
        for node in flight_network:
            if out_degree[node] > 0:
                share = pagerank[node] / out_degree[node]
                for neighbor in flight_network[node]:
                    new_pagerank[neighbor] += alpha * share
            else:
                for neighbor in flight_network:
                    new_pagerank[neighbor] += alpha * (pagerank[node] / N)

       
        for node in new_pagerank:
            new_pagerank[node] += (1 - alpha) / N

       
        if all(abs(new_pagerank[node] - pagerank[node]) < tol for node in flight_network):
            break

        pagerank = new_pagerank  

    return pagerank

def bfs_shortest_paths(flight_network, source):
    """
    Calculates shortest path distances and number of shortest paths from a source node to all other nodes using BFS.

    Args:
        flight_network (dict): The flight network.
        source (str): The source node.

    Returns:
        tuple: A tuple containing two dictionaries:
            - distances: A dictionary mapping nodes to their shortest path distances from the source.
            - num_paths: A dictionary mapping nodes to the number of shortest paths from the source.
    """
    queue = deque([source])
    distances = {node: float('inf') for node in flight_network}
    distances[source] = 0
    num_paths = {node: 0 for node in flight_network}
    num_paths[source] = 1

    while queue:
        current = queue.popleft()
        for neighbor in flight_network[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
            if distances[neighbor] == distances[current] + 1:
                num_paths[neighbor] += num_paths[current]

    return distances, num_paths

def compute_betweenness(flight_network):
    """
    Calculates the betweenness centrality for all nodes in the network.

    Returns:
        dict: A dictionary mapping nodes to their betweenness centrality values.
    """
    N = len(flight_network)
    betweenness = {node: 0 for node in flight_network}

    for s in flight_network:
        distances, num_paths = bfs_shortest_paths(flight_network, s)
        for t in flight_network:
            if t == s:
                continue
            if distances[t] < float('inf'):
                for node in flight_network:
                    if node != s and node != t and distances[node] < distances[t]:
                        betweenness[node] += num_paths[node] / num_paths[t]

    # Normalize
    normalization_factor = (N - 1) * (N - 2) / 2 if N > 2 else 1
    for node in betweenness:
        betweenness[node] /= normalization_factor

    return betweenness

def compute_closeness(flight_network):
    """
    Calculates the closeness centrality for all nodes in the network.

    Returns:
        dict: A dictionary mapping nodes to their closeness centrality values.
    """
    N = len(flight_network)
    closeness = {}

    for node in flight_network:
        # Calculate distances from the node using BFS
        distances, _ = bfs_shortest_paths(flight_network, node)

        # Sum of distances, excluding infinite distances (unreachable nodes)
        total_distance = sum(d for d in distances.values() if d < float('inf'))

        # If the sum of distances is greater than zero, calculate closeness
        if total_distance > 0:
            closeness[node] = (N - 1) / total_distance
        else:
            closeness[node] = 0  # Isolated or unreachable node, centrality 0

    return closeness

def compute_degree(flight_network):
    """
    Calculates the degree centrality for all nodes in the network.

    Returns:
        dict: A dictionary mapping nodes to their degree centrality values.
    """
    degree = {node: len(flight_network[node]) for node in flight_network}
    N = len(flight_network)
    for node in degree:
        degree[node] = degree[node] / (N - 1) if N > 1 else 0
    return degree

def analyze_centrality_all(flight_network):
    """
    Calculates and returns all centrality measures for all nodes.

    Args:
        flight_network (dict): The flight network.

    Returns:
        dict: A dictionary mapping nodes to their centrality values.
    """
    pagerank = compute_pagerank(flight_network)
    betweenness = compute_betweenness(flight_network)
    closeness = compute_closeness(flight_network)
    degree = compute_degree(flight_network)

    centralities = {}
    for airport in flight_network:
        centralities[airport] = {
            "Betweenness Centrality": betweenness[airport],
            "Closeness Centrality": closeness[airport],
            "Degree Centrality": degree[airport],
            "PageRank": pagerank[airport]
        }

    return centralities

import matplotlib.pyplot as plt
import pandas as pd


def compare_centralities(flight_network):
    """
    Compares different centrality measures for all nodes in the flight network.

    Args:
        flight_network (dict): A dictionary representing the flight network.
    """

    # Calculate all centrality measures for all nodes
    centralities = analyze_centrality_all(flight_network)

    # Dictionaries to store centrality values for each node
    betweenness_values = {}
    closeness_values = {}
    degree_values = {}
    pagerank_values = {}

    # Extract centrality values for each airport
    for airport, centrality in centralities.items():
        betweenness_values[airport] = centrality["Betweenness Centrality"]
        closeness_values[airport] = centrality["Closeness Centrality"]
        degree_values[airport] = centrality["Degree Centrality"]
        pagerank_values[airport] = centrality["PageRank"]

    # Plot distributions (histograms) for each centrality measure

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Centrality Distributions")

    # Betweenness Centrality
    axes[0, 0].hist(betweenness_values.values(), bins=20, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Betweenness Centrality")
    axes[0, 0].set_xlabel("Centrality")
    axes[0, 0].set_ylabel("Frequency")

    # Closeness Centrality
    axes[0, 1].hist(closeness_values.values(), bins=20, color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Closeness Centrality")
    axes[0, 1].set_xlabel("Centrality")
    axes[0, 1].set_ylabel("Frequency")

    # Degree Centrality
    axes[1, 0].hist(degree_values.values(), bins=20, color="salmon", edgecolor="black")
    axes[1, 0].set_title("Degree Centrality")
    axes[1, 0].set_xlabel("Centrality")
    axes[1, 0].set_ylabel("Frequency")

    # PageRank
    axes[1, 1].hist(pagerank_values.values(), bins=20, color="lightcoral", edgecolor="black")
    axes[1, 1].set_title("PageRank")
    axes[1, 1].set_xlabel("Centrality")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Create lists of top 5 airports for each centrality measure
    top_5_betweenness = sorted(betweenness_values.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_closeness = sorted(closeness_values.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_degree = sorted(degree_values.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_pagerank = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)[:5]

    # Create a Pandas DataFrame for a formatted table
    data = {
        "Airport (Betweenness)": [airport for airport, _ in top_5_betweenness],
        "Betweenness Centrality": [score for _, score in top_5_betweenness],
        "Airport (Closeness)": [airport for airport, _ in top_5_closeness],
        "Closeness Centrality": [score for _, score in top_5_closeness],
        "Airport (Degree)": [airport for airport, _ in top_5_degree],
        "Degree Centrality": [score for _, score in top_5_degree],
        "Airport (PageRank)": [airport for airport, _ in top_5_pagerank],
        "PageRank": [score for _, score in top_5_pagerank],
    }

    df = pd.DataFrame(data)
    df = df.round(
        {
            "Betweenness Centrality": 4,
            "Closeness Centrality": 4,
            "Degree Centrality": 4,
              "PageRank": 4})
    # Visualize the centrality table graphically with Matplotlib

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create a Matplotlib table with formatted data
    table = ax.table(
        cellText=df.values,  # Use DataFrame values for table cells
        colLabels=df.columns,  # Use DataFrame column names for table headers
        loc='center',        # Center the table in the figure
        cellLoc='center',     # Center text within table cells
        colColours=['lightblue'] * df.shape[1]  # Set all column headers to light blue
    )

    plt.show()

    ### 2.4 ###

    import networkx as nx
import numpy as np

def compute_hits(flight_network, max_iter=100, tol=1e-6):
    """
    Calculates the HITS scores for all nodes in the graph (Hub and Authority).

    Parameters:
    - flight_network: The directed graph represented as a networkx DiGraph object.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence checking.
    """

    # Initialize Hub and Authority scores
    node_list = list(flight_network.nodes())  # List of nodes
    N = len(node_list)  # Number of nodes
    node_index = {node: idx for idx, node in enumerate(node_list)}  # Map node to index

    # Create adjacency matrix (unweighted)
    A = np.zeros((N, N))  # Adjacency matrix
    for node in flight_network.nodes():
        for neighbor in flight_network.neighbors(node):
            A[node_index[node], node_index[neighbor]] = 1

    # Initialize Hub and Authority scores
    hub_scores = np.ones(N)
    authority_scores = np.ones(N)

    # Iterate until convergence
    for _ in range(max_iter):
        # Calculate new Authority and Hub scores
        new_authority_scores = A.T.dot(hub_scores)  # Authority is the sum of Hub scores of nodes pointing to the node
        new_hub_scores = A.dot(authority_scores)  # Hub is the sum of Authority scores of nodes pointed to by the node

        # Normalize scores
        new_authority_scores /= np.linalg.norm(new_authority_scores, 2)  # L2 normalization for Authority
        new_hub_scores /= np.linalg.norm(new_hub_scores, 2)  # L2 normalization for Hub

        # Check for convergence
        if np.linalg.norm(new_authority_scores - authority_scores, ord=2) < tol and np.linalg.norm(new_hub_scores - hub_scores, ord=2) < tol:
            break

        # Update scores
        authority_scores = new_authority_scores
        hub_scores = new_hub_scores

    # Map Authority and Hub scores back to nodes
    authority_result = {node_list[i]: authority_scores[i] for i in range(N)}
    hub_result = {node_list[i]: hub_scores[i] for i in range(N)}

    return authority_result, hub_result

def top_5_authority_low_degree(flight_network, authority, degree):
    """
    Finds the top 5 nodes with high authority and low degree.

    Args:
        flight_network (dict): The flight network.
        authority (dict): A dictionary mapping nodes to their authority scores.
        degree (dict): A dictionary mapping nodes to their degree centrality scores.

    Returns:
        list: A list of dictionaries, each containing the airport name, authority score, and degree centrality score.
    """
    results = []
    for airport in flight_network:
        if degree[airport] < 0.2:  # Arbitrary threshold for low degree
            results.append({
                "Airport": airport,
                "Authority": round(authority[airport], 4),
                "Degree": round(degree[airport], 4)
            })

    # Sort by authority descending and degree ascending
    sorted_results = sorted(results, key=lambda x: (-x["Authority"], x["Degree"]))[:5]

    return sorted_results

def top_5_hub_low_betweenness(flight_network, hub, betweenness):
    """
    Finds the top 5 nodes with high hub and low betweenness.

    Args:
        flight_network (dict): The flight network.
        hub (dict): A dictionary mapping nodes to their hub scores.
        betweenness (dict): A dictionary mapping nodes to their betweenness centrality scores.

    Returns:
        list: A list of dictionaries, each containing the airport name, hub score, and betweenness centrality score.
    """
    results = []
    for airport in flight_network:
        if betweenness[airport] < 0.2:  # Arbitrary threshold for low betweenness
            results.append({
                "Airport": airport,
                "Hub": round(hub[airport], 4),
                "Betweenness": round(betweenness[airport], 4)
            })

    # Sort by hub descending and betweenness ascending
    sorted_results = sorted(results, key=lambda x: (-x["Hub"], x["Betweenness"]))[:5]

    return sorted_results

### 3 ###
import heapq

def dijkstra(graph, start, target, date):
    """
    Dijkstra's algorithm to find the shortest path in terms of distance,
    considering only edges that match the specified date.

    Args:
        graph: A NetworkX MultiDiGraph representing the flight network.
        start: The starting node (airport).
        target: The target node (airport).
        date: The date for which to find the shortest path.

    Returns:
        list: A list of nodes representing the shortest path from start to target, or None if no path exists.
    """

    queue = [(0, start)]  # Priority queue of (distance, node) tuples
    distances = {start: 0}  # Dictionary to store shortest distances
    previous_nodes = {start: None}  # Dictionary to store the previous node in the shortest path

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == target:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            return path[::-1]

        # Iterate over neighbors and their edges, filtering by date
        for neighbor in graph.neighbors(current_node):
            for edge_data in graph.get_edge_data(current_node, neighbor).values():
                if edge_data['date'] == date:
                    distance = edge_data['distance']
                    new_distance = current_distance + distance

                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(queue, (new_distance, neighbor))

    return None

def find_best_routes(graph, origin_airports, destination_airports, date):
    """
    Given the flight network, origin airports, destination airports, and a date,
    finds the best possible flight route between each origin-destination pair.

    Args:
        graph: A NetworkX MultiDiGraph representing the flight network.
        origin_airports: A list of origin airports.
        destination_airports: A list of destination airports.
        date: The date for which to find the best routes.

    Returns:
        pandas.DataFrame: A DataFrame containing the origin airport, destination airport, and the best route between them.
    """

    results = []  # List to store results

    # For each origin-destination pair, find the best route using Dijkstra's algorithm
    for origin_airport in origin_airports:
        for destination_airport in destination_airports:
            best_route = dijkstra(graph, origin_airport, destination_airport, date)
            route_str = " → ".join(best_route) if best_route else "No route found."
            results.append({
                'Origin_city_airport': origin_airport,
                'Destination_city_airport': destination_airport,
                'Best_route': route_str
            })

    # Convert the results into a DataFrame for better readability
    result_df = pd.DataFrame(results)
    return result_df

def find_origin_cities_by_name(city_name, df):
    """
    Finds origin cities matching the given city name.

    Args:
        city_name (str): The city name to search for.
        df (pd.DataFrame): The DataFrame containing flight data.

    Returns:
        list: A list of unique Origin_airport values matching the city name.
    """

    matching_cities = df[df['Origin_city'].str.contains(city_name, case=False)]
    # Filter further to exclude matches with additional words or codes
    matching_cities = matching_cities[matching_cities['Origin_city'].str.split(',').str[0] == city_name]
    exact_matches = matching_cities['Origin_airport'].unique()

    return exact_matches

def find_destination_cities_by_name(city_name, df):
    """
    Finds origin cities matching the given city name.

    Args:
        city_name (str): The city name to search for.
        df (pd.DataFrame): The DataFrame containing flight data.

    Returns:
        list: A list of unique Destination_airport values matching the city name.
    """
    matching_cities = df[df['Destination_city'].str.contains(city_name, case=False)]
    # Filter further to exclude matches with additional words or codes
    matching_cities = matching_cities[matching_cities['Destination_city'].str.split(',').str[0] == city_name]
    exact_matches = matching_cities['Destination_airport'].unique()

    return exact_matches

### maps ###

def create_usa_airports_map(df):
    """
    Creates a map of the United States with all airports from the dataset.

    Parameters:
    df (pandas.DataFrame): DataFrame containing airport coordinates.

    Returns:
    folium.Map: Interactive map of airports.
    """

    # Extract origin airports
    airports_origin = df[['Origin_airport', 'Org_airport_lat', 'Org_airport_long']].drop_duplicates(subset='Origin_airport')

    # Extract destination airports
    airports_dest = df[['Destination_airport', 'Dest_airport_lat', 'Dest_airport_long']].drop_duplicates(subset='Destination_airport')
    airports_dest.columns = ['Origin_airport', 'Org_airport_lat', 'Org_airport_long']  # Rename columns for consistency

    # Combine origin and destination airports
    airports = pd.concat([airports_origin, airports_dest]).drop_duplicates(subset='Origin_airport')

    # Remove rows with missing (NaN) coordinates
    airports = airports.dropna(subset=['Org_airport_lat', 'Org_airport_long'])

    # Create the map centered on the United States
    usa_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # Add markers for each airport
    for idx, row in airports.iterrows():
        try:
            # Convert coordinates to float
            lat = float(row['Org_airport_lat'])
            lon = float(row['Org_airport_long'])

            # Add marker with information and styling
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=row['Origin_airport'],
                color='blue',
                fill=True,
                fillColor='blue'
            ).add_to(usa_map)
        except (ValueError, TypeError):
            # Skip airports with invalid coordinates
            print(f"Invalid coordinates for airport: {row['Origin_airport']}")

    return usa_map

def add_routes_to_map(usa_map, routes_df, airports_df):
    """
    Adds the best routes to the existing map with different colors for each route.

    Parameters:
    usa_map (folium.Map): Map to add routes to.
    routes_df (pandas.DataFrame): DataFrame containing the best routes.
    airports_df (pandas.DataFrame): DataFrame containing airport coordinates.

    Returns:
    folium.Map: Updated map with routes.
    """

    # Create a dictionary to efficiently retrieve airport coordinates
    airport_coords = pd.concat([
        airports_df[['Origin_airport', 'Org_airport_lat', 'Org_airport_long']],
        airports_df[['Destination_airport', 'Dest_airport_lat', 'Dest_airport_long']]
            .rename(columns={'Destination_airport': 'Origin_airport',
                             'Dest_airport_lat': 'Org_airport_lat',
                             'Dest_airport_long': 'Org_airport_long'})
    ]).drop_duplicates(subset='Origin_airport').set_index('Origin_airport')

    # Function to generate random colors
    def random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Hex color format

    for _, row in routes_df.iterrows():
        route = row['Best_route']
        if route == "No route found.":
            continue

        # Extract airport codes from the route string
        airports = route.split(" → ")

        # Get coordinates for each airport in the route
        route_coords = []
        for airport in airports:
            if airport in airport_coords.index:
                lat = float(airport_coords.loc[airport, 'Org_airport_lat'])
                lon = float(airport_coords.loc[airport, 'Org_airport_long'])
                route_coords.append((lat, lon))

        # Add route as a line to the map with a random color
        if len(route_coords) > 1:
            PolyLine(
                locations=route_coords,
                color=random_color(),  
                weight=3,
                opacity=0.8
            ).add_to(usa_map)
    
    return usa_map