# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df
# Create trip_id to route_id mapping
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Map route_id to a list of stops in order of their sequence
    for _, row in df_stop_times.iterrows():
        route_id = trip_to_route[row['trip_id']]
        stop_id = row['stop_id']
        if stop_id not in route_to_stops[route_id]:
            route_to_stops[route_id].append(stop_id)
        stop_trip_count[stop_id] += 1

    # Create fare rules for routes
    for _, row in df_fare_rules.iterrows():
        fare_rules[row['route_id']] = row['fare_id']

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_attributes, df_fare_rules, on='fare_id', how='left')

    # Create trip_id to route_id mapping

    # Map route_id to a list of stops in order of their sequence

    # Ensure each route only has unique stops
    
    # Count trips per stop

    # Create fare rules for routes

    # Merge fare rules and attributes into a single DataFrame

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_count = defaultdict(int)
    for route_id in trip_to_route.values():
        route_trip_count[route_id] += 1

    # Sort routes by trip count in descending order and return the top 5
    top_5_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_5_routes

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    # Sort stops by trip count in descending order and return the top 5
    top_5_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_5_stops


# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # Count the unique routes passing through each stop
    stop_route_count = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_route_count[stop_id].add(route_id)

    # Convert to (stop_id, route_count) and sort by route_count in descending order
    top_5_busiest_stops = sorted([(stop, len(routes)) for stop, routes in stop_route_count.items()], 
                                 key=lambda x: x[1], reverse=True)[:5]
    return top_5_busiest_stops

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    # Find pairs of stops with only one route
    stop_pairs = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            pair = (stops[i], stops[i+1])
            stop_pairs[pair].add(route_id)

    # Filter pairs with only one route and calculate trip frequency for each pair
    single_route_pairs = []
    for pair, routes in stop_pairs.items():
        if len(routes) == 1:
            route_id = list(routes)[0]
            trip_count_pair = stop_trip_count[pair[0]] + stop_trip_count[pair[1]]
            single_route_pairs.append((pair, route_id, trip_count_pair))
    
    # Sort by combined trip count and get the top 5
    top_5_single_route_pairs = sorted(single_route_pairs, key=lambda x: x[2], reverse=True)[:5]
    return [(pair, route_id) for pair, route_id, _ in top_5_single_route_pairs]

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Extract stop coordinates
    stop_coords = {}
    for _, row in df_stops.iterrows():
        stop_coords[row['stop_id']] = (row['stop_lat'], row['stop_lon'])
    
    # Create a scatter plot for all stops
    stop_x = [coord[1] for coord in stop_coords.values()]  # Longitude for x-axis
    stop_y = [coord[0] for coord in stop_coords.values()]  # Latitude for y-axis

    # Create edge traces for each route
    edge_traces = []
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            start_stop = stops[i]
            end_stop = stops[i + 1]
            start_coord = stop_coords[start_stop]
            end_coord = stop_coords[end_stop]
            
            # Create a line trace between each consecutive stop in the route
            edge_trace = go.Scattergeo(
                lon=[start_coord[1], end_coord[1]],
                lat=[start_coord[0], end_coord[0]],
                mode='lines',
                line=dict(width=1, color='#888'),
                opacity=0.5,
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
    
    # Create a node trace for all stops
    node_trace = go.Scattergeo(
        lon=stop_x,
        lat=stop_y,
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            line_width=1
        ),
        text=[f"Stop ID: {stop_id}" for stop_id in stop_coords.keys()],
        hoverinfo='text'
    )
    
    # Add all edge traces and node trace to the figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Transit Network Graph",
        showlegend=False,
        geo=dict(
            scope='world',
            projection_type='equirectangular',
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)"
        )
    )
    fig.show()
# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    routes = []
    route_items = iter(route_to_stops.items())
    while True:
        try:
            route_id, stops = next(route_items)
            if start_stop in stops and end_stop in stops:
                start_idx = next(i for i, stop in enumerate(stops) if stop == start_stop)
                end_idx = next(i for i, stop in enumerate(stops) if stop == end_stop)
                if start_idx < end_idx:
                    routes.append(route_id)
        except StopIteration:
            break
    return routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop in stops:
            # Add facts to the knowledge base
            +RouteHasStop(route_id, stop)  

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    # Run the query and retrieve data
    DirectRoute(X, Y,R) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)) 

    result = DirectRoute(start, end, R)
    # Extract route IDs from results using a while loop
    routes = set()
    i = 0
    while i < len(result):
        routes.add(result[i][0])
        i += 1
    return sorted(routes)

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
     # Define the optimal route with a maximum of one interchange and an intermediate stop
    OptimalRoute(R1, R2, stop_id_to_include) <= (
        DirectRoute(start_stop_id, stop_id_to_include, R1) &  # Route from start to via stop
        DirectRoute(stop_id_to_include, end_stop_id, R2) &   # Route from via stop to end
        (R1 != R2)                                           # Ensure a route interchange
    )

    # Run the query to get optimal routes based on the criteria
    if max_transfers < 1:
        return []  # No valid routes if no transfer is allowed

    # Extract results from OptimalRoute query
    result = OptimalRoute(R1, R2, stop_id_to_include).data

    # Return the result as a list of tuples with (route_id1, stop_id, route_id2)
    return [(route[0], stop_id_to_include, route[1]) for route in result]

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Define OptimalRoute to match the required route with one interchange via stop_id_to_include
    OptimalRoute(R1, R2, stop_id_to_include) <= (
        DirectRoute(start_stop_id, stop_id_to_include, R1) &
        DirectRoute(stop_id_to_include, end_stop_id, R2) &
        (R1 != R2)
    )

    # Run the query if at least one transfer is allowed
    if max_transfers < 1:
        return []  # No valid routes if no transfer is allowed

    # Retrieve the query results for OptimalRoute
    result = OptimalRoute(R1, R2, stop_id_to_include).data

    # Convert results into a list of tuples, ensuring the order matches expected output
    optimal_paths = [(res[1], stop_id_to_include, res[0]) for res in result if len(res) == 2]

    # Sort the results to ensure consistent output order
    return sorted(optimal_paths)

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    pass  # Implementation here

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here


#visualize_stop_route_graph_interactive(route_to_stops)