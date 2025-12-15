import numpy as np
import pickle
from collections import deque
from queue import PriorityQueue
import math
import time
import psutil
import os
from memory_profiler import memory_usage

def dist(node1, node2, node_attributes):
    x1, y1 = node_attributes[node1]['x'], node_attributes[node1]['y']
    x2, y2 = node_attributes[node2]['x'], node_attributes[node2]['y']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_ids_path(adj_matrix, start_node, goal_node):
    def dfs_limited(node, goal, depth_limit, path):
        if node == goal:
            return path
        if depth_limit == 0:
            return None
        for neighbor, cost in enumerate(adj_matrix[node]):
            if cost > 0:
                result = dfs_limited(neighbor, goal, depth_limit - 1, path + [neighbor])
                if result is not None:
                    return result
        return None

    max_depth = len(adj_matrix)
    for depth in range(max_depth):
        result = dfs_limited(start_node, goal_node, depth, [start_node])
        if result is not None:
            return result
    return None


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    def bfs(queue, visited, parent):
        while queue:
            node = queue.popleft()
            if node in other_visited:
                return node, visited, parent
            for neighbor, cost in enumerate(adj_matrix[node]):
                if cost > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None, visited, parent

    forward_queue = deque([start_node])
    backward_queue = deque([goal_node])
    forward_visited = set([start_node])
    backward_visited = set([goal_node])
    forward_parent = {start_node: None}
    backward_parent = {goal_node: None}

    while forward_queue and backward_queue:
        # Forward search
        other_visited = backward_visited
        meet_node, forward_visited, forward_parent = bfs(forward_queue, forward_visited, forward_parent)
        if meet_node:
            break

        # Backward search
        other_visited = forward_visited
        meet_node, backward_visited, backward_parent = bfs(backward_queue, backward_visited, backward_parent)
        if meet_node:
            break
    
    if not meet_node:
        return None

    # Reconstruct the path
    forward_path = []
    backward_path = []
    
    # Forward path
    node = meet_node
    while node is not None:
        forward_path.append(node)
        node = forward_parent[node]
    forward_path = forward_path[::-1]
    
    # Backward path
    node = backward_parent[meet_node]
    while node is not None:
        backward_path.append(node)
        node = backward_parent[node]
    
    return forward_path + backward_path


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node1, node2):
        x1, y1 = node_attributes[node1]['x'], node_attributes[node1]['y']
        x2, y2 = node_attributes[node2]['x'], node_attributes[node2]['y']
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    open_set = PriorityQueue()
    open_set.put((0, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    open_set_hash = {start_node}

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == goal_node:
            return reconstruct_path(came_from, current)

        for neighbor, cost in enumerate(adj_matrix[current]):
            if cost > 0:  # If there's a connection
                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node)
                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

    return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(node1, node2):
        x1, y1 = node_attributes[node1]['x'], node_attributes[node1]['y']
        x2, y2 = node_attributes[node2]['x'], node_attributes[node2]['y']
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def reconstruct_path(came_from_start, came_from_goal, middle):
        path = []
        current = middle
        while current is not None:
            path.append(current)
            current = came_from_start.get(current)
        path = path[::-1]
        current = came_from_goal.get(middle)
        while current is not None:
            path.append(current)
            current = came_from_goal.get(current)
        return path

    def bidirectional_search():
        open_start = PriorityQueue()
        open_goal = PriorityQueue()
        open_start.put((0, start_node))
        open_goal.put((0, goal_node))
        came_from_start = {}
        came_from_goal = {}
        g_score_start = {start_node: 0}
        g_score_goal = {goal_node: 0}
        f_score_start = {start_node: heuristic(start_node, goal_node)}
        f_score_goal = {goal_node: heuristic(goal_node, start_node)}
        open_set_hash_start = {start_node}
        open_set_hash_goal = {goal_node}
        closed_set_start = set()
        closed_set_goal = set()

        while not open_start.empty() and not open_goal.empty():
            current_start = open_start.get()[1]
            current_goal = open_goal.get()[1]

            if current_start in closed_set_goal:
                return reconstruct_path(came_from_start, came_from_goal, current_start)
            if current_goal in closed_set_start:
                return reconstruct_path(came_from_start, came_from_goal, current_goal)

            closed_set_start.add(current_start)
            closed_set_goal.add(current_goal)

            for direction, current, open_set, open_set_hash, closed_set, g_score, f_score, came_from, target in [
                ('start', current_start, open_start, open_set_hash_start, closed_set_start, g_score_start, f_score_start, came_from_start, goal_node),
                ('goal', current_goal, open_goal, open_set_hash_goal, closed_set_goal, g_score_goal, f_score_goal, came_from_goal, start_node)
            ]:
                for neighbor, cost in enumerate(adj_matrix[current]):
                    if cost > 0 and neighbor not in closed_set:
                        tentative_g_score = g_score[current] + cost
                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)
                            if neighbor not in open_set_hash:
                                open_set.put((f_score[neighbor], neighbor))
                                open_set_hash.add(neighbor)

        return None  # No path found

    return bidirectional_search()


def measure_performance(func, *args):
    # Measure execution time
    start_time = time.time()
    
    # Measure memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # Initial memory usage in MB
    
    # Run the function and measure peak memory usage
    result, peak_memory = memory_usage((func, args), max_usage=True, retval=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate the peak memory usage
    peak_memory_usage = peak_memory - initial_memory
    
    return result, execution_time, peak_memory_usage

def run_test_cases(algorithms, test_cases, adj_matrix, node_attributes):
    results = {}
    
    for algo_name, algo_func in algorithms.items():
        results[algo_name] = []
        
        for i, (start_node, goal_node) in enumerate(test_cases):
            print(f"Running {algo_name} for Test Case {i+1}: Start = {start_node}, Goal = {goal_node}")
            
            path, execution_time, peak_memory = measure_performance(
                algo_func, adj_matrix, node_attributes, start_node, goal_node
            )
            
            results[algo_name].append({
                'test_case': i+1,
                'start': start_node,
                'goal': goal_node,
                'path': path,
                'execution_time': execution_time,
                'peak_memory': peak_memory
            })
            
            print(f"  Path: {path}")
            print(f"  Execution Time: {execution_time:.6f} seconds")
            print(f"  Peak Memory Usage: {peak_memory:.2f} MB")
            print()
    
    return results
# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

import matplotlib.pyplot as plt

def visualize_performance(results):
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'purple']
    markers = ['o', 's', '^', 'D']

    for (algo_name, algo_results), color, marker in zip(results.items(), colors, markers):
        execution_times = [result['execution_time'] for result in algo_results]
        peak_memories = [result['peak_memory'] for result in algo_results]
        
        plt.scatter(execution_times, peak_memories, c=color, marker=marker, s=100, label=algo_name)

    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Algorithm Performance: Time vs Memory Usage')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for each point
    for algo_name, algo_results in results.items():
        for i, result in enumerate(algo_results):
            plt.annotate(f"Test {i+1}", 
                         (result['execution_time'], result['peak_memory']),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center')

    plt.tight_layout()
    plt.savefig('algorithm_performance.png')
    plt.close()

def bonus_problem(adj_matrix):

  return []


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')