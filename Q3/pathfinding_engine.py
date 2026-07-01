"""
Heuristic Graph Pathfinding Agent Search Engine
Implements A* Search, Dijkstra's Algorithm, and pathfinding agent framework
with metrics tracking and visualization
"""

import heapq
import time
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PathfindingMetrics:
    """Track pathfinding metrics"""
    algorithm: str
    grid_size: Tuple[int, int]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path_length: int
    steps_taken: int
    nodes_expanded: int
    runtime_ms: float
    path_found: bool
    path: List[Tuple[int, int]]
    
    def to_dict(self):
        return asdict(self)


class GridEnvironment:
    """Represents the grid maze environment"""
    
    def __init__(self, grid: np.ndarray):
        """
        Initialize grid environment
        Args:
            grid: 2D numpy array where 0=open, 1=obstacle
        """
        self.grid = grid
        self.height, self.width = grid.shape
        self.obstacle_count = np.sum(grid)
        
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid and not an obstacle"""
        x, y = pos
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] == 0)
    
    def get_neighbors(self, pos: Tuple[int, int], diagonal: bool = False) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (4-way or 8-way)"""
        x, y = pos
        neighbors = []
        
        # 4-way movement (up, down, left, right)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # Add diagonals if enabled
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid((nx, ny)):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def heuristic_manhattan(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def heuristic_euclidean(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def heuristic_chebyshev(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Chebyshev distance heuristic"""
        return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))


class PathfindingAgent:
    """Agent capable of pathfinding using various algorithms"""
    
    def __init__(self, environment: GridEnvironment):
        self.env = environment
        self.metrics_history = []
    
    def astar(self, start: Tuple[int, int], goal: Tuple[int, int], 
              heuristic: str = 'manhattan', diagonal: bool = False) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """
        A* pathfinding algorithm
        Args:
            start: Starting position
            goal: Goal position
            heuristic: 'manhattan', 'euclidean', or 'chebyshev'
            diagonal: Whether to allow diagonal movement
        """
        start_time = time.time()
        nodes_expanded = 0
        
        # Select heuristic function
        if heuristic == 'manhattan':
            h_func = self.env.heuristic_manhattan
        elif heuristic == 'euclidean':
            h_func = self.env.heuristic_euclidean
        else:
            h_func = self.env.heuristic_chebyshev
        
        # Priority queue: (f_score, counter, position)
        open_set = [(0, 0, start)]
        counter = 0
        came_from = {}
        g_score = {start: 0}
        f_score = {start: h_func(start, goal)}
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            nodes_expanded += 1
            closed_set.add(current)
            
            # Goal reached
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                runtime = (time.time() - start_time) * 1000
                
                metrics = PathfindingMetrics(
                    algorithm=f"A* ({heuristic})",
                    grid_size=(self.env.height, self.env.width),
                    start=start,
                    goal=goal,
                    path_length=len(path),
                    steps_taken=len(path) - 1,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=runtime,
                    path_found=True,
                    path=path
                )
                self.metrics_history.append(metrics)
                return path, metrics
            
            # Explore neighbors
            for neighbor in self.env.get_neighbors(current, diagonal):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + h_func(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        runtime = (time.time() - start_time) * 1000
        metrics = PathfindingMetrics(
            algorithm=f"A* ({heuristic})",
            grid_size=(self.env.height, self.env.width),
            start=start,
            goal=goal,
            path_length=0,
            steps_taken=0,
            nodes_expanded=nodes_expanded,
            runtime_ms=runtime,
            path_found=False,
            path=[]
        )
        self.metrics_history.append(metrics)
        return [], metrics
    
    def dijkstra(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 diagonal: bool = False) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """
        Dijkstra's pathfinding algorithm
        Args:
            start: Starting position
            goal: Goal position
            diagonal: Whether to allow diagonal movement
        """
        start_time = time.time()
        nodes_expanded = 0
        
        # Priority queue: (distance, counter, position)
        open_set = [(0, 0, start)]
        counter = 0
        came_from = {}
        distances = {start: 0}
        visited = set()
        
        while open_set:
            current_dist, _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            nodes_expanded += 1
            visited.add(current)
            
            # Goal reached
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                runtime = (time.time() - start_time) * 1000
                
                metrics = PathfindingMetrics(
                    algorithm="Dijkstra",
                    grid_size=(self.env.height, self.env.width),
                    start=start,
                    goal=goal,
                    path_length=len(path),
                    steps_taken=len(path) - 1,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=runtime,
                    path_found=True,
                    path=path
                )
                self.metrics_history.append(metrics)
                return path, metrics
            
            # Explore neighbors
            for neighbor in self.env.get_neighbors(current, diagonal):
                if neighbor in visited:
                    continue
                
                tentative_dist = distances[current] + 1
                
                if neighbor not in distances or tentative_dist < distances[neighbor]:
                    distances[neighbor] = tentative_dist
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (tentative_dist, counter, neighbor))
        
        # No path found
        runtime = (time.time() - start_time) * 1000
        metrics = PathfindingMetrics(
            algorithm="Dijkstra",
            grid_size=(self.env.height, self.env.width),
            start=start,
            goal=goal,
            path_length=0,
            steps_taken=0,
            nodes_expanded=nodes_expanded,
            runtime_ms=runtime,
            path_found=False,
            path=[]
        )
        self.metrics_history.append(metrics)
        return [], metrics
    
    def bfs(self, start: Tuple[int, int], goal: Tuple[int, int], 
            diagonal: bool = False) -> Tuple[List[Tuple[int, int]], PathfindingMetrics]:
        """
        Breadth-First Search pathfinding
        Args:
            start: Starting position
            goal: Goal position
            diagonal: Whether to allow diagonal movement
        """
        start_time = time.time()
        nodes_expanded = 0
        
        queue = deque([start])
        visited = {start}
        came_from = {}
        
        while queue:
            current = queue.popleft()
            nodes_expanded += 1
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                runtime = (time.time() - start_time) * 1000
                
                metrics = PathfindingMetrics(
                    algorithm="BFS",
                    grid_size=(self.env.height, self.env.width),
                    start=start,
                    goal=goal,
                    path_length=len(path),
                    steps_taken=len(path) - 1,
                    nodes_expanded=nodes_expanded,
                    runtime_ms=runtime,
                    path_found=True,
                    path=path
                )
                self.metrics_history.append(metrics)
                return path, metrics
            
            for neighbor in self.env.get_neighbors(current, diagonal):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        # No path found
        runtime = (time.time() - start_time) * 1000
        metrics = PathfindingMetrics(
            algorithm="BFS",
            grid_size=(self.env.height, self.env.width),
            start=start,
            goal=goal,
            path_length=0,
            steps_taken=0,
            nodes_expanded=nodes_expanded,
            runtime_ms=runtime,
            path_found=False,
            path=[]
        )
        self.metrics_history.append(metrics)
        return [], metrics
    
    @staticmethod
    def _reconstruct_path(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


class PathfindingVisualizer:
    """Generate visualization logs for pathfinding results"""
    
    @staticmethod
    def create_ascii_visualization(grid: np.ndarray, path: List[Tuple[int, int]], 
                                    start: Tuple[int, int], goal: Tuple[int, int]) -> str:
        """Create ASCII visualization of grid and path"""
        viz_grid = grid.copy().astype(str)
        viz_grid[viz_grid == '0'] = '.'
        viz_grid[viz_grid == '1'] = '#'
        
        # Mark path
        for x, y in path[1:-1]:  # Exclude start and goal
            viz_grid[x, y] = '*'
        
        # Mark start and goal
        if path:
            viz_grid[start[0], start[1]] = 'S'
            viz_grid[goal[0], goal[1]] = 'G'
        
        return '\n'.join(''.join(row) for row in viz_grid)
    
    @staticmethod
    def export_metrics_log(metrics_list: List[PathfindingMetrics], 
                          filename: str = 'pathfinding_metrics.json') -> str:
        """Export metrics to JSON file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(metrics_list),
            'metrics': [m.to_dict() for m in metrics_list]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    @staticmethod
    def generate_summary_report(metrics_list: List[PathfindingMetrics]) -> str:
        """Generate summary report of pathfinding experiments"""
        report = []
        report.append("="*80)
        report.append("PATHFINDING METRICS SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        # Group by algorithm
        by_algorithm = defaultdict(list)
        for m in metrics_list:
            by_algorithm[m.algorithm].append(m)
        
        for algo, metrics in sorted(by_algorithm.items()):
            report.append(f"\nAlgorithm: {algo}")
            report.append("-" * 80)
            
            success_count = sum(1 for m in metrics if m.path_found)
            total_count = len(metrics)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            report.append(f"  Total Runs: {total_count}")
            report.append(f"  Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
            
            if success_count > 0:
                successful_metrics = [m for m in metrics if m.path_found]
                avg_nodes = np.mean([m.nodes_expanded for m in successful_metrics])
                avg_runtime = np.mean([m.runtime_ms for m in successful_metrics])
                avg_path_length = np.mean([m.path_length for m in successful_metrics])
                
                report.append(f"  Average Nodes Expanded: {avg_nodes:.1f}")
                report.append(f"  Average Runtime: {avg_runtime:.3f} ms")
                report.append(f"  Average Path Length: {avg_path_length:.1f}")
                report.append(f"  Min/Max Runtime: {min(m.runtime_ms for m in successful_metrics):.3f}/{max(m.runtime_ms for m in successful_metrics):.3f} ms")
        
        report.append("\n" + "="*80)
        return "\n".join(report)
