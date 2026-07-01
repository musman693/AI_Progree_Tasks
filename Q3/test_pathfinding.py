"""
Comprehensive testing and demonstration of the Heuristic Pathfinding Engine
Tests A*, Dijkstra, and BFS across varying grid sizes and obstacle configurations
"""

import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, r'c:\Users\ASA\Desktop\Progree Tasks')

from pathfinding_engine import (
    GridEnvironment, PathfindingAgent, PathfindingVisualizer, PathfindingMetrics
)


def create_test_grids():
    """Create various test grids with different sizes and obstacle configurations"""
    grids = {}
    
    # Simple 10x10 grid with few obstacles
    grid1 = np.zeros((10, 10), dtype=int)
    grid1[3:7, 3:7] = 1  # Central obstacle
    grids['10x10_simple'] = grid1
    
    # Medium 20x20 grid with scattered obstacles
    grid2 = np.zeros((20, 20), dtype=int)
    np.random.seed(42)
    obstacle_positions = np.random.choice(400, 40, replace=False)
    for pos in obstacle_positions:
        x, y = divmod(pos, 20)
        if (x, y) != (0, 0) and (x, y) != (19, 19):  # Don't block start/goal
            grid2[x, y] = 1
    grids['20x20_scattered'] = grid2
    
    # Maze-like 25x25 grid
    grid3 = np.zeros((25, 25), dtype=int)
    # Create maze walls
    for i in range(25):
        if i % 2 == 0:
            grid3[i, 5:20] = 1
    for j in range(25):
        if j % 3 == 0:
            grid3[8:17, j] = 1
    grids['25x25_maze'] = grid3
    
    # Dense 15x15 grid (30% obstacles)
    grid4 = np.zeros((15, 15), dtype=int)
    np.random.seed(123)
    num_obstacles = int(15 * 15 * 0.3)
    obstacle_positions = np.random.choice(225, num_obstacles, replace=False)
    for pos in obstacle_positions:
        x, y = divmod(pos, 15)
        if (x, y) != (0, 0) and (x, y) != (14, 14):
            grid4[x, y] = 1
    grids['15x15_dense'] = grid4
    
    return grids


def run_pathfinding_tests():
    """Run comprehensive pathfinding tests"""
    print("="*80)
    print("HEURISTIC GRAPH PATHFINDING ENGINE - COMPREHENSIVE TESTING")
    print("="*80)
    print()
    
    # Create test grids
    test_grids = create_test_grids()
    all_metrics = []
    
    # Test each grid
    for grid_name, grid in test_grids.items():
        print(f"\n{'='*80}")
        print(f"Testing Grid: {grid_name}")
        print(f"{'='*80}")
        print(f"Grid Size: {grid.shape}")
        print(f"Obstacles: {np.sum(grid)} / {grid.size}")
        print(f"Obstacle Density: {np.sum(grid)/grid.size*100:.1f}%")
        
        # Create environment and agent
        env = GridEnvironment(grid)
        agent = PathfindingAgent(env)
        
        # Define start and goal
        start = (0, 0)
        goal = (grid.shape[0]-1, grid.shape[1]-1)
        
        # Check if start and goal are valid
        if not env.is_valid(start) or not env.is_valid(goal):
            print("ERROR: Start or goal position blocked!")
            continue
        
        print(f"Start: {start}, Goal: {goal}")
        print()
        
        # Test A* with different heuristics
        print("Testing A* with Manhattan heuristic...")
        path_a_manhattan, metrics_a_manhattan = agent.astar(start, goal, heuristic='manhattan')
        print(f"  [OK] Path found: {metrics_a_manhattan.path_found}")
        print(f"  [OK] Path length: {metrics_a_manhattan.path_length}")
        print(f"  [OK] Nodes expanded: {metrics_a_manhattan.nodes_expanded}")
        print(f"  [OK] Runtime: {metrics_a_manhattan.runtime_ms:.3f} ms")
        all_metrics.append(metrics_a_manhattan)
        
        print("\nTesting A* with Euclidean heuristic...")
        path_a_euclidean, metrics_a_euclidean = agent.astar(start, goal, heuristic='euclidean')
        print(f"  [OK] Path found: {metrics_a_euclidean.path_found}")
        print(f"  [OK] Path length: {metrics_a_euclidean.path_length}")
        print(f"  [OK] Nodes expanded: {metrics_a_euclidean.nodes_expanded}")
        print(f"  [OK] Runtime: {metrics_a_euclidean.runtime_ms:.3f} ms")
        all_metrics.append(metrics_a_euclidean)
        
        print("\nTesting A* with Chebyshev heuristic...")
        path_a_chebyshev, metrics_a_chebyshev = agent.astar(start, goal, heuristic='chebyshev')
        print(f"  [OK] Path found: {metrics_a_chebyshev.path_found}")
        print(f"  [OK] Path length: {metrics_a_chebyshev.path_length}")
        print(f"  [OK] Nodes expanded: {metrics_a_chebyshev.nodes_expanded}")
        print(f"  [OK] Runtime: {metrics_a_chebyshev.runtime_ms:.3f} ms")
        all_metrics.append(metrics_a_chebyshev)
        
        # Test Dijkstra
        print("\nTesting Dijkstra's algorithm...")
        path_dijkstra, metrics_dijkstra = agent.dijkstra(start, goal)
        print(f"  [OK] Path found: {metrics_dijkstra.path_found}")
        print(f"  [OK] Path length: {metrics_dijkstra.path_length}")
        print(f"  [OK] Nodes expanded: {metrics_dijkstra.nodes_expanded}")
        print(f"  [OK] Runtime: {metrics_dijkstra.runtime_ms:.3f} ms")
        all_metrics.append(metrics_dijkstra)
        
        # Test BFS
        print("\nTesting Breadth-First Search...")
        path_bfs, metrics_bfs = agent.bfs(start, goal)
        print(f"  [OK] Path found: {metrics_bfs.path_found}")
        print(f"  [OK] Path length: {metrics_bfs.path_length}")
        print(f"  [OK] Nodes expanded: {metrics_bfs.nodes_expanded}")
        print(f"  [OK] Runtime: {metrics_bfs.runtime_ms:.3f} ms")
        all_metrics.append(metrics_bfs)
        
        # Visualization for successful paths
        if metrics_a_manhattan.path_found:
            print("\nVisualization (A* Manhattan):")
            print("  Legend: S=Start, G=Goal, #=Obstacle, *=Path, .=Open")
            viz = PathfindingVisualizer.create_ascii_visualization(
                grid, path_a_manhattan, start, goal
            )
            for line in viz.split('\n')[:15]:  # Show first 15 lines
                print(f"  {line}")
            if len(viz.split('\n')) > 15:
                print(f"  ... ({len(viz.split(chr(10)))-15} more lines)")
    
    # Export and display summary
    print("\n" + "="*80)
    print("EXPORTING METRICS")
    print("="*80)
    
    metrics_file = PathfindingVisualizer.export_metrics_log(
        all_metrics, 
        filename='pathfinding_metrics.json'
    )
    print(f"[OK] Metrics exported to: {metrics_file}")
    
    # Generate and display summary report
    print("\n" + "="*80)
    summary_report = PathfindingVisualizer.generate_summary_report(all_metrics)
    print(summary_report)
    
    # Detailed test comparison
    print("\n" + "="*80)
    print("DETAILED ALGORITHM COMPARISON")
    print("="*80)
    print()
    print(f"{'Algorithm':<25} {'Grid':<20} {'Found':<8} {'Path Len':<12} {'Nodes':<12} {'Time (ms)':<12}")
    print("-" * 89)
    
    for m in all_metrics:
        grid_info = f"{m.grid_size[0]}x{m.grid_size[1]}"
        found_str = "Yes" if m.path_found else "No"
        print(f"{m.algorithm:<25} {grid_info:<20} {found_str:<8} {m.path_length:<12} {m.nodes_expanded:<12} {m.runtime_ms:<12.3f}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nTotal tests run: {len(all_metrics)}")
    print(f"Successful paths: {sum(1 for m in all_metrics if m.path_found)}/{len(all_metrics)}")
    print(f"Average nodes expanded: {np.mean([m.nodes_expanded for m in all_metrics]):.1f}")
    print(f"Average runtime: {np.mean([m.runtime_ms for m in all_metrics]):.3f} ms")
    
    return all_metrics


if __name__ == "__main__":
    os.chdir(r'c:\Users\ASA\Desktop\Progree Tasks')
    metrics = run_pathfinding_tests()
    print("\n[OK] All tests completed successfully!")
