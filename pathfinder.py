import sys
import heapq as pq
from collections import deque
import math

STUDENT_ID = 'a1884774' # your student ID
DEGREE = 'UG' # or PG if you are in the postgraduate course

def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# def bfs(graph, start, end, n):
#     q = deque()
#     q.append((start, []))

#     seen = set()
    
    
#     while len(q):
#         a = q.popleft()
#         # print(a)
#         x = a[0]
#         # print("X", x)
#         i, j = x

#         dirs = {(-1, 0), (1, 0), (0, -1), (0, 1)}

#         a[1].append(a[0])
#         seen.add(a[0])

#         if a[0] == end:
#             return a[1]

#         close = []

#         for dirx, diry in dirs:
#             tempx = i + dirx
#             tempy = i + diry

#             if (tempx >= 0 and tempy >= 0 and tempx < n[0] and tempy < n[1]):
#                 if (graph[tempx][tempy] != 'X'):
#                     close.append((tempx, tempy))
        
#         for thing in close:
#             if thing not in seen:
#                 q.append((thing, a[1][:]))
#                 seen.add(thing)

def bfs(graph, start, end, n):
    rows, cols = n
    q = deque()
    q.append((start, []))

    seen = set()

    while q:
        current, path = q.popleft()
        i, j = current

        new_path = path + [current] 
        if current == end:
            return new_path

        seen.add(current)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if graph[ni][nj] != 'X' and neighbor not in seen:
                    q.append((neighbor, new_path))
                    seen.add(neighbor)

def path_cost(from_node, to_node):
    return 1 + max(0, to_node - from_node)

def ucs(graph, start, end, n):
    rows, cols = n
    fringe = []
    pq.heappush(fringe, (0, start, []))

    seen = {}

    while fringe:
        total_cost, current, path = pq.heappop(fringe)
        i, j = current

        if current in seen and seen[current] <= total_cost:
            continue
        seen[current] = total_cost

        new_path = path + [current]

        if current == end:
            return new_path

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if graph[ni][nj] != 'X':
                    from_elev = int(graph[i][j])
                    to_elev = int(graph[ni][nj])
                    step = path_cost(from_elev, to_elev)
                    pq.heappush(fringe, (total_cost + step, neighbor, new_path))

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]
    
def astar(grid, start, goal, size, heuristic):
    rows, cols = size

    open_set = []
    pq.heappush(open_set, (heuristic(start, goal), start))

    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    in_open_set = {start}

    while open_set:
        _, current = pq.heappop(open_set)
        in_open_set.discard(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        i, j = current
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if grid[ni][nj] == 'X':
                    continue

                from_elev = int(grid[i][j])
                to_elev = int(grid[ni][nj])
                move_cost = path_cost(from_elev, to_elev)

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    if neighbor not in in_open_set:
                        pq.heappush(open_set, (f_score[neighbor], neighbor))
                        in_open_set.add(neighbor)


def graph_search():
    print('graph_search')

def main():
    mode = sys.argv[1]
    map_file = sys.argv[2]
    algo = sys.argv[3]
    heuristic = sys.argv[4] if len(sys.argv) > 4 else None

    # Read map and positions
    with open(map_file, 'r') as f:
        lines = f.read().splitlines()

    grid_rows, grid_cols = map(int, lines[0].split())
    start_row, start_col = map(int, lines[1].split())
    end_row, end_col = map(int, lines[2].split())
    grid = [line.split() for line in lines[3:]]

    start = (start_row - 1, start_col - 1)
    end = (end_row - 1, end_col - 1)
    size = (grid_rows, grid_cols)

    # Choose algorithm
    if algo == 'bfs':
        path = bfs(grid, start, end, size)
    elif algo == 'ucs':
        path = ucs(grid, start, end, size)
    elif algo == 'astar':
        if heuristic == 'euclidean':
            h_fn = euclidean
        elif heuristic == 'manhattan':
            h_fn = manhattan
        else:
            print("Invalid heuristic. Use 'euclidean' or 'manhattan'.")
            return
        path = astar(grid, start, end, size, h_fn)
    else:
        print("Invalid algorithm. Use 'bfs', 'ucs', or 'astar'.")
        return

    if mode == 'debug':
        print("path:")
        if path:
            for i, j in path:
                grid[i][j] = '*'
            for row in grid:
                print(" ".join(row))
        else:
            print("null")

        print("#visits:\n...")
        print("first visit:\n...")
        print("last visit:\n...")

    elif mode == 'release':
        if path:
            for i, j in path:
                grid[i][j] = '*'
            for row in grid:
                print(" ".join(row))
        else:
            print("path:")
            print("null")
    else:
        print("Invalid mode. Use 'debug' or 'release'.")
        

if __name__ == "__main__":
    main()


