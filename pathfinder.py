import sys
import heapq as pq
from collections import deque
import math
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

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if graph[ni][nj] != 'X' and neighbor not in seen:
                    q.append((neighbor, new_path))
                    seen.add(neighbor)

def path_cost(from_node, to_node):
    return 1 + max(0, to_node - from_node)

def ucs(grid, start, goal, size):
    count = 0
    rows, cols = size

    q = []
    pq.heappush(q, (0, count, start))
    count += 1

    dest_start = {}
    cost_so_far = {start: 0}

    while q:
        current_cost, _, current = pq.heappop(q)

        if current == goal:
            return reconstruct_path(dest_start, current)

        i, j = current
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if grid[ni][nj] == 'X':
                    continue

                from_e = int(grid[i][j])
                to_e = int(grid[ni][nj])
                step = path_cost(from_e, to_e)
                new_cost = cost_so_far[current] + step

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    dest_start[neighbor] = current
                    pq.heappush(q, (new_cost, count, neighbor))
                    count += 1

    return None


def reconstruct_path(dest_start, current):
    path = [current]
    while current in dest_start:
        current = dest_start[current]
        path.append(current)
    return path[::-1]
    
def astar(grid, start, goal, size, heuristic):
    count = 0
    rows, cols = size

    open_set = []
    pq.heappush(open_set, (heuristic(start, goal), count, start))
    count += 1

    dest_start = {}
    g_score = {start: 0}

    while open_set:
        _, _, current = pq.heappop(open_set)

        if current == goal:
            return reconstruct_path(dest_start, current)

        i, j = current
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if grid[ni][nj] == 'X':
                    continue

                from_e = int(grid[i][j])
                to_e = int(grid[ni][nj])
                move_cost = path_cost(from_e, to_e)
                notconfirmed_g = g_score[current] + move_cost

                if notconfirmed_g < g_score.get(neighbor, float('inf')):
                    dest_start[neighbor] = current
                    g_score[neighbor] = notconfirmed_g
                    pq.heappush(open_set, (notconfirmed_g + heuristic(neighbor, goal), count, neighbor))
                    count += 1

    return None

def graph_search():
    print('graph_search')

def main():
    mode = sys.argv[1]
    map_file = sys.argv[2]
    algo = sys.argv[3]
    heuristic = sys.argv[4] if len(sys.argv) > 4 else None

    with open(map_file, 'r') as f:
        lines = f.read().splitlines()

    grid_rows, grid_cols = map(int, lines[0].split())
    start_row, start_col = map(int, lines[1].split())
    end_row, end_col = map(int, lines[2].split())
    grid = [line.split() for line in lines[3:]]

    start = (start_row - 1, start_col - 1)
    end = (end_row - 1, end_col - 1)
    size = (grid_rows, grid_cols)

    if algo == 'bfs':
        path = bfs(grid, start, end, size)
    elif algo == 'ucs':
        path = ucs(grid, start, end, size)
    elif algo == 'astar':
        if heuristic == 'euclidean':
            h_fn = euclidean
        elif heuristic == 'manhattan':
            h_fn = manhattan
        path = astar(grid, start, end, size, h_fn)

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
        

if __name__ == "__main__":
    main()


