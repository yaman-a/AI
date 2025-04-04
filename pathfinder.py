import sys
import heapq as pq
from collections import deque
import math

STUDENT_ID = 'a1884774' # your student ID
DEGREE = 'UG' # or PG if you are in the postgraduate course

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
        curr, path = q.popleft()
        i, j = curr

        new_path = path + [curr] 
        if curr == end:
            return new_path

        seen.add(curr)

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if graph[ni][nj] != 'X' and neighbor not in seen:
                    q.append((neighbor, new_path))
                    seen.add(neighbor)

def path_cost(from_n, to_n):
    return 1 + max(0, to_n - from_n)

def ucs(graph, start, goal, size):
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
                if graph[ni][nj] == 'X':
                    continue

                from_e = int(graph[i][j])
                to_e = int(graph[ni][nj])
                step = path_cost(from_e, to_e)
                new_cost = cost_so_far[current] + step

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    dest_start[neighbor] = current
                    pq.heappush(q, (new_cost, count, neighbor))
                    count += 1

    return None

def reconstruct_path(dest_start, curr):
    path = [curr]
    while curr in dest_start:
        curr = dest_start[curr]
        path.append(curr)
    return path[::-1]
    
def astar(graph, start, end, n, heuristic):
    count = 0
    rows, cols = n

    open_set = []
    pq.heappush(open_set, (heuristic(start, end), count, start))
    count += 1

    dest_start = {}
    g_score = {start: 0}

    while open_set:
        _, _, curr = pq.heappop(open_set)

        if curr == end:
            return reconstruct_path(dest_start, curr)

        i, j = curr
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in dirs:
            ni, nj = i + dx, j + dy
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor = (ni, nj)
                if graph[ni][nj] == 'X':
                    continue

                from_e = int(graph[i][j])
                to_e = int(graph[ni][nj])
                move_cost = path_cost(from_e, to_e)
                notconfirmed_g = g_score[curr] + move_cost

                if notconfirmed_g < g_score.get(neighbor, float('inf')):
                    dest_start[neighbor] = curr
                    g_score[neighbor] = notconfirmed_g
                    pq.heappush(open_set, (notconfirmed_g + heuristic(neighbor, end), count, neighbor))
                    count += 1

    return None

def graph_search():
    print('graph_search')

def main():
    mode = sys.argv[1]
    map_file = sys.argv[2]
    algo = sys.argv[3]
    if len(sys.argv) > 4:
        heuristic = sys.argv[4] 

    with open(map_file, 'r') as f:
        lines = f.read().splitlines()

    graph_rows, graph_cols = map(int, lines[0].split())
    start_row, start_col = map(int, lines[1].split())
    end_row, end_col = map(int, lines[2].split())
    graph = [line.split() for line in lines[3:]]

    start = (start_row - 1, start_col - 1)
    end = (end_row - 1, end_col - 1)
    n = (graph_rows, graph_cols)

    if algo == 'bfs':
        path = bfs(graph, start, end, n)
    elif algo == 'ucs':
        path = ucs(graph, start, end, n)
    elif algo == 'astar':
        if heuristic == 'euclidean':
            h_fn = euclidean
        elif heuristic == 'manhattan':
            h_fn = manhattan
        path = astar(graph, start, end, n, h_fn)

    if mode == 'debug':
        print("path:")
        if path:
            for i, j in path:
                graph[i][j] = '*'
            for row in graph:
                print(" ".join(row))
        else:
            print("null")

        print("#visits:\n...")
        print("first visit:\n...")
        print("last visit:\n...")

    elif mode == 'release':
        if path:
            for i, j in path:
                graph[i][j] = '*'
            for row in graph:
                print(" ".join(row))
        else:
            print("null")
        

if __name__ == "__main__":
    main()


