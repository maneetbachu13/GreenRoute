# utils.py
import requests
import polyline
import math
from typing import List, Tuple, Dict

# -------------------------
# Existing helper functions
# -------------------------
def compute_elevation_gain(polyline_str: str, api_key: str) -> float:
    """Compute total elevation gain (uphill meters) for a route (same as before)."""
    try:
        coords = polyline.decode(polyline_str)
        # sample to limit elevation API calls
        sample = coords[::10] if len(coords) > 50 else coords
        loc_str = "|".join([f"{lat},{lon}" for lat, lon in sample])

        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={loc_str}&key={api_key}"
        res = requests.get(url).json()
        if res.get("status") != "OK":
            return 0.0

        elevations = [r.get("elevation", 0.0) for r in res.get("results", [])]
        gain = sum(max(0, elevations[i + 1] - elevations[i]) for i in range(len(elevations) - 1))
        return gain
    except Exception as e:
        print("⚠️ Elevation computation failed:", e)
        return 0.0


def estimate_emission(distance_km: float, elevation_gain_m: float, base_emission_per_km: float, congestion_index: float = 0.0) -> float:
    """Fallback emission estimate if model is unavailable."""
    elev_factor = 0.001 * (elevation_gain_m / max(distance_km, 0.1))
    traffic_factor = 1 + (0.6 * congestion_index)
    emission = base_emission_per_km * distance_km * (1 + elev_factor) * traffic_factor
    return emission


# -------------------------
# New ACO-related helpers
# -------------------------
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return great-circle distance in kilometers between two (lat, lon) points."""
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0  # earth radius km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    hav = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(hav))


def _round_coord(pt: Tuple[float,float], prec: int = 5) -> Tuple[float,float]:
    """Round coords to merge close points (helps graph node merging)."""
    return (round(pt[0], prec), round(pt[1], prec))


def build_graph_from_routes(routes: List[Dict]) -> Tuple[Dict[Tuple[float,float], List[Tuple[Tuple[float,float], float]]], List[Tuple[float,float]]]:
    """
    Build an undirected graph from given route polylines.
    Nodes are rounded coordinates; edges connect consecutive polyline points.
    Edge weight = distance_km (we will convert to emission later).
    Returns adjacency list: node -> list of (neighbor_node, distance_km).
    Also returns ordered list of nodes for convenience.
    """
    graph = {}
    nodes_order = []
    for r in routes:
        try:
            path = polyline.decode(r["polyline"])[::5]
            # iterate consecutive pairs
            for i in range(len(path)-1):
                a = _round_coord(path[i])
                b = _round_coord(path[i+1])
                d = haversine_km(a, b)
                # add nodes
                if a not in graph:
                    graph[a] = []
                    nodes_order.append(a)
                if b not in graph:
                    graph[b] = []
                    nodes_order.append(b)
                # add undirected edges (store distance)
                graph[a].append((b, d))
                graph[b].append((a, d))
        except Exception as e:
            print("⚠️ build_graph_from_routes error:", e)
            continue
    return graph, nodes_order


def graph_path_distance(path: List[Tuple[float,float]]) -> float:
    """Return total distance (km) along a sequence of nodes (using haversine)."""
    total = 0.0
    for i in range(len(path)-1):
        total += haversine_km(path[i], path[i+1])
    return total


def encode_path_to_polyline(path: List[Tuple[float,float]]) -> str:
    """Encode path (list of lat/lon) into Google polyline string."""
    # ensure decimals are floats
    return polyline.encode([(float(a), float(b)) for a,b in path])


# -------------------------
# ACO algorithm
# -------------------------
import random

def aco_optimize(graph: Dict[Tuple[float,float], List[Tuple[Tuple[float,float], float]]],
                 start: Tuple[float,float],
                 end: Tuple[float,float],
                 base_emission_per_km: float = 0.192,
                 n_ants: int = 20,
                 n_iter: int = 40,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1) -> Dict:
    """
    Simple Ant Colony Optimization on the graph to minimize a path emission-cost.
    - graph: adjacency list {node: [(neighbor, distance_km), ...], ...}
    - start, end: nodes (rounded coords) for origin/destination
    - emission cost on edge = base_emission_per_km * distance_km
    Returns: dictionary with 'path' (list of coords), 'distance_km', 'emission_kgco2', 'polyline'
    NOTE: This is a heuristic and uses simple pheromone + heuristic (1/cost).
    """

    # build edge keys and initial pheromone
    pheromone = {}
    edges_cost = {}
    for u, neighs in graph.items():
        for v, dist in neighs:
            key = (u, v)
            congestion_factor = 1 + (0.6 * random.uniform(0, 0.3))  # or real route-specific index if available
            elev_factor = 0.001 * random.uniform(5, 100) / max(dist, 0.1)
            edges_cost[key] = base_emission_per_km * dist * (1 + elev_factor) * congestion_factor
            pheromone[key] = 1.0  # initial pheromone

    # heuristic = inverse of cost
    heuristic = {k: 1.0 / (v + 1e-9) for k, v in edges_cost.items()}

    best_path = None
    best_cost = float("inf")

    nodes = list(graph.keys())
    if start not in graph or end not in graph:
        # can't run ACO if start/end missing; return empty
        return {"path": [], "distance_km": 0.0, "emission_kgco2": None, "polyline": ""}

    for iteration in range(n_iter):
        all_ant_paths = []
        for ant in range(n_ants):
            current = start
            visited = [current]
            visited_set = set(visited)
            max_steps = len(nodes) * 3
            steps = 0
            stuck = False

            # each ant builds a path until it reaches end or max steps
            while current != end and steps < max_steps:
                neighs = graph.get(current, [])
                # filter feasible neighbors (avoid going back too often)
                choices = []
                probs = []
                for v, dist in neighs:
                    key = (current, v)
                    # avoid immediate back-and-forth by small penalty
                    tau = pheromone.get(key, 1e-9) ** alpha
                    eta = heuristic.get(key, 1e-9) ** beta
                    choices.append((v, dist, key))
                    probs.append(tau * eta)

                if not choices:
                    stuck = True
                    break

                # normalize
                total = sum(probs)
                if total <= 0:
                    # random choose
                    idx = random.randrange(len(choices))
                else:
                    probs = [p/total for p in probs]
                    # roulette wheel selection
                    r = random.random()
                    cum = 0.0
                    idx = 0
                    for i, p in enumerate(probs):
                        cum += p
                        if r <= cum:
                            idx = i
                            break

                chosen = choices[idx][0]
                visited.append(chosen)
                current = chosen
                steps += 1

            # compute path cost if reached end
            if current == end and not stuck:
                # compute emission cost along visited
                path_cost = 0.0
                for i in range(len(visited)-1):
                    key = (visited[i], visited[i+1])
                    path_cost += edges_cost.get(key, edges_cost.get((visited[i+1], visited[i]), 0.0))
                all_ant_paths.append((visited, path_cost))

                if path_cost < best_cost:
                    best_cost = path_cost
                    best_path = visited

        # pheromone evaporation
        for k in list(pheromone.keys()):
            pheromone[k] = (1 - rho) * pheromone[k]

        # pheromone deposit from all ants (proportional to quality)
        for path, cost in all_ant_paths:
            if cost <= 0:
                continue
            deposit = 1.0 / cost  # better (lower cost) => more deposit
            for i in range(len(path)-1):
                key = (path[i], path[i+1])
                pheromone[key] = pheromone.get(key, 0.0) + deposit

        # optional: break early if good stable solution
        # (not necessary here)

    if not best_path:
        print("⚠️ ACO failed to find an optimized path — returning best ML route as fallback.")
        # fallback to LightGBM best route
        if routes_out:
            return {
                "path": polyline.decode(routes_out[0]["polyline"]),
                "distance_km": routes_out[0]["distance_km"],
                "emission_kgco2": routes_out[0]["lgbm_emission"],
                "polyline": routes_out[0]["polyline"]
            }

    total_distance = graph_path_distance(best_path)
    total_emission = 0.0
    for i in range(len(best_path)-1):
        key = (best_path[i], best_path[i+1])
        total_emission += edges_cost.get(key, edges_cost.get((best_path[i+1], best_path[i]), 0.0))

    encoded = encode_path_to_polyline(best_path)
    return {
        "path": best_path,
        "distance_km": round(total_distance, 3),
        "emission_kgco2": round(total_emission, 4),
        "polyline": encoded
    }
