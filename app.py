# app.py
from flask import Flask, request, jsonify, render_template
import os
import requests
import polyline
from dotenv import load_dotenv
from utils import compute_elevation_gain, estimate_emission, build_graph_from_routes, aco_optimize
import csv
from datetime import datetime
from math import sqrt
import joblib
import pandas as pd

# ===============================
# 1. INITIAL SETUP
# ===============================
load_dotenv()
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

app = Flask(__name__)
# Load both ML models if available
rf_model_path = "emission_rf.pkl"
lgb_model_path = "emission_lgbm.pkl"

rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None
lgb_model = joblib.load(lgb_model_path) if os.path.exists(lgb_model_path) else None

print(f"✅ Models Loaded - RF: {rf_model is not None}, LightGBM: {lgb_model is not None}")

# These scores ideally come from training output; update after retrain
MODEL_SCORES = {"Random Forest": 0.96, "LightGBM": 0.95}
# -------------------------------
# Helper: Google Directions call
# -------------------------------
def fetch_directions(origin, destination, api_key):
    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": "driving",
        "alternatives": "true",
        "departure_time": "now",
        "traffic_model": "best_guess",
        "key": api_key
    }
    res = requests.get(directions_url, params=params)
    return res.json()

# -------------------------------
# Route endpoints
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html", MAPS_API_KEY=MAPS_API_KEY)

def find_nearest_node(point, node_list):
    """Find nearest existing graph node to a given point."""
    px, py = point
    nearest = min(node_list, key=lambda n: sqrt((n[0]-px)**2 + (n[1]-py)**2))
    return nearest

@app.route("/compare_routes", methods=["GET"])
def compare_routes():
    global rf_model, lgb_model
    origin = request.args.get("origin")
    destination = request.args.get("destination")
    vehicle_type = request.args.get("vehicle", "car")
    fuel_type = request.args.get("fuel", "petrol")

    print(f"\n🚦 Request: {origin} → {destination}, Fuel: {fuel_type}, Vehicle: {vehicle_type}")

    # 1) Get routes
    directions = fetch_directions(origin, destination, MAPS_API_KEY)
    if directions.get("status") != "OK":
        print("❌ Directions API Error:", directions)
        return jsonify({"error": directions.get("error_message", "Directions failed"), "routes": [], "aco_route": {}})

    routes_out = []
    route_objs = []  # keep minimal objects for graph building

    # 2) Process each route
    for idx, route in enumerate(directions.get("routes", [])):
        try:
            leg = route["legs"][0]
            distance_m = leg["distance"]["value"]
            duration_s = leg["duration"]["value"]
            duration_traffic_s = leg.get("duration_in_traffic", {}).get("value", duration_s)
            distance_km = distance_m / 1000.0
            congestion_index = (duration_traffic_s / duration_s) - 1 if duration_s > 0 else 0.0

            polyline_str = route["overview_polyline"]["points"]

            elevation_gain_m = compute_elevation_gain(polyline_str, MAPS_API_KEY)

            # AQI - using midpoint (best-effort)
            def get_aqi(poly_s):
                try:
                    coords = polyline.decode(poly_s)
                    mid = coords[len(coords)//2]
                    url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={MAPS_API_KEY}"
                    payload = {"location": {"latitude": mid[0], "longitude": mid[1]}, "extraComputations": ["HEALTH_RECOMMENDATIONS"]}
                    r = requests.post(url, json=payload).json()
                    if "currentConditions" in r:
                        idxs = r["currentConditions"].get("indexes", [{}])
                        aqi_data = idxs[0] if idxs else {}
                        return aqi_data.get("aqi"), aqi_data.get("category")
                    else:
                        idxs = r.get("indexes", [{}])
                        aqi_data = idxs[0] if idxs else {}
                        return aqi_data.get("aqi"), aqi_data.get("category")
                except Exception as e:
                    return None, "Unknown"

            aqi, aqi_category = get_aqi(polyline_str)

            # base emission per km by fuel
            base_emission_per_km = 0.192
            if fuel_type.lower() == "diesel":
                base_emission_per_km = 0.171
            elif fuel_type.lower() == "bike":
                base_emission_per_km = 0.103
            elif fuel_type.lower() == "bus":
                base_emission_per_km = 0.27
            elif fuel_type.lower() == "ev":
                base_emission_per_km = 0.07

            # prepare features
            fuel_encoded = {"petrol": 0, "diesel": 1, "ev": 2, "bike": 3, "bus": 4}.get(fuel_type.lower(), 0)
            X_pred = pd.DataFrame([[distance_km, elevation_gain_m, congestion_index, base_emission_per_km, fuel_encoded]],
                                  columns=["distance_km", "elevation_gain_m", "congestion_index", "base_emission_per_km", "fuel_encoded"])
            # predictions
            try:
                rf_emission = float(rf_model.predict(X_pred)[0]) if rf_model else estimate_emission(distance_km, elevation_gain_m, base_emission_per_km, congestion_index)
            except Exception as e:
                print("⚠️ RF predict error:", e)
                rf_emission = estimate_emission(distance_km, elevation_gain_m, base_emission_per_km, congestion_index)

            try:
                lgbm_emission = float(lgb_model.predict(X_pred)[0]) if lgb_model else estimate_emission(distance_km, elevation_gain_m, base_emission_per_km, congestion_index)
            except Exception as e:
                print("⚠️ LGBM predict error:", e)
                lgbm_emission = estimate_emission(distance_km, elevation_gain_m, base_emission_per_km, congestion_index)

            route_data = {
                "route_index": idx + 1,
                "distance_km": round(distance_km, 3),
                "duration_min": round(duration_s / 60.0, 1),
                "duration_traffic_min": round(duration_traffic_s / 60.0, 1),
                "elevation_gain_m": round(elevation_gain_m, 1),
                "congestion_index": round(congestion_index, 3),
                "rf_emission": round(rf_emission, 4),
                "lgbm_emission": round(lgbm_emission, 4),
                "aqi": aqi,
                "aqi_category": aqi_category,
                "polyline": polyline_str
            }
            routes_out.append(route_data)

            # keep light object for graph building
            route_objs.append({"polyline": polyline_str})

            # log to training CSV (append) + auto retraining every 500 records
            try:
                with open("training_data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        distance_km,
                        elevation_gain_m,
                        congestion_index,
                        fuel_type,
                        base_emission_per_km,
                        estimate_emission(distance_km, elevation_gain_m, base_emission_per_km, congestion_index),  
                        aqi
                    ])

                # 🔄 Auto retraining trigger
                if os.path.exists("training_data.csv"):
                    line_count = sum(1 for _ in open("training_data.csv"))
                    if line_count % 500 == 0:  # retrain every 500 samples
                        print(f"🔁 Retraining models automatically after {line_count} samples...")
                        os.system("py train_model.py")  # run training script

                        # Reload updated models
                        rf_model = joblib.load("emission_rf.pkl")
                        lgb_model = joblib.load("emission_lgbm.pkl")

                        print("✅ Models retrained and reloaded successfully.")
            except Exception as e:
                print("⚠️ Could not log/retrain:", e)

        except Exception as e:
            print("⚠️ Error processing route:", e)
            continue

    # 3) Run ACO on merged graph of all route polylines (if graph possible)
    aco_result = {"path": [], "distance_km": 0.0, "emission_kgco2": None, "polyline": ""}
    try:
        graph, nodes_list = build_graph_from_routes(route_objs)
        if graph and len(graph) > 1:
            # Decode first and last route polylines
            first_path = polyline.decode(routes_out[0]["polyline"])
            last_path = polyline.decode(routes_out[-1]["polyline"])

            # --- Nearest node snapping fix ---
            from math import sqrt
            def find_nearest_node(point, node_list):
                px, py = point
                nearest = min(node_list, key=lambda n: sqrt((n[0]-px)**2 + (n[1]-py)**2))
                return nearest

            start_estimate = (first_path[0][0], first_path[0][1])
            end_estimate = (first_path[-1][0], first_path[-1][1])
            start_node = find_nearest_node(start_estimate, list(graph.keys()))
            end_node = find_nearest_node(end_estimate, list(graph.keys()))

            base_emission = 0.192  # default; could use avg from routes
            print(f"🐜 Running ACO from {start_node} to {end_node} on {len(graph)} nodes...")

            # Run ACO with reduced parameters for speed
            aco = aco_optimize(
                graph, start_node, end_node,
                base_emission_per_km=base_emission,
                n_ants=20, n_iter=40, alpha=1.0, beta=2.0, rho=0.12
            )

            if aco and aco.get("path"):
                aco_result = {
                    "distance_km": aco["distance_km"],
                    "emission_kgco2": aco["emission_kgco2"],
                    "polyline": aco["polyline"]
                }
            else:
                print("⚠️ ACO failed to find an optimized path — returning best ML route as fallback.")
                if routes_out:
                    best = routes_out[0]
                    aco_result = {
                        "distance_km": best["distance_km"],
                        "emission_kgco2": best["lgbm_emission"],
                        "polyline": best["polyline"]
                    }
        else:
            print("⚠️ Graph too small — using fallback route for ACO result.")
            if routes_out:
                best = routes_out[0]
                aco_result = {
                    "distance_km": best["distance_km"],
                    "emission_kgco2": best["lgbm_emission"],
                    "polyline": best["polyline"]
                }

    except Exception as e:
        print("⚠️ ACO error:", e)
        if routes_out:
            best = routes_out[0]
            aco_result = {
                "distance_km": best["distance_km"],
                "emission_kgco2": best["lgbm_emission"],
                "polyline": best["polyline"]
            }
            
    # 4) Determine best route (by LightGBM and compare with ACO)
    best_route = None
    if routes_out:
        best_route = min(routes_out, key=lambda x: x["lgbm_emission"])

    # if ACO returned an emission and it's better than best_route, mark it
    better_than_lgbm = False
    if aco_result.get("emission_kgco2") is not None and best_route:
        try:
            better_than_lgbm = aco_result["emission_kgco2"] < best_route["lgbm_emission"]
        except Exception:
            better_than_lgbm = False

    print(f"✅ Processed {len(routes_out)} routes. Best route index (LGBM) = {best_route['route_index'] if best_route else 'N/A'}; ACO returned {'a route' if aco_result.get('path') else 'no route'}; ACO better: {better_than_lgbm}")

    return jsonify({
        "routes": routes_out,
        "best_route": best_route,
        "aco_route": aco_result,
        "aco_better_than_lgbm": better_than_lgbm
    })


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
