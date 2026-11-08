"""
explore_api.py  (Stage 1)
Asks the MLB Stats API two questions and prints what comes back,
so we can see the real data's shape before building the pipeline.
Run from the project root:  python src/explore_api.py
"""

import json
import requests

BASE = "https://statsapi.mlb.com/api/v1"

# --- Question 1: which levels exist, and what are their IDs? ---
sports = requests.get(f"{BASE}/sports", timeout=30).json()["sports"]
print("Levels and their sport IDs:")
for s in sports:
    print(f"  {s['id']:>4}  {s['name']}")

# --- Question 2: what does one season of hitting stats look like? ---
params = {
    "stats": "season",
    "group": "hitting",
    "playerPool": "All",
    "sportId": 12,        # Double-A
    "season": 2015,
    "limit": 5000,
}
data = requests.get(f"{BASE}/stats", params=params, timeout=60).json()
splits = data["stats"][0]["splits"]

print(f"\nDouble-A 2015: {len(splits)} rows returned")
print("\nOne full row, so we can see every available field:")
print(json.dumps(splits[0], indent=2))

# --- Question 3: how far back does minor league data go? ---
print("\nRows per season at Double-A:")
for year in [1995, 2000, 2005, 2010]:
    params["season"] = year
    rows = requests.get(f"{BASE}/stats", params=params, timeout=60).json()["stats"][0]["splits"]
    print(f"  {year}: {len(rows)}")
