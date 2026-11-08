from pathlib import Path
import pandas as pd
from pybaseball import bwar_bat

raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "milb_hitting"
milb = pd.concat([pd.read_csv(f) for f in raw_dir.glob("*.csv") if f.stat().st_size > 1])
war = bwar_bat(return_all=True)

print("Latest WAR season:", war["year_ID"].max())

war_ids = set(war["mlb_ID"].dropna().astype(int))
hitters = milb[(milb["position_type"] != "Pitcher") & (milb["league"] != "MEX")]
milb_ids = hitters["player_id"].unique()
matched = sum(pid in war_ids for pid in milb_ids)
print(f"Minor league hitters: {len(milb_ids)}, found in MLB WAR file: {matched} ({matched / len(milb_ids):.1%})")

# Spot check a known star
# trout = milb[milb["name"] == "Mike Trout"]
# print("\nMike Trout's minor league rows:")
# print(trout[["season", "age", "sport_id", "league", "pa"]])
# trout_id = trout["player_id"].iloc[0]
# print("\nHis MLB WAR rows:")
# print(war[war["mlb_ID"] == trout_id][["year_ID", "age", "PA", "WAR"]])

ROOT = Path(__file__).resolve().parent.parent
labels = pd.read_csv(ROOT / "data" / "processed" / "labels.csv")
milb = pd.concat([pd.read_csv(f) for f in (ROOT / "data" / "raw" / "milb_hitting").glob("*.csv")
                  if f.stat().st_size > 1])
names = milb.groupby("player_id")["name"].first()
labels["name"] = labels["player_id"].map(names)

cols = ["name", "position", "career_war", "graduation_season", "last_mlb_season", "career_complete"]
print("Tier 5:")
print(labels[labels["tier"] == 5].sort_values("career_war", ascending=False)[cols].to_string(index=False))
print("\nTop 10 of tier 4:")
print(labels[labels["tier"] == 4].sort_values("career_war", ascending=False)[cols].head(10).to_string(index=False))
print("\nMike Trout:")
print(labels[labels["name"] == "Mike Trout"][cols + ["tier"]].to_string(index=False))