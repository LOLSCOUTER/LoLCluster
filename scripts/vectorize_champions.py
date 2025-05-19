import json
import pandas as pd
import os

def extract_features(matches):
    rows = []
    for match in matches:
        for p in match['info']['participants']:
            row = {
                "match_id": match['metadata']['matchId'],
                "team_id": p['teamId'],
                "champion": p['championName'],
                "win": p['win'],
                "kills": p['kills'],
                "deaths": p['deaths'],
                "assists": p['assists'],
                "damage": p['totalDamageDealtToChampions'],
                "taken": p['totalDamageTaken'],
                "heal": p['totalHeal'],
                "items": [p[f"item{i}"] for i in range(6)]
            }
            rows.append(row)
    return pd.DataFrame(rows)

def encode_items(df):
    all_items = set()
    for items in df['items']:
        all_items.update(items)
    all_items = sorted(all_items)
    for item in all_items:
        df[f"item_{item}"] = df['items'].apply(lambda x: int(item in x))
    df.drop(columns=['items'], inplace=True)
    return df

def run():
    with open("LOLCLUSTER/data/raw_matches.json") as f:
        matches = json.load(f)
    df = extract_features(matches)
    df = encode_items(df)

    os.makedirs("LOLCLUSTER/champion_vectors", exist_ok=True)
    output_path = "LOLCLUSTER/champion_vectors/champion_vectors.csv"
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["match_id", "team_id", "champion"])
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run()
