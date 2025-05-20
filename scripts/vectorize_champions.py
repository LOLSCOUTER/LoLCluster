import os
import json
import pandas as pd

def extract_features(matches):
    rows = []
    for match in matches:
        for p in match['info']['participants']:
            row = {
                "match_id": str(match['metadata']['matchId']),
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
    all_items = sorted({item for items in df['items'] for item in items})
    for item in all_items:
        df[f"item_{item}"] = df['items'].apply(lambda x: int(item in x))
    df.drop(columns=['items'], inplace=True)
    return df

def run():
    all_matches = []
    data_dir = "LOLCLUSTER/data"
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename.startswith("matches_"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                    if isinstance(content, list):
                        all_matches.extend(content)
                except json.JSONDecodeError:
                    print(f"무시: {filename} - JSON 파싱 실패")

    if not all_matches:
        print(" 수집된 매치 데이터가 없습니다.")
        return

    df = extract_features(all_matches)
    df = encode_items(df)
    df["match_id"] = df["match_id"].astype(str) 

    os.makedirs("LOLCLUSTER/champion_vectors", exist_ok=True)
    output_path = "LOLCLUSTER/champion_vectors/champion_vectors.csv"

    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        prev["match_id"] = prev["match_id"].astype(str)  
        df = pd.concat([prev, df], ignore_index=True)
        df.drop_duplicates(subset=["match_id", "team_id", "champion"], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"champion_vectors.csv 저장 완료: {len(df)}개")

if __name__ == "__main__":
    run()
