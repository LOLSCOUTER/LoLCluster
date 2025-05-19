import pandas as pd
from sklearn.cluster import KMeans
import os

def run():
    df = pd.read_csv("LOLCLUSTER/champion_vectors/champion_vectors.csv")
    X = df.drop(columns=["champion", "match_id", "team_id", "win"])
    model = KMeans(n_clusters=5, random_state=42)
    df["role_cluster"] = model.fit_predict(X)

    output_path = "LOLCLUSTER/champion_vectors/champion_with_roles.csv"
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(subset=["match_id", "team_id", "champion"])
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    run()
