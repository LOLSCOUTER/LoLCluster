import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import os
import time

def create_team_vectors(df):
    data = []
    grouped = df.groupby("match_id")
    for _, group in grouped:
        if len(group) != 10:
            continue
        t1 = group[group["team_id"] == 100]
        t2 = group[group["team_id"] == 200]
        if len(t1) != 5 or len(t2) != 5:
            continue

        def role_count(team):
            counts = team['role_cluster'].value_counts().to_dict()
            return [counts.get(i, 0) for i in range(5)]

        features = role_count(t1) + role_count(t2)
        win_label = int(t1["win"].mean() > t2["win"].mean())
        data.append(features + [win_label])

    columns = [f"t1_role_{i}" for i in range(5)] + [f"t2_role_{i}" for i in range(5)] + ["label"]
    return pd.DataFrame(data, columns=columns)

def run():
    df = pd.read_csv("LOLCLUSTER/champion_vectors/champion_with_roles.csv")
    team_df = create_team_vectors(df)
    X = team_df.drop(columns=["label"])
    y = team_df["label"]

    model = CatBoostClassifier(verbose=0)
    model.fit(X, y)

    os.makedirs("LOLCLUSTER/models", exist_ok=True)
    with open("LOLCLUSTER/models/team_model.pkl", "wb") as f:
        pickle.dump(model, f)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open("train_log.txt", "a", encoding="utf-8") as log:
        log.write(f"[{now}] acc: {acc:.4f}, f1: {f1:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, data: {len(X)}\n")

if __name__ == "__main__":
    run()
