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

    if team_df.empty:
        print("유효한 팀 데이터가 없습니다. 학습 중단.")
        return

    X = team_df.drop(columns=["label"])
    y = team_df["label"]

    print("학습 데이터 분포:", y.value_counts().to_dict())

    model = CatBoostClassifier(verbose=0)
    start_time = time.time()
    model.fit(X, y)
    elapsed = time.time() - start_time

    os.makedirs("LOLCLUSTER/models", exist_ok=True)
    model_path = "LOLCLUSTER/models/team_model.pkl"

    if os.path.exists(model_path):
        os.replace(model_path, model_path.replace(".pkl", "_backup.pkl"))

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open("train_log.txt", "a", encoding="utf-8") as log:
        log.write(
            f"[{now}] acc: {acc:.4f}, f1: {f1:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, "
            f"data: {len(X)}, time: {elapsed:.2f}s\n"
        )

    print(f"모델 학습 및 저장 완료 ({len(X)}개, {elapsed:.2f}s)")
    print(f"acc: {acc:.4f} | f1: {f1:.4f} | precision: {prec:.4f} | recall: {rec:.4f}")

if __name__ == "__main__":
    run()
