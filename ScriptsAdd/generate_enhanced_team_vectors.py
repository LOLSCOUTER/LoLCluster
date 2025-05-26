import pandas as pd
import numpy as np
from scipy.stats import entropy
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import time
import os

# 경로 설정
INPUT_PATH = "LOLCLUSTER/champion_vectors/champion_with_roles.csv"
OUTPUT_CSV = "LOLCLUSTER/champion_vectors/team_vectors_v3.csv"
OUTPUT_PKL = "LOLCLUSTER/models/team_model_v3.pkl"

# 팀 벡터 생성 함수
def make_team_vector_enhanced(team_df):
    team_df = team_df.sort_values(by='champion')
    vector = []

    total_kills = team_df['kills'].sum()
    total_assists = team_df['assists'].sum()
    total_deaths = team_df['deaths'].sum()
    team_kda = (total_kills + total_assists) / max(total_deaths, 1)

    avg_damage = team_df['damage'].mean()
    avg_taken = team_df['taken'].mean()
    avg_heal = team_df['heal'].mean()

    role_counts = team_df['role_cluster'].value_counts(normalize=True).sort_index()
    role_dist = np.zeros(5)
    for i in range(5):
        if i in role_counts:
            role_dist[i] = role_counts[i]
    role_entropy = entropy(role_dist + 1e-10)

    vector += [avg_damage, avg_taken, avg_heal, team_kda, role_entropy]

    for _, row in team_df.iterrows():
        vector += [
            row['champion'],
            row['kills'], row['deaths'], row['assists'],
            row['damage'], row['taken'], row['heal'],
            row['role_cluster']
        ]
        vector += row.filter(like='item_').tolist()

    return vector

# CSV 저장용 header 설정
raw_df = pd.read_csv(INPUT_PATH)
first_vec = None
for (_, _), group in raw_df.groupby(['match_id', 'team_id']):
    if len(group) == 5:
        first_vec = make_team_vector_enhanced(group)
        break
if first_vec is None:
    raise ValueError("데이터셋에 유효한 팀이 없습니다.")

extra_cols = ['avg_damage', 'avg_taken', 'avg_heal', 'team_kda', 'role_entropy']
player_cols = ['champion', 'kills', 'deaths', 'assists', 'damage', 'taken', 'heal', 'role_cluster']
item_cols = [col for col in raw_df.columns if col.startswith("item_")]
header = extra_cols + player_cols * 5 + item_cols * 5 + ['match_id', 'team_id', 'win']

# Streaming으로 저장
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
    f.write(",".join(map(str, header)) + "\n")
    for (match_id, team_id), group in raw_df.groupby(['match_id', 'team_id']):
        if len(group) != 5:
            continue
        vec = make_team_vector_enhanced(group)
        vec += [match_id, team_id, group['win'].iloc[0]]
        f.write(",".join(map(str, vec)) + "\n")
print(f"Streaming 방식으로 CSV 저장 완료 → {OUTPUT_CSV}")

# 모델 학습
team_df = pd.read_csv(OUTPUT_CSV)
X = team_df.drop(columns=['match_id', 'team_id', 'win'])
y = team_df['win']
cat_features = X.select_dtypes(include='object').columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    eval_metric='F1',
    verbose=100
)

start = time.time()
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)
elapsed = time.time() - start

# 평가 지표
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
print(f"acc: {acc:.4f} | f1: {f1:.4f} | precision: {prec:.4f} | recall: {recall:.4f}")

# 로그 저장
now = time.strftime('%Y-%m-%d %H:%M:%S')
with open("train_log.txt", "a", encoding="utf-8") as log:
    log.write(
        f"[{now}] acc: {acc:.4f}, f1: {f1:.4f}, precision: {prec:.4f}, recall: {recall:.4f}, data: {len(X)}, time: {elapsed:.2f}s\n"
    )

# 모델 저장
os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(model, f)
print(f"모델 저장 완료: {OUTPUT_PKL}")
