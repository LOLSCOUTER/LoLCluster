from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import time
import pickle
import os

df = pd.read_csv("LOLCLUSTER/champion_vectors/team_vectors.csv")

X = df.drop(columns=['match_id', 'team_id', 'win'])
y = df['win']

cat_features = X.select_dtypes(include='object').columns.tolist()
print("Categorical features:", cat_features)

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

y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")

now = time.strftime('%Y-%m-%d %H:%M:%S')
with open("train_log.txt", "a", encoding="utf-8") as log:
    log.write(
        f"[{now}] acc: {acc:.4f}, f1: {f1:.4f}, precision: {prec:.4f}, recall: {recall:.4f}, data: {len(X)}, time: {elapsed:.2f}s\n"
    )

os.makedirs("LOLCLUSTER/models", exist_ok=True)
with open("LOLCLUSTER/models/team_model.pkl", "wb") as f:
    pickle.dump(model, f)
