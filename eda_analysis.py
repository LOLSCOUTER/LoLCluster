import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc("font", family="Malgun Gothic")
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("LOLCLUSTER/champion_vectors/champion_with_roles.csv")

print("전체 컬럼명:")
print(df.columns.tolist())

if "role_cluster" in df.columns:
    print("역할 라벨링 클러스터 분포:")
    print(df["role_cluster"].value_counts())
else:
    print("role_cluster 컬럼이 없습니다.")

print("\n결측치 개수:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0)

print("\n수치형 컬럼 통계:")
print(df[numeric_cols].describe())

col_map = {}
for col in df.columns:
    name = col.lower()
    if "kill" in name and "skill" not in name:
        col_map["kills"] = col
    elif "death" in name:
        col_map["deaths"] = col
    elif "assist" in name:
        col_map["assists"] = col
    elif "damage" in name:
        col_map["damage"] = col
    elif "taken" in name:
        col_map["taken"] = col
    elif "heal" in name:
        col_map["heal"] = col
    elif "champ" in name:
        col_map["champion"] = col

print("col_map:", col_map)

if {"kills", "deaths", "assists"}.issubset(col_map):
    df["kda"] = (df[col_map["kills"]] + df[col_map["assists"]]) / df[col_map["deaths"]].replace(0, 1)
    print("\nKDA 통계:")
    print(df["kda"].describe())
    print("KDA 예시:", df["kda"].head())

    plt.figure(figsize=(10, 6))
    sns.histplot(df["kda"], bins=50)
    plt.title("KDA 분포")
    plt.xlabel("KDA")
    plt.savefig("kda_distribution.png")
    plt.close()

    print("kda_distribution.png 저장 시도 완료")
else:
    print("KDA 그래프 생략됨: kills/deaths/assists 컬럼 없음")

if "damage" in col_map:
    print("Damage 예시:", df[col_map["damage"]].head())

    plt.figure(figsize=(10, 6))
    sns.histplot(df[col_map["damage"]], bins=50)
    plt.title("딜량 분포")
    plt.xlabel("damage")
    plt.savefig("damage_distribution.png")
    plt.close()

    print("damage_distribution.png 저장 시도 완료")
else:
    print("딜량 그래프 생략됨: damage 컬럼 없음")

if "role_cluster" in df.columns and "champion" in col_map:
    print("\n클러스터별 평균 스탯:")
    stat_cols = [col_map.get(k) for k in ["kills", "deaths", "assists", "damage", "taken", "heal"] if col_map.get(k)]
    print(df.groupby("role_cluster")[stat_cols].mean())

    print("\n클러스터별 대표 챔피언 상위 10개:")
    for cluster_id in sorted(df["role_cluster"].dropna().unique()):
        top_champs = df[df["role_cluster"] == cluster_id][col_map["champion"]].value_counts().head(10)
        print(f"\n클러스터 {int(cluster_id)}:")
        print(top_champs)
else:
    print("\nrole_cluster 또는 champion 컬럼이 누락되어 클러스터별 챔피언 출력 불가.")
