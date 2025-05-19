import time
import subprocess

def run_script(path):
    try:
        subprocess.run(["python", path], check=True)
    except subprocess.CalledProcessError:
        print(f"실패 무시 {path}")

while True:
    print("매치 수집")
    run_script("scripts/recursive_fetch_matches.py")

    print("챔피언 벡터화")
    run_script("scripts/vectorize_champions.py")

    print("역할 클러스터링")
    run_script("scripts/cluster_roles.py")

    print("추천 모델 학습")
    run_script("scripts/train_recommendation.py")

    print("다음 루프로 바로 이동\n")
    time.sleep(5) 
