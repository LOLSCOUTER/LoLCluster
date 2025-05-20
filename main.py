import time
import subprocess

def run_script(path):
    print(f"실행 중: {path}")
    try:
        subprocess.run(["python", path], check=True)
    except subprocess.CalledProcessError:
        print(f"실패 무시: {path}")

while True:
    print("\n새로운 루프 시작\n")

    run_script("scripts/recursive_fetch_matches.py")
    run_script("scripts/vectorize_champions.py")
    run_script("scripts/cluster_roles.py")
    run_script("scripts/train_recommendation.py")

    print("5초 대기 후 다음 루프로...\n")
    time.sleep(5)
