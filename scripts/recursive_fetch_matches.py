import os, json, asyncio, aiohttp
from dotenv import load_dotenv
from collections import deque

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
GAME_NAME = os.getenv("SEED_GAME_NAME")
TAG_LINE = os.getenv("SEED_TAG_LINE")
HEADERS = {"X-Riot-Token": API_KEY}
REGION = "asia"
QUEUE_ID = 450

SAVE_INTERVAL = 100
FILE_INTERVAL = 1000
MAX_DEPTH = 4
semaphore = asyncio.Semaphore(10)

DATA_DIR = "LOLCLUSTER/data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_set(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return set(data)
            except json.JSONDecodeError:
                print(f"경고: {path} 파일이 비어 있거나 손상됨. 빈 set으로 처리.")
                return set()
    return set()

def save_set(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(data), f)

def get_file_index_and_total():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("matches_") and f.endswith(".json")]
    if not files:
        return 0, 0
    files.sort()
    latest = files[-1]
    index = int(latest.split("_")[1].split(".")[0])
    with open(os.path.join(DATA_DIR, latest), "r", encoding="utf-8") as f:
        data = json.load(f)
    return index, len(data)

collected_matches = load_set(os.path.join(DATA_DIR, "collected_matches.json"))
visited_puuids = load_set(os.path.join(DATA_DIR, "visited_puuids.json"))
match_data = []
file_index, file_total = get_file_index_and_total()

def save_batch():
    global match_data, file_index, file_total

    path = os.path.join(DATA_DIR, f"matches_{file_index:04d}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing += match_data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"저장 완료 {len(match_data)}개 → {path} (총: {len(existing)})")

    match_data.clear()
    file_total = len(existing)

    save_set(collected_matches, os.path.join(DATA_DIR, "collected_matches.json"))
    save_set(visited_puuids, os.path.join(DATA_DIR, "visited_puuids.json"))

    if file_total >= FILE_INTERVAL:
        file_index += 1
        file_total = 0

async def safe_get(session, url):
    async with semaphore:
        for _ in range(3):
            try:
                async with session.get(url, headers=HEADERS) as res:
                    if res.status == 200:
                        return await res.json()
                    elif res.status == 429:
                        retry = float(res.headers.get("Retry-After", 1.5))
                        print(f"Rate Limit 발생. {retry}초 대기")
                        await asyncio.sleep(retry)
                    else:
                        print(f"요청 실패: {url} | 상태코드: {res.status}")
                        return None
            except Exception as e:
                print(f"예외 발생: {e}")
                await asyncio.sleep(1)
    return None

async def get_puuid(session, game_name, tag_line):
    url = f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    print(f"PUUID 요청: {game_name}#{tag_line}")
    data = await safe_get(session, url)
    if data and 'puuid' in data:
        print(f"PUUID 성공: {data['puuid']}")
        return data['puuid']
    print("PUUID 실패: 확인 필요")
    return None

async def fetch_matches_bfs(session, root_puuid):
    queue = deque([(root_puuid, 0)])

    while queue:
        puuid, depth = queue.popleft()
        if puuid in visited_puuids or depth > MAX_DEPTH:
            continue
        visited_puuids.add(puuid)

        print(f"{depth}단계 매치 조회 중: {puuid}")
        url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=100&queue={QUEUE_ID}"
        match_ids = await safe_get(session, url) or []

        for match_id in match_ids:
            if match_id in collected_matches:
                continue
            collected_matches.add(match_id)

            print(f"→ 매치 수집: {match_id}")
            detail_url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
            match = await safe_get(session, detail_url)
            if not match:
                continue
            match_data.append(match)

            if len(match_data) >= SAVE_INTERVAL:
                save_batch()
                continue 

            for p_puuid in match.get('metadata', {}).get('participants', []):
                if p_puuid not in visited_puuids:
                    queue.append((p_puuid, depth + 1))

async def main():
    async with aiohttp.ClientSession() as session:
        puuid = await get_puuid(session, GAME_NAME, TAG_LINE)
        if not puuid:
            print("PUUID 가져오기 실패")
            return
        await fetch_matches_bfs(session, puuid)

    if match_data:
        save_batch()
    print(f"총 수집된 match 수: {len(collected_matches)}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
