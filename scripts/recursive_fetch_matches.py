import os, json, time, asyncio, aiohttp
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
GAME_NAME = os.getenv("SEED_GAME_NAME")
TAG_LINE = os.getenv("SEED_TAG_LINE")
HEADERS = {"X-Riot-Token": API_KEY}
REGION = "asia"
QUEUE_ID = 450

semaphore = asyncio.Semaphore(10)
visited_puuids = set()
collected_matches = set()
match_data = []

async def safe_get(session, url):
    async with semaphore:
        for _ in range(3):
            try:
                async with session.get(url, headers=HEADERS) as res:
                    if res.status == 200:
                        return await res.json()
                    elif res.status == 429:
                        retry = float(res.headers.get("Retry-After", 1.5))
                        print(f"Rate Limit 잠시 대기 ({retry}초)")
                        await asyncio.sleep(retry)
                    else:
                        print(f"요청 실패 URL: {url} | 상태코드: {res.status}")
                        return None
            except Exception as e:
                print(f"[예외 발생] {e}")
                await asyncio.sleep(1)
    return None

async def get_puuid(session, game_name, tag_line):
    url = f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    print(f"PUUID 요청 {game_name}#{tag_line}")
    data = await safe_get(session, url)
    if data:
        print(f"PUUID 성공 {data['puuid']}")
    return data.get("puuid") if data else None

async def fetch_matches_by_puuid(session, puuid, depth=0):
    if puuid in visited_puuids or depth > 2:
        return
    visited_puuids.add(puuid)

    print(f"[{depth}단계] {puuid} 매치 조회 중")
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=10&queue={QUEUE_ID}"
    match_ids = await safe_get(session, url) or []

    for match_id in match_ids:
        if match_id in collected_matches:
            continue
        collected_matches.add(match_id)

        print(f"매치 수집: {match_id}")
        detail_url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        match = await safe_get(session, detail_url)
        if not match:
            continue
        match_data.append(match)

        if len(match_data) >= 100:
            print(f"\n수집 완료 100개 수집됨. 저장 중")
            os.makedirs("LOLCLUSTER/data", exist_ok=True)
            with open("LOLCLUSTER/data/raw_matches.json", "w", encoding="utf-8") as f:
                json.dump(match_data, f, indent=2)
            print("저장 완료: LOLCLUSTER/data/raw_matches.json")
            exit(0)

        for p in match['metadata']['participants']:
            await fetch_matches_by_puuid(session, p, depth + 1)

async def main():
    async with aiohttp.ClientSession() as session:
        puuid = await get_puuid(session, GAME_NAME, TAG_LINE)
        if not puuid:
            print("PUUID를 찾을 수 없습니다. 이름/태그를 확인하세요.")
            return
        await fetch_matches_by_puuid(session, puuid)

    os.makedirs("LOLCLUSTER/data", exist_ok=True)
    with open("LOLCLUSTER/data/raw_matches.json", "w", encoding="utf-8") as f:
        json.dump(match_data, f, indent=2)
    print(f"\n총 수집된 매치 수: {len(match_data)}")
    print("저장 완료: LOLCLUSTER/data/raw_matches.json")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
