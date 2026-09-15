"""CNT3054: 유튜브 최상위 댓글 수집 실습.
참고: 111usionBin/jtbc-2025의 data_scrape.py. 수업용으로 별도 작성.
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent
COLUMNS = ["video_id", "comment_id", "text_raw", "published_at",
           "like_count", "reply_count", "collected_at"]


def parse_video_id(value):
    """영상 ID 또는 일반 영상/공유/Shorts/live URL에서 ID를 읽는다."""
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    u = urlparse(value)
    host = (u.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = u.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        pieces = u.path.strip("/").split("/")
        candidate = (parse_qs(u.query).get("v", [""])[0]
                     if u.path == "/watch" else
                     pieces[1] if len(pieces) > 1 and pieces[0] in {"shorts", "live", "embed"} else "")
    else:
        candidate = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        raise ValueError("영상 ID 11자리 또는 유튜브 영상 URL을 입력하세요. 재생목록 URL은 사용할 수 없습니다.")
    return candidate


def make_youtube_client():
    """코드와 같은 폴더의 .env에서 키를 읽고 API 클라이언트를 만든다."""
    from dotenv import load_dotenv
    from googleapiclient.discovery import build
    import httplib2
    load_dotenv(BASE / ".env", override=False)
    api_key = (os.getenv("google_cloud_api_key") or "").strip()
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError(".env의 google_cloud_api_key에 본인 API 키를 입력하세요.")
    return build("youtube", "v3", developerKey=api_key,
                 http=httplib2.Http(timeout=30), cache_discovery=False)


def get_video_info(youtube, video_id):
    response = youtube.videos().list(part="snippet", id=video_id).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError("조회 가능한 영상이 없습니다. 영상 URL과 공개 여부를 확인하세요.")
    s = items[0]["snippet"]
    return {"video_id": video_id, "title": s["title"],
            "channel_title": s["channelTitle"], "channel_id": s["channelId"],
            "video_published_at": s["publishedAt"]}


def parse_comment(item, video_id, collected_at):
    """중첩 딕셔너리에서 CSV 한 행에 필요한 값만 선택한다."""
    top = item["snippet"]["topLevelComment"]
    comment = top["snippet"]
    return {
        "video_id": video_id,
        "comment_id": top["id"],
        "text_raw": comment["textDisplay"],
        "published_at": comment["publishedAt"],
        "like_count": comment["likeCount"],
        "reply_count": item["snippet"].get("totalReplyCount", 0),
        "collected_at": collected_at,
    }


def error_info(exc):
    """키가 포함될 수 있는 요청 URL 대신 상태 코드와 오류 이유만 반환한다."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    reason = type(exc).__name__
    try:
        body = json.loads(exc.content)
        reason = body["error"]["errors"][0]["reason"]
    except (AttributeError, KeyError, IndexError, ValueError, TypeError):
        pass
    return {"http_status": status, "reason": reason}


def collect_comments(youtube, video_id, max_pages=1, page_size=100):
    rows, seen = [], set()
    page_token = None
    log = {"pages_requested": 0, "pages_received": 0,
           "received_items": 0, "duplicates_skipped": 0,
           "stop_reason": "page_limit", "has_more": None,
           "error": None, "collected_at": datetime.now(timezone.utc).isoformat()}
    for _ in range(max_pages):
        try:
            log["pages_requested"] += 1
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id,
                maxResults=page_size, order="time",
                textFormat="plainText", pageToken=page_token,
            ).execute()
        except Exception as exc:
            log.update(stop_reason="error", has_more=None, error=error_info(exc))
            break
        log["pages_received"] += 1
        for item in response.get("items", []):
            log["received_items"] += 1
            row = parse_comment(item, video_id, log["collected_at"])
            if row["comment_id"] in seen:
                log["duplicates_skipped"] += 1
                continue
            seen.add(row["comment_id"])
            rows.append(row)
        page_token = response.get("nextPageToken")
        log["has_more"] = bool(page_token)
        if not page_token:
            log["stop_reason"] = "no_next_page"
            break
    log["saved_rows"] = len(rows)
    return rows, log


def save_results(rows, log, video_info, max_pages, page_size, output_root, demo=False):
    """실행마다 새 폴더에 원문 CSV와 수집 조건 JSON을 함께 저장한다."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = output_root / f"{video_info['video_id']}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "comments.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"mode": "DEMO_FICTIONAL" if demo else "LIVE_API", **video_info,
                "order": "time", "scope": "top_level_only",
                "max_pages": max_pages, "page_size": page_size,
                "comment_date_filter": None, **log}
    (folder / "collection_log.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


class DemoRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class DemoYouTube:
    """실제 API를 호출하지 않는 가상 응답. JTBC 실제 댓글이 아니다."""
    def __init__(self):
        self.pages = json.loads((BASE / "demo_pages.json").read_text(encoding="utf-8"))

    def commentThreads(self):
        return self

    def list(self, **kwargs):
        index = 1 if kwargs.get("pageToken") == "DEMO_PAGE_2" else 0
        return DemoRequest(self.pages[index])


def main():
    parser = argparse.ArgumentParser(description="유튜브 최상위 댓글을 CSV로 저장합니다.")
    parser.add_argument("--video", help="영상 URL 또는 영상 ID")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="키와 영상 조회만 확인")
    parser.add_argument("--demo", action="store_true", help="가상 응답으로 실행")
    args = parser.parse_args()
    if args.max_pages < 1 or not 1 <= args.page_size <= 100:
        parser.error("max-pages는 1 이상, page-size는 1~100이어야 합니다.")
    if args.demo and args.check:
        parser.error("--demo와 --check는 함께 사용하지 않습니다.")
    if not args.demo and not args.video:
        parser.error("--video에 영상 URL 또는 ID를 입력하세요.")
    try:
        if args.demo:
            youtube = DemoYouTube()
            info = {"video_id": "DEMO_VIDEO", "title": "가상 뉴스 영상: 수업용",
                    "channel_title": "가상 채널", "channel_id": "DEMO_CHANNEL",
                    "video_published_at": "2026-09-01T00:00:00Z"}
            print("[DEMO] 실제 JTBC 댓글이 아닌 가상 응답입니다. API 호출 없음.")
            print("데모는 페이지당 2개, 총 2페이지로 고정되어 있습니다.")
            page_size = 2
        else:
            video_id = parse_video_id(args.video)
            youtube = make_youtube_client()
            info = get_video_info(youtube, video_id)
            page_size = args.page_size
        print(f"채널: {info['channel_title']} / 제목: {info['title']}")
        if args.check:
            print("API 연결 및 영상 조회 성공. 선택한 JTBC 뉴스룸 영상인지 확인하세요.")
            return 0
        rows, log = collect_comments(youtube, info["video_id"], args.max_pages, page_size)
        folder = save_results(rows, log, info, args.max_pages, page_size, BASE / "output", args.demo)
        print(f"저장: {len(rows)}개 / 받은 페이지: {log['pages_received']}")
        print(f"종료 이유: {log['stop_reason']} / 다음 페이지 존재: {log['has_more']}")
        print(f"저장 폴더: {folder}")
        if log["error"]:
            print(f"오류: {log['error']} / 지금까지 받은 댓글만 저장했습니다.")
            return 1
        return 0
    except (ValueError, ModuleNotFoundError) as exc:
        print(f"설정 확인: {exc}")
        return 1
    except Exception as exc:
        print(f"요청 실패: {error_info(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
