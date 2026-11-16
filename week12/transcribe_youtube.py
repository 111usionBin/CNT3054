"""YouTube → Deepgram STT → timestamped, speaker-attributed files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = Path(__file__).resolve().parent


def youtube_url(value):
    parsed = urlparse(value.strip())
    host = (parsed.hostname or '').lower()
    if parsed.scheme not in ('https', 'http'):
        raise ValueError('https://로 시작하는 유튜브 영상 링크를 입력하세요.')
    if host in ('youtu.be', 'www.youtu.be'):
        video_id = parsed.path.strip('/').split('/')[0]
    elif host in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com'):
        parts = parsed.path.strip('/').split('/')
        video_id = parse_qs(parsed.query).get('v', [''])[0]
        if len(parts) == 2 and parts[0] in ('shorts', 'live', 'embed'):
            video_id = parts[1]
    else:
        raise ValueError('유튜브 영상 링크만 지원합니다.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
        raise ValueError('개별 영상 링크를 입력하세요. 채널·재생목록 링크는 지원하지 않습니다.')
    return f'https://www.youtube.com/watch?v={video_id}'


def run_command(command, timeout):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=timeout).stdout
    except FileNotFoundError:
        raise RuntimeError(f'실행 프로그램을 찾을 수 없습니다: {command[0]}') from None
    except subprocess.TimeoutExpired:
        raise RuntimeError('다운로드/변환 제한시간을 초과했습니다.') from None
    except subprocess.CalledProcessError:
        # Do not echo subprocess logs: they may contain signed URLs or local data.
        raise RuntimeError('다운로드/변환 실패. yt-dlp 업데이트, Deno·FFmpeg 설치 및 영상 공개 상태를 확인하세요. 로그인·지역 제한 영상은 로컬 파일로 입력하세요.') from None


def download_audio(url, folder, max_minutes):
    if not shutil.which('ffmpeg'):
        raise RuntimeError('FFmpeg가 필요합니다. README의 설치 명령을 실행하세요.')
    command = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '--no-warnings']
    info = json.loads(run_command(command + ['--skip-download', '--dump-single-json', url], 180))
    if info.get('is_live') or info.get('live_status') in ('is_live', 'is_upcoming'):
        raise ValueError('현재 방송 중인 영상은 지원하지 않습니다. 종료된 방송 또는 녹화 영상을 사용하세요.')
    duration = info.get('duration')
    if duration is None or duration <= 0 or duration > max_minutes * 60:
        raise ValueError(f'영상 길이를 확인할 수 없거나 {max_minutes:g}분을 초과했습니다. --max-minutes로 한도를 변경할 수 있습니다.')
    print('[1/3] 유튜브 음성 다운로드 중...', flush=True)
    run_command(command + ['-f', 'bestaudio/best', '-o', str(folder / 'source.%(ext)s'), url], 1800)
    sources = [p for p in folder.glob('source.*') if p.suffix not in ('.part', '.ytdl')]
    if len(sources) != 1:
        raise RuntimeError('다운로드된 음성 파일을 찾을 수 없습니다.')
    return sources[0], {'source_url': url, 'video_id': info.get('id'),
                       'title': info.get('title'), 'duration_seconds': duration}


def prepare_audio(source, folder, max_minutes):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('FFmpeg와 ffprobe를 설치하세요. README를 참고하세요.')
    duration = float(run_command(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                 '-of', 'default=noprint_wrappers=1:nokey=1', str(source)], 60).strip())
    if not math.isfinite(duration) or duration <= 0 or duration > max_minutes * 60:
        raise ValueError(f'오디오 길이는 0분 초과, {max_minutes:g}분 이하여야 합니다.')
    audio = folder / 'audio.wav'
    run_command(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
                 '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(audio)], 600)
    return audio


def transcribe(audio, key, model, language):
    params = {'model': model, 'language': language, 'diarize_model': 'latest',
              'utterances': 'true', 'punctuate': 'true', 'smart_format': 'true'}
    request = Request('https://api.deepgram.com/v1/listen?' + urlencode(params),
                      data=audio.read_bytes(), method='POST',
                      headers={'Authorization': f'Token {key}', 'Content-Type': 'audio/wav'})
    try:
        # No automatic POST retry: a timeout may still have incurred transcription usage.
        with urlopen(request, timeout=600) as response:
            return json.load(response)
    except HTTPError as error:
        advice = {401: 'DEEPGRAM_API_KEY를 확인하세요.', 402: '크레딧 잔액을 확인하세요.',
                  403: 'API 키 권한을 확인하세요.', 429: '요청 한도 초과입니다. 잠시 뒤 다시 실행하세요.'}
        raise RuntimeError(f'Deepgram HTTP {error.code}: ' + advice.get(error.code, '모델·언어 설정 및 서비스 상태를 확인하세요.')) from None
    except (URLError, TimeoutError):
        raise RuntimeError('Deepgram 연결/응답 시간 초과. 처리·과금 여부를 확인한 뒤 다시 실행하세요.') from None


def timestamp(seconds, srt=False):
    ms = round(float(seconds) * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{secs:02}{"," if srt else "."}{ms:03}'


def make_rows(payload, names=None):
    names = names or {}
    results = payload.get('results', {})
    utterances = results.get('utterances') or []
    if not utterances:
        channels = results.get('channels') or []
        alternatives = channels[0].get('alternatives', []) if channels else []
        words = alternatives[0].get('words', []) if alternatives else []
        utterances = [{'words': words}] if words else []
    rows = []

    def append(words, fallback):
        if not words:
            return
        start, end = float(words[0]['start']), float(words[-1]['end'])
        if start < 0 or end < start or not all(map(math.isfinite, (start, end))):
            raise ValueError('잘못된 발화 시간 정보입니다.')
        speaker = words[0].get('speaker', fallback)
        speaker_id = f'SPEAKER_{int(speaker):02}' if speaker is not None else 'UNKNOWN'
        text = ' '.join(w.get('punctuated_word', w.get('word', '')) for w in words).strip()
        if not text:
            return
        rows.append({'utterance_id': f'U{len(rows)+1:04}', 'speaker_id': speaker_id,
                     'speaker_name': names.get(speaker_id, speaker_id),
                     'start_seconds': start, 'end_seconds': end,
                     'start_time': timestamp(start), 'end_time': timestamp(end), 'text': text})

    for utterance in utterances:
        fallback = utterance.get('speaker')
        words = utterance.get('words') or []
        if not words and utterance.get('transcript'):
            words = [{'start': utterance['start'], 'end': utterance['end'],
                      'word': utterance['transcript'], 'speaker': fallback}]
        group = []
        for word in words:
            if group and (word.get('speaker', fallback) != group[-1].get('speaker', fallback)
                          or float(word['start']) - float(group[-1]['end']) > 1.0
                          or float(word['end']) - float(group[0]['start']) > 20):
                append(group, fallback)
                group = []
            group.append(word)
        append(group, fallback)
    return rows


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def save_outputs(folder, payload, metadata, names=None):
    write_json(folder / 'deepgram_raw.json', payload)
    write_json(folder / 'metadata.json', metadata)
    rows = make_rows(payload, names)
    if not rows:
        raise ValueError('인식된 발화가 없습니다. 원본 JSON은 저장했습니다. 음성·언어를 확인하세요.')
    with (folder / 'utterances.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        # Prevent spreadsheet formula execution while keeping JSON/TXT faithful.
        for row in rows:
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
                             for k, v in row.items()})
    write_json(folder / 'utterances.json', rows)
    write_json(folder / 'speaker_names.json', {r['speaker_id']: r['speaker_name'] for r in rows})
    (folder / 'transcript.txt').write_text('\n'.join(
        f"[{r['start_time']} → {r['end_time']}] {r['speaker_name']}: {r['text']}" for r in rows), encoding='utf-8')
    (folder / 'subtitles.srt').write_text('\n\n'.join(
        f"{i}\n{timestamp(r['start_seconds'], True)} --> {timestamp(r['end_seconds'], True)}\n[{r['speaker_name']}] {r['text']}"
        for i, r in enumerate(rows, 1)) + '\n', encoding='utf-8')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='유튜브 링크 → Deepgram 화자별 STT 저장')
    parser.add_argument('url', nargs='?', help='유튜브 영상 URL; 생략하면 입력창 표시')
    parser.add_argument('--setup', action='store_true', help='.env를 처음 생성 (기존 파일 보존)')
    parser.add_argument('--file', type=Path, help='유튜브 대신 로컬 음성/영상 파일')
    parser.add_argument('--response', type=Path, help='기존 deepgram_raw.json 재변환; API 호출 없음')
    parser.add_argument('--speaker-names', type=Path, help='SPEAKER_00 등의 이름 매핑 JSON')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'outputs')
    parser.add_argument('--max-minutes', type=float, default=60)
    parser.add_argument('--keep-audio', action='store_true', help='성공 후 음성 파일 보관')
    args = parser.parse_args(argv)
    try:
        if args.setup:
            target = BASE / '.env'
            if not target.exists():
                target.write_text((BASE / '.env.example').read_text(encoding='utf-8'), encoding='utf-8')
            print(f'{target}\nDEEPGRAM_API_KEY 값을 입력하고 저장하세요. 기존 파일은 덮어쓰지 않습니다.')
            return 0
        if sum(bool(x) for x in (args.url, args.file, args.response)) > 1:
            raise ValueError('URL, --file, --response 중 하나만 사용하세요.')
        if not math.isfinite(args.max_minutes) or not 0 < args.max_minutes <= 240:
            raise ValueError('--max-minutes는 0 초과 240 이하로 지정하세요.')
        names = json.loads(args.speaker_names.read_text(encoding='utf-8')) if args.speaker_names else {}
        if not isinstance(names, dict) or not all(isinstance(v, str) for v in names.values()):
            raise ValueError('화자 이름 파일은 문자열 값으로 이루어진 JSON 객체여야 합니다.')
        source_url = None
        if not args.response:
            from dotenv import load_dotenv
            load_dotenv(BASE / '.env', override=False)
            key = os.getenv('DEEPGRAM_API_KEY', '').strip()
            if not key or key == 'your_deepgram_api_key_here':
                raise ValueError('python transcribe_youtube.py --setup 실행 후 week12/.env에 API 키를 입력하세요.')
            if not args.file:
                source_url = youtube_url(args.url or input('유튜브 영상 링크: ').strip())
            elif not args.file.is_file():
                raise ValueError('로컬 파일을 찾을 수 없습니다.')
        folder = args.output_dir.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        folder.mkdir(parents=True, exist_ok=False)
        metadata = {'created_at': datetime.now(timezone.utc).isoformat()}
        if args.response:
            payload = json.loads(args.response.read_text(encoding='utf-8'))
            previous = args.response.parent / 'metadata.json'
            if previous.exists():
                metadata.update(json.loads(previous.read_text(encoding='utf-8')))
            metadata['reexported_at'] = datetime.now(timezone.utc).isoformat()
        else:
            if args.file:
                source = args.file.resolve()
                metadata['source_file'] = source.name
            else:
                source, info = download_audio(source_url, folder, args.max_minutes)
                metadata.update(info)
            audio = prepare_audio(source, folder, args.max_minutes)
            model = os.getenv('DEEPGRAM_MODEL', 'nova-3')
            language = os.getenv('DEEPGRAM_LANGUAGE', 'ko')
            metadata.update(model=model, language=language, diarize_model='latest')
            print('[2/3] Deepgram 전사·화자 분리 중... (음성이 Deepgram으로 전송됩니다)', flush=True)
            payload = transcribe(audio, key, model, language)
        rows = save_outputs(folder, payload, metadata, names)
        if not args.keep_audio:
            for path in [folder / 'audio.wav', *folder.glob('source.*')]:
                path.unlink(missing_ok=True)
        print(f'[3/3] 완료: {len(rows)}개 발화\n저장 위치: {folder}')
        if any(r['speaker_id'] == 'UNKNOWN' for r in rows):
            print('일부 발화의 화자가 미확인입니다. 원본과 대조하세요.')
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        print(f'오류: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
