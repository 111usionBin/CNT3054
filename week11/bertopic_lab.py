"""CNT3054 week11: BERTopic 실행, 원문 검수, 결과물 ZIP 생성."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_config(path):
    p = Path(path).resolve()
    cfg = json.loads(p.read_text(encoding='utf-8-sig'))
    for k in ('input', 'text_column', 'model', 'output'):
        if not cfg.get(k):
            raise ValueError(f'config.json의 {k} 값을 확인하세요.')
    return cfg, p.parent


def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_input(cfg, root):
    import pandas as pd
    p = root / cfg['input']
    if not p.is_file():
        raise ValueError(f'입력 파일이 없습니다: {p}')
    try:
        df = pd.read_csv(p, encoding=cfg.get('encoding', 'utf-8-sig'))
    except UnicodeDecodeError as exc:
        raise ValueError('CSV를 UTF-8로 저장하거나 encoding을 cp949로 바꾸세요.') from exc
    return df, p


def prepare(df, cfg):
    import pandas as pd
    col = cfg['text_column']
    if col not in df:
        raise ValueError(f'본문 열 {col!r}이 없습니다. 실제 열: {list(df.columns)}')
    result = pd.DataFrame({'source_row': range(1, len(df) + 1),
                           'text': df[col].fillna('').astype(str)})
    result['text'] = result['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    reasons = pd.Series('', index=result.index)
    reasons[result.text.eq('')] = '빈 본문'
    if cfg.get('remove_duplicates', True):
        reasons[(reasons == '') & result.text.duplicated()] = '중복 본문'
    excluded = result.loc[reasons != ''].copy()
    excluded['reason'] = reasons[reasons != '']
    return result.loc[reasons == ''].reset_index(drop=True), excluded


def inspect(cfg, root):
    df, p = read_input(cfg, root)
    print(f'파일: {p.name}\n행 수: {len(df):,}\n열 이름: {list(df.columns)}')
    print(df.head(3).to_string(index=False))
    if cfg['text_column'] in df:
        clean, excluded = prepare(df, cfg)
        print(f'\n본문 열: {cfg["text_column"]} / 분석 대상: {len(clean):,} / 제외: {len(excluded):,}')
    else:
        print('\nconfig.json의 text_column을 본문 열 이름으로 수정하세요.')


def choose_samples(docs, emb, labels, keywords, seed):
    import numpy as np
    rows = []
    rng = np.random.default_rng(seed)
    for cid in sorted(set(labels)):
        ix = np.flatnonzero(labels == cid)
        selected = []
        if cid == -1:
            selected = [('미배정', int(i), None) for i in rng.permutation(ix)[:5]]
        else:
            center = emb[ix].mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            sim = emb[ix] @ center
            order = np.argsort(-sim)
            picks = [(int(i), '중심') for i in order[:3]]
            used = {i for i, _ in picks}
            for i in order[::-1]:
                if int(i) not in used:
                    picks.append((int(i), '주변'));used.add(int(i));break
            remaining = [i for i in range(len(ix)) if i not in used]
            if remaining:
                picks.append((int(rng.choice(remaining)), '무작위'))
            selected = [(role, int(ix[i]), float(sim[i])) for i, role in picks]
        for role, i, score in selected:
            rows.append({'cluster': int(cid), 'sample_type': role,
                         'source_row': int(docs.iloc[i].source_row),
                         'cosine_to_center': score,
                         'text': docs.iloc[i].text})
    return rows


def run(cfg, root):
    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from kiwipiepy import Kiwi
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer

    df, input_path = read_input(cfg, root)
    docs, excluded = prepare(df, cfg)
    n = len(docs)
    minimum = int(cfg.get('min_cluster_size', 15))
    density = int(cfg.get('min_samples', 5))
    if n < 30:
        raise ValueError(f'분석 대상이 {n}건입니다. 본문 열을 확인하고 최소 30건을 준비하세요.')
    if not 2 <= minimum < n or not 1 <= density < n:
        raise ValueError('min_cluster_size는 2 이상, min_samples는 1 이상이며 문서 수보다 작아야 합니다.')
    seed = int(cfg.get('seed', 42))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f'[1/4] 원본 {len(df):,}건 / 분석 {n:,}건 / 제외 {len(excluded):,}건', flush=True)
    model_id = cfg['model']
    print(f'[2/4] 임베딩 모델: {model_id}', flush=True)
    encoder = SentenceTransformer(model_id, device=cfg.get('device', 'cpu'),
                                  cache_folder=str(root / '.cache/models'), trust_remote_code=False)
    texts = docs.text.tolist()
    lengths = [len(encoder.tokenizer.encode(t, truncation=False)) for t in texts]
    long_count = sum(x > encoder.max_seq_length for x in lengths)
    if long_count:
        raise ValueError(f'{long_count}건이 모델 입력 한도({encoder.max_seq_length}토큰)를 넘습니다. '
                         '긴 글을 문장/문단으로 나누어 CSV를 다시 준비하세요. 자동으로 잘라 분석하지 않습니다.')
    revision = getattr(encoder[0].auto_model.config, '_commit_hash', None)
    key = hashlib.sha256(json.dumps([model_id, revision, texts], ensure_ascii=False).encode()).hexdigest()
    cache = root / '.cache';cache.mkdir(exist_ok=True)
    cache_path = cache / f'{key}.npy'
    if cache_path.exists():
        emb = np.load(cache_path, allow_pickle=False)
        print('      같은 모델과 입력의 임베딩을 재사용합니다.', flush=True)
    else:
        emb = encoder.encode(texts, batch_size=int(cfg.get('batch_size', 16)),
                             normalize_embeddings=True, show_progress_bar=True,
                             convert_to_numpy=True)
        np.save(cache_path, emb)
    if emb.shape[0] != n:
        raise ValueError('임베딩 행 수가 입력과 다릅니다. .cache의 npy 파일을 제거하고 재실행하세요.')
    kiwi = Kiwi(num_workers=2)
    stop_path = root / 'stopwords.txt'
    stop = {x.strip() for x in stop_path.read_text(encoding='utf-8').splitlines()
            if x.strip() and not x.startswith('#')} if stop_path.exists() else set()
    def tokenize(text):
        return [t.form for t in kiwi.tokenize(text)
                if t.tag in {'NNG', 'NNP', 'SL'} and len(t.form) > 1 and t.form not in stop]
    if not any(tokenize(t) for t in texts):
        raise ValueError('대표 단어로 사용할 어휘가 없습니다. 본문과 불용어를 확인하세요.')
    reducer = UMAP(n_neighbors=min(15, n-1), n_components=5, min_dist=0.0,
                   metric='cosine', random_state=seed, n_jobs=1)
    clusterer = HDBSCAN(min_cluster_size=minimum, min_samples=density,
                         metric='euclidean', cluster_selection_method='eom')
    model = BERTopic(embedding_model=encoder, umap_model=reducer,
                     hdbscan_model=clusterer,
                     vectorizer_model=CountVectorizer(tokenizer=tokenize, token_pattern=None,
                                                       lowercase=False, min_df=1),
                     ctfidf_model=ClassTfidfTransformer(),
                     calculate_probabilities=False, top_n_words=10, verbose=False)
    print('[3/4] UMAP / HDBSCAN / 대표 단어 추출', flush=True)
    labels, _ = model.fit_transform(texts, embeddings=emb)
    labels = np.asarray(labels, dtype=int)
    ids = sorted(set(labels))
    rows = []
    keywords = {}
    for cid in ids:
        words = model.get_topic(int(cid)) or []
        keywords[int(cid)] = ', '.join(w for w, _ in words[:10])
        count = int((labels == cid).sum())
        rows.append({'cluster': int(cid), 'count': count,
                     'percent_all': round(count / n * 100, 2), 'keywords': keywords[int(cid)]})
    summary = pd.DataFrame(rows)
    docs['cluster'] = labels
    docs['n_tokens'] = lengths
    samples = pd.DataFrame(choose_samples(docs, emb, labels, keywords, seed))
    out = root / cfg.get('output', 'outputs')
    # 이전 실행 및 수작업 검수 내용을 보존한다.
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out.name}에 이전 결과가 있습니다. config.json의 output을 outputs_v2 등 새 이름으로 바꾸세요.')
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('documents', docs), ('clusters', summary), ('review_samples', samples), ('excluded', excluded)]:
        frame.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    review = summary.loc[summary.cluster != -1, ['cluster', 'count', 'keywords']].copy()
    for column in ['topic_name', 'common_feature', 'evidence_rows', 'exceptions', 'interpretation']:
        review[column] = ''
    review.to_csv(out / 'cluster_review.csv', index=False, encoding='utf-8-sig')
    k = int(sum(cid != -1 for cid in ids))
    noise = int((labels == -1).sum())
    meta = {'created_at': datetime.now().isoformat(timespec='seconds'),
            'input_rows': len(df), 'analyzed_rows': n, 'excluded_rows': len(excluded),
            'n_clusters': k, 'unassigned_rows': noise, 'unassigned_percent': round(noise/n*100, 2),
            'embedding_dimension': int(emb.shape[1]), 'max_seq_length': encoder.max_seq_length,
            'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'embedding_input_sha256': key, 'config': cfg,
            'model_revision': getattr(encoder[0].auto_model.config, '_commit_hash', None),
            'python': sys.version.split()[0],
            'versions': {p: version(p) for p in ['torch', 'bertopic', 'sentence-transformers',
                         'transformers', 'umap-learn', 'hdbscan', 'pandas', 'numpy', 'kiwipiepy']}}
    save_json(out / 'run_summary.json', meta)
    save_json(out / 'config_used.json', cfg)
    input_suffix = input_path.suffix
    # 실제 분석한 CSV 원본을 고정해 ZIP에 같은 입력을 담는다.
    (out / f'input_snapshot{input_suffix}').write_bytes(input_path.read_bytes())
    (out / 'bertopic_lab_used.py').write_bytes(Path(__file__).read_bytes())
    for filename in ('requirements.txt', 'stopwords.txt'):
        (out / filename).write_bytes((root / filename).read_bytes())
    lines = ['# BERTopic 검수', '',
             f'분석 {n:,}건 · 클러스터 {k}개 · 미배정(-1) {noise:,}건 ({noise/n*100:.1f}%)', '',
             '클러스터 수에는 -1을 포함하지 않습니다. 비율의 분모는 분석한 전체 문서입니다.', '',
             '|클러스터|문서 수|비율(%)|대표 단어|', '|---|---:|---:|---|']
    for row in rows:
        lines.append(f'|{row["cluster"]}|{row["count"]}|{row["percent_all"]}|{row["keywords"]}|')
    for row in rows:
        cid = row['cluster']
        lines += ['', f'## 클러스터 {cid} ({row["count"]}건)', '', f'대표 단어: {row["keywords"]}', '']
        for item in samples.loc[samples.cluster == cid].itertuples():
            lines += [f'### {item.sample_type} · 원본 데이터 {item.source_row}행', '',
                      item.text.replace('<', '&lt;').replace('>', '&gt;'), '']
    (out / 'review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[4/4] 저장 완료: {out.name}/', flush=True)
    print(f'클러스터: {k}개 / 미배정(-1): {noise}건 ({noise/n*100:.1f}%)')
    print(summary.to_string(index=False))
    print(f'\n검수: {out.name}의 review.md를 열고 Ctrl+Shift+V를 누르세요.')


def pack(cfg, root, group):
    out = root / cfg['output']
    required = ['documents.csv', 'clusters.csv', 'review_samples.csv', 'excluded.csv',
                'cluster_review.csv', 'run_summary.json', 'config_used.json',
                'review.md', 'input_snapshot.csv', 'bertopic_lab_used.py',
                'requirements.txt', 'stopwords.txt']
    for name in required:
        if not (out / name).is_file():
            raise ValueError(f'{name}이 없습니다. run을 완료한 결과 폴더를 지정하세요.')
    meta = json.loads((out / 'run_summary.json').read_text(encoding='utf-8'))
    if hashlib.sha256((out / 'input_snapshot.csv').read_bytes()).hexdigest() != meta['input_sha256']:
        raise ValueError('입력 스냅샷이 바뀌었습니다. 분석에 사용한 원본을 복구하세요.')
    safe = re.sub(r'[^\w가-힣-]', '_', group).strip('_') or 'team'
    target = root / f'week11_{safe}_{datetime.now():%Y%m%d_%H%M%S}.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in required:
            z.write(out / name, f'results/{name}')
        for name in ['README.md']:
            if (root / name).is_file():z.write(root / name, name)
    print(f'결과물 ZIP: {target.name}\n결과물 drop-box에 제출하세요. 보고서는 보고서 drop-box에 따로 제출합니다.')


def main():
    p = argparse.ArgumentParser(description='BERTopic 실행과 원문 검수')
    p.add_argument('command', choices=['inspect', 'run', 'pack'])
    p.add_argument('--config', default=str(BASE / 'config.json'))
    p.add_argument('--group', default='team', help='ZIP 파일에 표시할 조 이름')
    a = p.parse_args()
    try:
        cfg, root = load_config(a.config)
        if a.command == 'inspect':inspect(cfg, root)
        elif a.command == 'run':run(cfg, root)
        else:pack(cfg, root, a.group)
    except Exception as exc:
        print(f'\n확인할 내용: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
