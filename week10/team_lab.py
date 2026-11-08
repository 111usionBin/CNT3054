"""CNT3054: 조별 CSV에서 토픽을 찾고 해당 원문을 읽습니다."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from run_lab import tokenize, write_csv, write_json, aggregate_time, set_font

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

def read_config(path):
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for key in ('input', 'out'):
        cfg[key] = REPO / cfg[key]
    return cfg

def read_source(cfg):
    return pd.read_csv(cfg['input'], encoding=cfg['encoding'], dtype=str, keep_default_na=False)

def prepare(cfg):
    raw = read_source(cfg)
    text_col = cfg['text_column']
    if text_col not in raw:
        raise ValueError(f'text_column을 실제 열 이름으로 바꿔 주세요. 현재 열: {list(raw.columns)}')
    data = pd.DataFrame({'document_id': [f'R{i+2:06d}' for i in range(len(raw))],
                         'source_row': np.arange(2, len(raw)+2), 'text': raw[text_col].str.strip()})
    blank = data.text.eq('')
    data = data.loc[~blank].copy()
    date_col = cfg.get('date_column')
    if date_col:
        if date_col not in raw: raise ValueError('date_column에 지정한 열이 없습니다.')
        dates = pd.to_datetime(raw.loc[data.index, date_col], format='mixed', errors='coerce')
        if dates.isna().any():
            raise ValueError('날짜를 읽지 못한 행이 있습니다. 날짜를 확인하거나 date_column을 null로 두고 원문 해석부터 진행하세요.')
        try: data['period'] = dates.dt.strftime('%Y-%m')
        except AttributeError as e: raise ValueError('날짜 열의 시간대 표기를 통일해 주세요.') from e
    else: data['period'] = ''
    if len(data) < 8: raise ValueError('빈 내용을 제외한 문서가 8개 미만입니다. 분석할 자료를 더 모아 주세요.')
    return data.reset_index(drop=True), int(blank.sum())

def run(cfg):
    data, blank = prepare(cfg)
    k = cfg['topics']
    if not isinstance(k, int) or isinstance(k, bool) or not 2 <= k <= 12:
        raise ValueError('topics에는 2부터 12 사이 정수를 넣어 주세요.')
    out = cfg['out']; out.mkdir(parents=True, exist_ok=True)
    tokens = tokenize(data.text)
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=2, max_df=.95)
    counts = vectorizer.fit_transform([' '.join(t) for t in tokens])
    keep = np.asarray(counts.sum(axis=1)).ravel() > 0
    write_csv(data.loc[~keep], out/'excluded_documents.csv')
    data = data.loc[keep].reset_index(drop=True); counts = counts[keep]
    if min(counts.shape) < k: raise ValueError('남은 문서나 어휘보다 토픽 수가 많습니다. topics를 줄이거나 자료를 확인하세요.')
    model = LatentDirichletAllocation(n_components=k, random_state=42, learning_method='batch',
                                     max_iter=50, doc_topic_prior=.1, topic_word_prior=.1, n_jobs=1)
    theta = model.fit_transform(counts)
    names = [f'T{i+1}' for i in range(k)]
    for i, name in enumerate(names): data[name] = theta[:, i]
    data['dominant_topic'] = [names[i] for i in theta.argmax(axis=1)]
    write_csv(data, out/'document_topics.csv')
    preview = data[['document_id', 'source_row', 'text']].copy()
    preview['tokens'] = [' '.join(t) for t, use in zip(tokens, keep) if use]
    write_csv(preview, out/'tokens_preview.csv')
    words = vectorizer.get_feature_names_out()
    topic_lines = ['# 우리 조 데이터의 토픽', '']
    evidence = []
    for i, name in enumerate(names):
        top = words[np.argsort(-model.components_[i])[:10]]
        topic_lines += [f'## {name}', ', '.join(top), '']
        print(name + ': ' + ', '.join(top))
        ranked = data.sort_values([name, 'document_id'], ascending=[False, True])
        leading = ranked.head(5)
        values = np.sort(data[names].to_numpy(), axis=1)
        top_two = np.argsort(-data[names].to_numpy(), axis=1)[:, :2]
        candidates = data.loc[np.any(top_two == i, axis=1)].copy()
        candidates['gap'] = (values[:, -1] - values[:, -2])[candidates.index]
        mixed = candidates.loc[~candidates.document_id.isin(leading.document_id)].sort_values(['gap','document_id']).head(2)
        lines = [f'# {name} 원문', ', '.join(top), '']
        for label, part in [('토픽 비중이 높은 글', leading), ('두 토픽이 섞인 후보', mixed)]:
            for row in part.itertuples():
                lines += [f'## {label} / {row.document_id}',
                          f'CSV 행: {row.source_row} / 비중: {getattr(row, name):.3f}', row.text, '']
                evidence.append({'topic':name,'kind':label,'document_id':row.document_id,
                                 'source_row':row.source_row,'text':row.text})
        (out/f'{name}_read.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'topics.md').write_text('\n'.join(topic_lines), encoding='utf-8')
    write_csv(pd.DataFrame(evidence), out/'evidence.csv')
    write_json(out/'metadata.json', {'input':str(cfg['input']), 'sha256':hashlib.sha256(cfg['input'].read_bytes()).hexdigest(),
               'text_column':cfg['text_column'],'date_column':cfg.get('date_column'), 'topics':k,
               'blank_removed':blank,'no_tokens_removed':int((~keep).sum()),'documents':len(data),
               'vocabulary':len(words),'seed':42})
    for old in ['time_summary.csv','time_comparison.png']:
        (out/old).unlink(missing_ok=True)
    print(f'분석한 문서: {len(data)}개 / 빈 내용 제외: {blank}개 / 어휘 없음 제외: {int((~keep).sum())}개')
    print('열기:', out/'topics.md')
    print('이어서 T1_read.md 등 관심 있는 토픽의 원문 파일을 여세요.')

def compare_time(cfg):
    out = cfg['out']
    meta = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    if not meta['date_column']:
        print('날짜를 사용하지 않은 분석입니다. 토픽별 원문을 읽는 단계까지 진행하세요.'); return
    data = pd.read_csv(out/'document_topics.csv', encoding='utf-8-sig', dtype={'period':str})
    periods = sorted(data.period.unique())
    if len(periods) < 2:
        print('자료가 한 달에 모여 있어 월별 변화를 비교하지 않습니다. 토픽별 원문을 읽어 주세요.'); return
    names = [f'T{i+1}' for i in range(meta['topics'])]
    result = aggregate_time(data, names); write_csv(result, out/'time_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    set_font()
    fig, ax = plt.subplots(figsize=(max(9,len(periods)*.65),5), layout='constrained')
    for name in names:
        part=result.loc[result.topic.eq(name)]
        ax.plot(part.period, part.mean_proportion*100, marker='o', label=name)
    ax.set_ylabel('문서별 토픽 비중의 평균 (%)'); ax.set_xlabel('월'); ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(periods)),[f'{p}\n{sum(data.period.eq(p))}개' for p in periods],rotation=45 if len(periods)>8 else 0)
    ax.legend(ncol=min(6,len(names))); ax.grid(alpha=.2)
    fig.savefig(out/'time_comparison.png',dpi=180); plt.close(fig)
    print(result.to_string(index=False, float_format=lambda x:f'{x:.3f}'))
    print('열기:',out/'time_comparison.png')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=['inspect','run','time'])
    p.add_argument('--config',type=Path,default=ROOT/'team_config.json')
    args=p.parse_args()
    try:
        cfg=read_config(args.config)
        if args.step=='inspect':
            raw=read_source(cfg)
            print('현재 CSV:',cfg['input']);print('행 수:',len(raw));print('열 이름:')
            for col in raw.columns: print(' -',col)
            print('team_config.json의 text_column에 분석할 본문 열 이름을 넣고 저장하세요.')
        elif args.step=='run':run(cfg)
        else:compare_time(cfg)
    except (ValueError,KeyError,OSError,ImportError) as exc:
        print('확인할 내용:',exc,file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
