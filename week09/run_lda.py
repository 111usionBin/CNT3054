"""CNT3054 Week 9: CSV -> counts -> LDA -> CSV and offline HTML.
No API calls. Only the selected text column is used for fitting.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent


def read_documents(path, column):
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False, skip_blank_lines=False)
    if column not in frame:
        raise ValueError(f"'{column}' 열이 없습니다. 사용 가능한 열: {', '.join(frame.columns)}")
    raw_text = frame[column].copy()
    frame['source_row'] = np.arange(2, len(frame) + 2)
    frame['document_id'] = ['D%04d' % n for n in range(1, len(frame) + 1)]
    frame['original_text'] = raw_text
    frame['text'] = raw_text.map(lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s)).strip())
    empty = frame.text.eq('')
    duplicate = frame.text.duplicated(keep='first') & ~empty
    excluded = frame.loc[empty | duplicate, ['document_id', 'source_row', 'original_text']].copy()
    excluded['reason'] = np.where(empty[empty | duplicate], 'empty_text', 'duplicate_text')
    clean = frame.loc[~(empty | duplicate), ['document_id', 'source_row', 'original_text', 'text']].copy()
    if len(clean) < 5:
        raise ValueError('서로 다른 비어 있지 않은 문서가 최소 5개 필요합니다.')
    return clean.reset_index(drop=True), excluded


def make_tokenizer(kind, stopwords):
    if kind == 'kiwi':
        from kiwipiepy import Kiwi
        kiwi = Kiwi(num_workers=1)
        def tokenize(text):
            return [t.form for t in kiwi.tokenize(text) if t.tag in {'NNG','NNP','SL'}
                    and len(t.form) >= 2 and t.form.lower() not in stopwords]
    else:
        def tokenize(text):
            return [t for t in re.findall(r'[가-힣a-zA-Z]{2,}', text.lower()) if t not in stopwords]
    return tokenize


def prepare(path, column, tokenizer, stopwords_path, min_df, max_df):
    clean, excluded = read_documents(path, column)
    stopwords = {x.strip().lower() for x in stopwords_path.read_text(encoding='utf-8-sig').splitlines()
                 if x.strip() and not x.lstrip().startswith('#')}
    tokenize = make_tokenizer(tokenizer, stopwords)
    clean['tokens'] = clean.text.map(tokenize)
    # Token strings preserve the exact Korean preprocessing output for inspection.
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                 min_df=min_df, max_df=max_df)
    try:
        counts = vectorizer.fit_transform(clean.tokens.map(' '.join))
    except ValueError as e:
        raise ValueError('남은 어휘가 없습니다. 전처리와 불용어를 확인하거나 --min-df 1 --max-df 1.0으로 재실행하세요.') from e
    nonempty = np.asarray(counts.sum(axis=1)).ravel() > 0
    missing = clean.loc[~nonempty, ['document_id','source_row','original_text']].copy()
    missing['reason'] = 'no_retained_terms'
    excluded = pd.concat([excluded, missing], ignore_index=True)
    clean = clean.loc[nonempty].reset_index(drop=True)
    counts = counts[nonempty]
    if len(clean) < 5:
        raise ValueError('전처리 후 유효 문서가 5개 미만입니다. 문서 길이와 불용어를 확인하세요.')
    return clean, excluded, counts, vectorizer.get_feature_names_out(), stopwords


def fit_topics(counts, words, documents, k, seed, alpha, eta, max_iter, top_n=10):
    if not 2 <= k <= min(counts.shape):
        raise ValueError(f'토픽 수는 2 이상, 유효 문서 수·어휘 수 중 작은 값({min(counts.shape)}) 이하여야 합니다.')
    model = LatentDirichletAllocation(n_components=k, learning_method='batch',
              random_state=seed, doc_topic_prior=alpha, topic_word_prior=eta,
              max_iter=max_iter, n_jobs=1, evaluate_every=-1)
    theta = model.fit_transform(counts)
    phi = model.components_ / model.components_.sum(axis=1, keepdims=True)
    assert np.allclose(theta.sum(axis=1), 1) and np.allclose(phi.sum(axis=1), 1)
    names = [f'topic_{i+1}' for i in range(k)]
    document_topics = documents[['document_id','source_row','original_text']].copy()
    for i, name in enumerate(names):
        document_topics[name] = theta[:, i]
    document_topics['dominant_topic'] = [names[i] for i in theta.argmax(axis=1)]
    rows, representatives, top_terms = [], [], []
    for i, name in enumerate(names):
        indexes = np.argsort(-phi[i], kind='stable')[:top_n]
        top_terms.extend(words[indexes])
        for rank, index in enumerate(indexes, 1):
            rows.append(dict(topic=name, rank=rank, word=words[index], probability=phi[i,index]))
        for rank, index in enumerate(np.argsort(-theta[:, i], kind='stable')[:3], 1):
            representatives.append(dict(topic=name, rank=rank, document_id=documents.iloc[index].document_id,
                         proportion=theta[index,i], text=documents.iloc[index].original_text))
    summary = pd.DataFrame({'topic':names, 'mean_document_proportion':theta.mean(axis=0),
                            'dominant_document_count':[(theta.argmax(axis=1)==i).sum() for i in range(k)]})
    metrics = dict(k=k, training_perplexity=float(model.perplexity(counts)),
                   top_word_diversity=len(set(top_terms))/len(top_terms),
                   iterations=int(model.n_iter_))
    return document_topics, pd.DataFrame(rows), pd.DataFrame(representatives), summary, phi, metrics


STYLE = '''body{font-family:Arial,"Noto Sans CJK KR","Malgun Gothic",sans-serif;color:#253b50;margin:40px auto;max-width:1160px;padding:0 24px;line-height:1.6}h1{font-size:30px;font-weight:500}h2{font-size:24px;font-weight:500;margin-top:36px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #d5dde3;padding:10px;text-align:left;vertical-align:top}th{background:#e8eef2;font-weight:500}.note{color:#596673}.bar{background:#426580;color:white;min-width:48px;padding:3px 8px;box-sizing:border-box}a{color:#205f98}nav a{margin-right:18px}'''


def page(title, body):
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style><body><h1>'+html.escape(title)+'</h1>'+body+'</body></html>'


def table(frame):
    # Escape user content and formula-looking strings are not executed by HTML.
    return frame.to_html(index=False, escape=True, float_format=lambda n:f'{n:.4f}')


def safe_csv(frame, path):
    # Protect spreadsheet users from formulas embedded in arbitrary input text.
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object','string']).columns:
        safe[col] = safe[col].map(lambda x: "'"+x if isinstance(x,str) and x.lstrip().startswith(('=','+','-','@')) else x)
    safe.to_csv(path, index=False, encoding='utf-8-sig')


def save_model(folder, documents, words, result):
    doc, topics, representatives, summary, phi, metrics = result
    folder.mkdir(parents=True, exist_ok=False)
    for name, frame in [('document_topics',doc),('topics',topics),('representative_docs',representatives),('topic_summary',summary)]:
        safe_csv(frame, folder/f'{name}.csv')
    full = pd.DataFrame(phi.T, columns=[f'topic_{i+1}' for i in range(phi.shape[0])])
    full.insert(0,'word',words)
    safe_csv(full, folder/'topic_word_distribution.csv')
    preview=documents[['document_id','original_text','tokens']].copy()
    preview['tokens']=preview.tokens.map(' '.join)
    safe_csv(preview,folder/'tokens_preview.csv')
    bars=''.join('<p>'+html.escape(r.topic)+f' · 문서 평균 {r.mean_document_proportion:.1%}</p><div class="bar" style="width:{100*r.mean_document_proportion:.1f}%">'+f'{r.mean_document_proportion:.1%}</div>' for r in summary.itertuples())
    body='<p class="note">모든 토픽 이름은 임시 번호입니다. 상위 단어와 대표 문서를 보고 이름을 붙이세요. 토픽 비율은 감정 점수가 아닙니다.</p><nav><a href="#summary">비중</a><a href="#words">상위 단어</a><a href="#docs">대표 문서</a><a href="#theta">문서별 비율</a><a href="#tokens">전처리</a></nav>'
    body+='<h2 id="summary">토픽 비중: 문서별 비율의 단순 평균</h2>'+bars
    compact=topics.groupby('topic',sort=False).apply(lambda x:', '.join(x.word),include_groups=False).reset_index(name='상위 단어')
    body+='<h2 id="words">토픽별 상위 단어</h2>'+table(compact)
    body+='<h2 id="docs">토픽별 대표 문서 3개</h2>'+table(representatives)
    body+='<h2 id="theta">문서별 토픽 비율: 앞 20개</h2>'+table(doc.head(20))
    body+='<h2 id="tokens">전처리 확인: 앞 20개</h2>'+table(preview.head(20))
    body+='<p>전체 행과 전체 어휘 분포는 같은 폴더의 CSV에서 확인하세요. CSV에서 수식으로 해석될 수 있는 문자열 앞에는 안전을 위해 작은따옴표를 붙였습니다.</p>'
    (folder/'report.html').write_text(page(f'LDA 결과 · K={metrics["k"]}',body),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/culture_reviews.csv')
    parser.add_argument('--text-column',default='text')
    parser.add_argument('--topics',type=int,default=4)
    parser.add_argument('--compare',type=int,nargs='+',help='같은 전처리 데이터로 여러 K 비교')
    parser.add_argument('--tokenizer',choices=['kiwi','regex'],default='kiwi')
    parser.add_argument('--stopwords',type=Path,default=ROOT/'data/stopwords.txt')
    parser.add_argument('--min-df',type=int,default=2)
    parser.add_argument('--max-df',type=float,default=.95)
    parser.add_argument('--alpha',type=float,default=.1)
    parser.add_argument('--eta',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-iter',type=int,default=30)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs')
    args=parser.parse_args(argv)
    try:
        if args.min_df<1 or not 0<args.max_df<=1 or args.alpha<=0 or args.eta<=0 or args.max_iter<1:
            raise ValueError('min-df·max-iter는 1 이상, alpha·eta는 양수, max-df는 0 초과 1 이하여야 합니다.')
        docs, excluded, counts, words, stopwords=prepare(args.input,args.text_column,args.tokenizer,args.stopwords,args.min_df,args.max_df)
        ks=list(dict.fromkeys(args.compare or [args.topics]))
        if any(not 2<=k<=min(counts.shape) for k in ks):
            raise ValueError(f'K는 2 이상 {min(counts.shape)} 이하여야 합니다.')
        print(f'유효 문서: {len(docs)} | 제외 문서: {len(excluded)} | 어휘: {len(words)}')
        print(f'입력: 정수 빈도 행렬 {counts.shape} | 추론: batch 변분추론')
        folder=args.output_root/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
        folder.mkdir(parents=True,exist_ok=False)
        safe_csv(excluded,folder/'excluded_documents.csv')
        metrics=[]
        for k in ks:
            result=fit_topics(counts,words,docs,k,args.seed,args.alpha,args.eta,args.max_iter)
            save_model(folder/f'k{k}',docs,words,result)
            metrics.append(result[-1]);print(f'K={k}: 저장 완료')
        comparison=pd.DataFrame(metrics)
        safe_csv(comparison,folder/'comparison.csv')
        links=''.join(f'<li><a href="k{k}/report.html">K={k} 결과 열기</a></li>' for k in ks)
        body=f'<p>유효 문서 {len(docs)}개, 어휘 {len(words)}개, 제외 문서 {len(excluded)}개</p><ul>{links}</ul>'
        body+='<h2>설정 비교</h2>'+table(comparison)+'<p>training_perplexity는 학습 자료 자체에서 계산한 참고값이며 일반화 성능이나 최적 K를 뜻하지 않습니다. top_word_diversity는 상위 단어의 중복 정도만 나타내며 의미적 일관성 지표가 아닙니다. 각 결과의 대표 문서를 읽고 토픽의 구분과 중복을 비교하세요. K가 다른 실행의 topic_1은 서로 다른 주제일 수 있습니다.</p>'
        (folder/'index.html').write_text(page('9주차 LDA 실습 결과',body),encoding='utf-8')
        metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'stopwords':sorted(stopwords),'valid_documents':len(docs),'excluded_documents':len(excluded),'vocabulary_size':len(words),'topic_counts':ks,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']},'python':sys.version,'algorithm':'sklearn LDA, batch variational inference','comparison_scope':'training corpus only; no held-out evaluation'}
        if args.tokenizer=='kiwi':metadata['versions']['kiwipiepy']=importlib.metadata.version('kiwipiepy')
        (folder/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'결과: {folder.resolve()}')
        print(f'브라우저로 열기: {(folder/"index.html").resolve()}')
        return folder
    except (ValueError,OSError,ImportError) as e:
        parser.exit(2,f'실행 실패: {e}\n')

if __name__=='__main__':main()
