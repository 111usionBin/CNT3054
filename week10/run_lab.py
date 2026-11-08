"""CNT3054 Week 10: 원문 검토, 토픽 해석, 시기 비교, 근거 검사.
실행 위치: CNT3054. 결과는 week10/outputs, 직접 작성할 파일은 week10/work.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unicodedata
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

ROOT=Path(__file__).resolve().parent
TOPIC_COUNT=4
SEED=42

def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def write_csv(frame,path):
    safe=frame.copy()
    for col in safe.select_dtypes(include=['object','string']):
        safe[col]=safe[col].map(lambda s: "'"+s if isinstance(s,str) and s.lstrip().startswith(('=','+','-','@')) else s)
    safe.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.6f')

def read_data(path):
    df=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    required={'document_id','date','text'}
    if not required.issubset(df):raise ValueError('필수 열: document_id,date,text')
    for col in required:df[col]=df[col].str.strip()
    if df[list(required)].eq('').any().any():raise ValueError('ID·날짜·원문에 빈 값이 있습니다.')
    if df.document_id.duplicated().any():raise ValueError('document_id가 중복됩니다.')
    dates=pd.to_datetime(df.date,format='%Y-%m-%d',errors='coerce')
    if dates.isna().any():raise ValueError('날짜 형식은 YYYY-MM-DD입니다.')
    if len(df)<8:raise ValueError('유효 문서가 최소 8개 필요합니다.')
    df['period']=dates.dt.strftime('%Y-%m')
    return df

def tokenize(texts):
    from kiwipiepy import Kiwi
    kiwi=Kiwi(num_workers=1)
    stop=set((ROOT/'data/stopwords.txt').read_text(encoding='utf-8').split())
    return [[t.form for t in kiwi.tokenize(unicodedata.normalize('NFKC',text))
             if t.tag in {'NNG','NNP','SL'} and len(t.form)>=2 and t.form not in stop] for text in texts]

def fit(args):
    df=read_data(args.input)
    tokens=tokenize(df.text)
    vectorizer=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,min_df=2,max_df=.95)
    counts=vectorizer.fit_transform([' '.join(t) for t in tokens])
    keep=np.asarray(counts.sum(axis=1)).ravel()>0
    write_csv(df.loc[~keep],args.out/'excluded_documents.csv')
    df=df.loc[keep].reset_index(drop=True); counts=counts[keep]
    if min(counts.shape)<TOPIC_COUNT:raise ValueError('전처리 뒤 문서 또는 어휘가 부족합니다.')
    # 전체 기간에서 한 모델을 학습합니다. 월별 모델을 따로 만들지 않습니다.
    lda=LatentDirichletAllocation(n_components=TOPIC_COUNT,random_state=SEED,
        learning_method='batch',max_iter=50,doc_topic_prior=.1,topic_word_prior=.1,n_jobs=1)
    theta=lda.fit_transform(counts)
    phi=lda.components_/lda.components_.sum(axis=1,keepdims=True)
    words=vectorizer.get_feature_names_out()
    names=[f'T{i+1}' for i in range(TOPIC_COUNT)]
    for i,t in enumerate(names):df[t]=theta[:,i]
    df['dominant_topic']=[names[i] for i in theta.argmax(axis=1)]
    write_csv(df,args.out/'document_topics.csv')
    preview=df[['document_id','text']].copy()
    preview['tokens']=[' '.join(t) for t,k in zip(tokens,keep) if k]
    write_csv(preview,args.out/'tokens_preview.csv')
    rows=[]; summary=['# 01 토픽의 상위 단어\n','전체 기간을 함께 학습한 K=4 모델입니다.\n']
    for i,t in enumerate(names):
        ix=np.argsort(-phi[i],kind='stable')[:10]
        for rank,j in enumerate(ix,1):rows.append({'topic':t,'rank':rank,'word':words[j],'probability':phi[i,j]})
        summary.extend([f'## {t}',', '.join(words[ix]),''])
        print(t+': '+', '.join(words[ix[:8]]))
    write_csv(pd.DataFrame(rows),args.out/'topics.csv')
    (args.out/'01_topics.md').write_text('\n'.join(summary),encoding='utf-8')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','kiwipiepy','matplotlib']}
    write_json(args.out/'run_metadata.json',{'input':str(args.input),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'stopwords_sha256':hashlib.sha256((ROOT/'data/stopwords.txt').read_bytes()).hexdigest(),'n_documents':len(df),'excluded':int((~keep).sum()),
        'vocabulary':len(words),'k':TOPIC_COUNT,'seed':SEED,'alpha':.1,'eta':.1,'max_iter':50,'versions':versions,
        'period_order':sorted(df.period.unique()),'aggregation':'equal document weights','python':sys.version})
    print(f'학습 완료: {len(df)}개 문서 / {len(words)}개 어휘 / 토픽 4개')
    print('다음: python week10/run_lab.py evidence')

def load_results(out):
    path=out/'document_topics.csv'
    if not path.exists():raise ValueError('먼저 fit 명령을 실행하세요.')
    df=pd.read_csv(path,dtype={'document_id':str,'period':str},encoding='utf-8-sig')
    names=[f'T{i+1}' for i in range(TOPIC_COUNT)]
    if not np.allclose(df[names].sum(axis=1),1,atol=1e-5):raise ValueError('문서별 토픽 비중 합이 1과 다릅니다. fit을 다시 실행하세요.')
    return df,names

def evidence(args):
    df,names=load_results(args.out); rng=np.random.default_rng(SEED)
    records=[]; lines=['# 02 단어에서 원문으로\n','대표 문서, 혼합 문서, 무작위 문서를 비교합니다.\n']
    for t in names:
        ordered=df.sort_values([t,'document_id'],ascending=[False,True])
        representative=ordered.head(3)
        # 1위와 2위 토픽의 차이가 작은 문서 중, 현재 토픽이 상위 2위인 문서.
        ranking=np.argsort(-df[names].to_numpy(),axis=1)
        candidate=df.loc[np.any(ranking[:,:2]==names.index(t),axis=1)].copy()
        probs=np.sort(candidate[names].to_numpy(),axis=1)
        candidate['gap']=probs[:,-1]-probs[:,-2]
        boundary=candidate.loc[~candidate.document_id.isin(representative.document_id)].sort_values(['gap','document_id']).head(2)
        used=set(representative.document_id)|set(boundary.document_id)
        pool=df.loc[~df.document_id.isin(used)]
        random=pool.iloc[rng.choice(len(pool),min(2,len(pool)),replace=False)]
        lines.append(f'## {t}\n')
        for kind,part in [('대표',representative),('혼합',boundary),('무작위',random)]:
            for r in part.itertuples():
                rec={'topic':t,'sample_kind':kind,'document_id':r.document_id,'period':r.period,'proportion':getattr(r,t),'text':r.text}
                records.append(rec)
                lines.extend([f'### {kind} / {r.document_id} / {r.period} / {getattr(r,t):.3f}',f'> {r.text}',''])
    for t in names:
        section=['# '+t+' 원문 검토\n']
        for r in records:
            if r['topic']==t:section.extend([f"## {r['sample_kind']} / {r['document_id']} / {r['period']} / {r['proportion']:.3f}",'> '+r['text'],''])
        (args.out/f'02_{t}_evidence.md').write_text('\n'.join(section),encoding='utf-8')
    write_csv(pd.DataFrame(records),args.out/'evidence.csv')
    (args.out/'02_evidence.md').write_text('\n'.join(lines),encoding='utf-8')
    work=ROOT/'work';work.mkdir(exist_ok=True)
    target=work/'interpretations.json'
    if not target.exists():write_json(target,[{'topic':t,'label':'','evidence_ids':[],'counterexample_id':'','cultural_question':'','scope_note':''} for t in names])
    print('대표 3개 + 혼합 최대 2개 + 무작위 2개를 토픽별로 저장했습니다.')
    print('열기: week10/outputs/02_evidence.md')
    print('작성: week10/work/interpretations.json (기존 작성 내용 보존)')
    print('다음: python week10/run_lab.py review')

def review(args):
    df,names=load_results(args.out)
    path=ROOT/'work/interpretations.json'
    if not path.exists():raise ValueError('먼저 evidence 명령을 실행하세요.')
    items=json.loads(path.read_text(encoding='utf-8')); ids=set(df.document_id)
    if not isinstance(items,list) or len(items)!=len(names):raise ValueError('토픽 4개의 기록이 필요합니다.')
    if sorted(x.get('topic','') for x in items)!=names:raise ValueError('T1~T4를 각각 한 번 기록하세요.')
    lines=['# 03 우리 조의 해석 기록\n']; complete=True
    for item in items:
        missing=[k for k in ['label','evidence_ids','counterexample_id','cultural_question','scope_note'] if not item.get(k)]
        cites=item.get('evidence_ids',[])
        if not isinstance(cites,list):raise ValueError('evidence_ids는 ID 배열이어야 합니다.')
        unknown=(set(cites)|{item.get('counterexample_id','')})-ids-{''}
        state='작성 완료' if not missing and not unknown else '추가 작성'
        complete=complete and state=='작성 완료'
        print(f"{item['topic']}: {state}"+(f' / 미입력 {", ".join(missing)}' if missing else '')+(f' / 없는 ID {unknown}' if unknown else ''))
        lines.extend([f"## {item['topic']} {item.get('label','')}",f"근거 ID: {', '.join(cites)}",f"다른 해석을 확인할 문서: {item.get('counterexample_id','')}",f"문화적 질문: {item.get('cultural_question','')}",f"해석 범위: {item.get('scope_note','')}",''])
    (args.out/'03_interpretations.md').write_text('\n\n'.join(lines),encoding='utf-8')
    print('형식 확인 결과이며, 근거가 해석을 뒷받침하는지는 직접 읽고 판단합니다.')
    print('다음: python week10/run_lab.py time')
    return complete

def aggregate_time(df,names):
    rows=[]
    for period,part in df.groupby('period',sort=True):
        for t in names:rows.append({'period':period,'topic':t,'n_documents':len(part),'dominant_count':int(part.dominant_topic.eq(t).sum()),'mean_proportion':float(part[t].mean())})
    return pd.DataFrame(rows)

def set_font():
    import matplotlib as mpl
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    for name in ['Malgun Gothic','AppleGothic','Noto Sans CJK KR','NanumGothic']:
        if name in available:mpl.rcParams['font.family']=name;break
    mpl.rcParams['axes.unicode_minus']=False

def time_analysis(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    set_font()
    df,names=load_results(args.out)
    periods=sorted(df.period.unique())
    if len(periods)<2:raise ValueError('변화 비교에는 서로 다른 월의 문서가 필요합니다.')
    result=aggregate_time(df,names);write_csv(result,args.out/'time_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#16796B','#D27A39','#4C70A3','#9C629F'];x=np.arange(len(periods));width=.8/len(names)
    for ax,metric,title,ylabel in zip(axes,['dominant_count','mean_proportion'],['가장 큰 토픽으로 센 문서 수','문서별 토픽 비중의 평균'],['문서 수 (개)','평균 비중 (%)']):
        for i,t in enumerate(names):
            values=result[result.topic.eq(t)].set_index('period').loc[periods,metric].to_numpy()
            if metric=='mean_proportion':values=values*100
            ax.bar(x+(i-1.5)*width,values,width,label=t,color=colors[i])
        ax.set_xticks(x,[f'{p}\n전체 {sum(df.period.eq(p))}개' for p in periods]);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.legend(ncol=4);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(args.out/'04_time_comparison.png',dpi=160);plt.close(fig)
    lines=['# 04 시기별 변화\n','분모는 각 월의 유효 문서 수입니다. 같은 모델의 토픽을 비교했습니다.\n','|월|토픽|전체 문서|최다 토픽 문서 수|평균 비중|','|---|---|---:|---:|---:|']
    for r in result.itertuples():lines.append(f'|{r.period}|{r.topic}|{r.n_documents}|{r.dominant_count}|{r.mean_proportion:.1%}|')
    lines+=['','![시기 비교](04_time_comparison.png)','','## 관찰과 해석','숫자로 관찰한 변화 한 문장과 원문으로 설명한 변화 한 문장을 작성하세요.','같은 단어가 시기별로 다른 입장에 쓰였는지 evidence.csv의 period와 원문을 함께 읽으세요.']
    (args.out/'04_time_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(result.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('열기: week10/outputs/04_time_comparison.png')
    print('다음: python week10/run_lab.py harness')

def audit_response(packet,response):
    errors=[]; evidence={x['document_id']:x['text'] for x in packet['documents']}
    if not isinstance(response,dict) or not isinstance(response.get('claims'),list) or not response['claims']:
        return {'status':'REJECT','errors':['claims 배열이 필요합니다.'],'human_review':[]}
    for i,c in enumerate(response['claims'],1):
        if not isinstance(c,dict):errors.append(f'{i}: 주장 객체가 필요합니다.');continue
        if not isinstance(c.get('statement'),str) or not c['statement'].strip():errors.append(f'{i}: statement가 필요합니다.')
        if not isinstance(c.get('citations'),list) or not c['citations']:errors.append(f'{i}: 근거 인용이 필요합니다.');continue
        for cite in c['citations']:
            if not isinstance(cite,dict):errors.append(f'{i}: 인용 객체가 필요합니다.');continue
            doc_id=cite.get('document_id');quote=cite.get('quote')
            if doc_id not in evidence:errors.append(f'{i}: 입력에 없는 문서 {doc_id}')
            elif not isinstance(quote,str) or not quote.strip() or quote not in evidence[doc_id]:errors.append(f'{i}: {doc_id}의 원문과 인용 불일치')
    return {'status':'REJECT' if errors else 'HUMAN_REVIEW','errors':errors,
            'human_review':['인용이 주장을 실제로 뒷받침하는가?','반대 사례와 소수 경험이 빠졌는가?','주장 범위가 입력 자료보다 넓은가?']}

def harness(args):
    df,names=load_results(args.out)
    evpath=args.out/'evidence.csv'
    if not evpath.exists():raise ValueError('먼저 evidence 명령을 실행하세요.')
    ev=pd.read_csv(evpath,encoding='utf-8-sig')
    ids=ev.loc[ev.topic.eq(args.topic),'document_id'].drop_duplicates().tolist()
    documents=df.set_index('document_id').loc[ids].reset_index()[['document_id','period','text']].to_dict('records')
    packet={'question':'이 토픽은 어떤 경험과 가치 판단을 담는가?','topic':args.topic,'documents':documents,
        'sampling':'대표 3, 상위 2위에 들어가는 혼합 최대 2, 나머지 무작위 2','output_schema':{'claims':[{'statement':'해석 한 문장','citations':[{'document_id':'ID','quote':'원문 그대로'}]}]},
        'prompt':'제공한 문서만 근거로 이름 후보와 해석을 작성하세요. 각 주장에 문서 ID와 원문 인용을 연결하세요. 상반된 해석과 추가로 확인할 자료를 함께 제시하세요.',
        'execution':'offline packet; no model or API was called'}
    write_json(args.out/'05_harness_input.json',packet)
    prompt=['# 05 해석 요청 묶음\n',f'대상 토픽: {args.topic}',packet['prompt'],'']
    for d in documents:prompt.extend([f"## {d['document_id']} / {d['period']}",d['text'],''])
    prompt+=['## 검토할 항목','이름 후보, 주장, 문서 ID, 정확한 인용, 다른 해석, 추가로 확인할 자료']
    (args.out/'05_harness_prompt.md').write_text('\n'.join(prompt),encoding='utf-8')
    first=documents[0]
    valid={'authorship':'수업용 수동 작성 검사 입력','claims':[{'statement':f"{first['document_id']}은 다음 경험을 서술한다.",'citations':[{'document_id':first['document_id'],'quote':first['text']}]}]}
    invalid={'authorship':'수업용 수동 작성 오류 입력','claims':[{'statement':'모든 관객이 같은 의견이다.','citations':[{'document_id':first['document_id'],'quote':'모든 관객이 같은 의견이다.'}]}]}
    write_json(args.out/'response_valid.json',valid);write_json(args.out/'response_invalid.json',invalid)
    results={name:audit_response(packet,r) for name,r in [('valid',valid),('invalid',invalid)]}
    write_json(args.out/'05_audit_results.json',results)
    lines=['# 05 근거 검사 결과\n','이 단계는 수동 작성 응답으로 검사 규칙을 실행합니다.\n']
    for name,result in results.items():
        lines.extend([f"## {name}: {result['status']}"]+result['errors']+result['human_review']+[''])
        print(f"{name}: {result['status']}"+(' / '+ '; '.join(result['errors']) if result['errors'] else ' / 원문 의미 검토 대기'))
    (args.out/'05_audit.md').write_text('\n\n'.join(lines),encoding='utf-8')
    print('원문 인용이 맞아도 최종 해석은 HUMAN_REVIEW에서 검토합니다.')
    print('열기: week10/outputs/05_audit.md')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['check','fit','evidence','review','time','harness','audit'])
    p.add_argument('--input',type=Path,default=ROOT/'data/concert_comments.csv')
    p.add_argument('--out',type=Path,default=ROOT/'outputs')
    p.add_argument('--topic',choices=['T1','T2','T3','T4'],default='T1')
    p.add_argument('--response',type=Path)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    try:
        if a.step=='check':
            df=read_data(a.input);print('Python:',sys.executable);print(f'문서 {len(df)}개 / ID 중복 0 / 빈 값 0');print(df.groupby('period').size().to_string());print('다음: python week10/run_lab.py fit')
        elif a.step=='fit':fit(a)
        elif a.step=='evidence':evidence(a)
        elif a.step=='review':review(a)
        elif a.step=='time':time_analysis(a)
        elif a.step=='harness':harness(a)
        else:
            if not a.response:raise ValueError('audit에는 --response 파일이 필요합니다.')
            packet=json.loads((a.out/'05_harness_input.json').read_text(encoding='utf-8'))
            result=audit_response(packet,json.loads(a.response.read_text(encoding='utf-8')))
            write_json(a.out/'custom_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
            if result['status']=='REJECT':return 1
    except (ValueError,OSError,ImportError,KeyError) as e:
        print(f'확인 필요: {e}',file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
