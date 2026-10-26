"""Validate teaching CSVs without third-party packages or network access."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def validate(root=ROOT):
    specifications = [('train_reviews.csv', {'review_id','text','label','group_id','domain','source_type'}, 'label', {'0','1'}), ('context_reviews.csv', {'review_id','text','gold_label','target','phenomenon'}, 'gold_label', {'긍정','부정','중립','혼합'})]
    texts = []
    for filename, required, label, allowed in specifications:
        with (root / 'data' / filename).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f'{filename}: 필수 열 누락')
            rows = list(reader)
        if not rows:
            raise ValueError(f'{filename}: 빈 데이터')
        ids = [r['review_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'{filename}: 중복 ID')
        for i, row in enumerate(rows, 2):
            if any(not row.get(k, '').strip() for k in required):
                raise ValueError(f'{filename}:{i}: 빈 필수 값')
            if row[label] not in allowed:
                raise ValueError(f'{filename}:{i}: 잘못된 라벨')
        texts.append({r['text'].strip() for r in rows})
        print(f'{filename}: {len(rows)}행 확인')
    if texts[0] & texts[1]:
        raise ValueError('학습/진단 문장 중복')
    print('검증 완료: 필수 열, 빈 값, ID, 라벨, 학습/진단 중복')

if __name__ == '__main__':
    validate()
