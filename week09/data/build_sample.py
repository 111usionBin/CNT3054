"""Rebuild deterministic synthetic teaching reviews; not scraped user reviews."""
import csv
import random
from pathlib import Path

rng=random.Random(3054)
fragments=[
 ['배우의 표정과 눈빛이 인물의 감정을 전했다.','주연 배우의 대사 전달과 발음이 또렷했다.','조연의 연기와 배우들의 호흡이 자연스러웠다.','캐릭터의 감정을 드러내는 배우의 몸짓이 기억난다.','대사와 표정이 어울려 인물의 성격을 이해할 수 있었다.','캐스팅이 캐릭터의 이미지와 잘 맞았다.','배우의 연기는 좋았지만 대사 전달에는 아쉬움이 있었다.','조연 배우의 표정 연기가 인물의 갈등을 잘 보여줬다.'],
 ['이야기 전개와 결말이 설득력 있게 이어졌다.','초반 복선이 마지막 반전과 연결되는 구성이 좋았다.','서사의 흐름과 사건의 인과관계가 명확했다.','결말까지 갈등이 이어져 이야기의 긴장감이 유지됐다.','원작 각색으로 사건의 순서와 결말이 달라졌다.','이야기 전개는 느렸지만 마지막 반전은 인상적이었다.','복선의 배치와 사건의 구성이 서사를 풍부하게 만들었다.','결말에서 갈등이 급하게 해결되어 이야기의 설득력이 약했다.'],
 ['배경 음악과 효과음이 분위기에 잘 어울렸다.','주제곡의 멜로디와 가사가 오래 기억에 남았다.','음향의 공간감과 배경 음악의 균형이 좋았다.','노래와 악기 연주가 감정을 전달했다.','음악의 리듬과 음향 효과가 긴장감을 높였다.','주제곡은 좋았지만 효과음의 볼륨은 과했다.','사운드트랙의 멜로디와 악기 구성이 인상적이었다.','배경 음악의 볼륨 때문에 대사가 잘 들리지 않았다.'],
 ['카메라 구도와 조명이 공간의 분위기를 만들었다.','색감과 촬영 기법이 화면의 아름다움을 살렸다.','편집의 속도와 화면 전환이 자연스러웠다.','미술과 의상 디자인이 시대적 배경을 보여줬다.','촬영과 조명의 조합으로 색감이 선명했다.','영상미는 좋았지만 화면 전환이 너무 빨랐다.','카메라 움직임과 편집이 공간의 깊이를 표현했다.','조명과 의상의 색감이 어울려 미술 디자인이 돋보였다.']]
rows=[]
for group in fragments:
 pairs=[(a,b) for a in range(8) for b in range(8) if a!=b]
 rng.shuffle(pairs)
 for a,b in pairs[:30]: rows.append(group[a]+' '+group[b])
for i in range(4):
 for j in range(i+1,4):
  pairs=[(a,b) for a in range(8) for b in range(8)]
  rng.shuffle(pairs)
  for a,b in pairs[:8]: rows.append(fragments[i][a]+' '+fragments[j][b])
rng.shuffle(rows)
root=Path(__file__).resolve().parent
with (root/'culture_reviews.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['review_id','text','source_type'])
 for i,text in enumerate(rows,1):w.writerow([f'SYN{i:03d}',text,'synthetic_educational'])
with (root/'my_reviews_template.csv').open('w',encoding='utf-8-sig',newline='') as f:
 csv.writer(f).writerow(['review_id','text'])
print(f'{len(rows)} synthetic reviews')
