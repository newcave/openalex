"""Read-only provenance audit for sub/scp tokens in the saved English corpus."""
from pathlib import Path
import sys
import re
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.text_utils import _TOKEN_RX, raw_tokens


def main():
    source = ROOT / 'analysis' / 'output_임계15'
    out = ROOT / 'reports' / 'rising_cluster_review'
    corpus = pd.read_parquet(source / 'filtered_corpus.parquet')
    corpus = corpus[corpus.language.eq('en')].drop_duplicates('id')
    dictionary = pd.read_csv(source / 'term_normalization_dict.csv')
    mapping = dict(zip(dictionary.variant, dictionary.canonical))
    saved = pd.read_parquet(source / 'unigram_long.parquet', columns=['id','term'])
    saved['term'] = saved.term.map(lambda t: mapping.get(t,t))
    records = []
    summaries = {}
    for target in ['sub','scp']:
        ids = set(saved.loc[saved.term.eq(target), 'id']) & set(corpus.id)
        for r in corpus[corpus.id.isin(ids)].itertuples():
            title = r.display_name
            tag_spans = [m.span() for m in re.finditer(r'<\s*/?\s*[A-Za-z][^>]*>', title)]
            hits = []
            for match in _TOKEN_RX.finditer(title.lower()):
                raw = match.group().strip('-')
                if mapping.get(raw,raw) != target:
                    continue
                is_tag = any(a <= match.start() and match.end() <= b for a,b in tag_spans)
                hits.append((is_tag, title[max(0,match.start()-35):min(len(title),match.end()+45)], raw))
            assert hits, (r.id,target)
            tag_n = sum(x[0] for x in hits)
            outside_n = len(hits) - tag_n
            records.append({'term':target,'id':r.id,'year':r.year,'title':title,'tag_occurrences':tag_n,'outside_tag_occurrences':outside_n,'origin':'tag_only' if not outside_n else ('mixed' if tag_n else 'outside_only'),'contexts':json.dumps(hits,ensure_ascii=False)})
    data = pd.DataFrame(records)
    data.to_csv(out / 'markup_provenance.csv',index=False,encoding='utf-8-sig')
    for target, sub in data.groupby('term'):
        summaries[target] = {'documents':len(sub),'origin_counts':sub.origin.value_counts().to_dict(),'tag_occurrences':int(sub.tag_occurrences.sum()),'outside_tag_occurrences':int(sub.outside_tag_occurrences.sum())}
    examples = ['sub-basin','sub‐basin','sub–basin','sub basin','CO<sub>2</sub>','<scp>GRACE</scp>','SCP']
    lines = ['# sub / scp 토큰 출처 전수 점검','', '기존 영어 코퍼스와 정규화된 유니그램 저장 결과를 대조했다. 원문 토큰 위치가 `<...>` 태그 구문 안에 있는지 검사했다. 구문상 태그 확인이며 출판사 원본 HTML까지 조회한 것은 아니다. 원본·전처리·군집은 수정하지 않았다.','', '## 집계','',json.dumps(summaries,ensure_ascii=False,indent=2),'','## 현재 토크나이저 재현','']
    for title in examples:
        lines.append(f'- `{title}` → `{raw_tokens(title)}`')
    for target in ['sub','scp']:
        lines += ['',f'## {target}: 태그 외부 출현이 있는 문서 전체','']
        for r in data[data.term.eq(target) & data.outside_tag_occurrences.gt(0)].itertuples():
            lines += [f'- {r.id} ({r.year})',f'  - `{r.title}`',f'  - {r.contexts}']
        lines += ['',f'## {target}: 태그 안에서만 나온 문서 예시','']
        for r in data[data.term.eq(target) & data.origin.eq('tag_only')].head(4).itertuples():
            lines += [f'- {r.id} ({r.year})',f'  - `{r.title}`']
    (out / 'markup_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
