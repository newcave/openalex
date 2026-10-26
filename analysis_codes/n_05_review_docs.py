"""[분석 5단계] 전체기간 근거문서 생성: n_04가 만든 커뮤니티마다 용어 통계/엣지/해당 논문 목록을 정리한다.
웹앱의 전체기간 분석 화면이 이 결과(terms.csv, edges.csv, all_papers.csv)를 읽는다.

두 개의 소스를 쓴다:
- PARTITION_SOURCE(02_full_period): n_04_cluster_network.py가 만든 rising_unigram_clusters.parquet
- SHARED_SOURCE(01_corpus): filtered_corpus/term_normalization_dict/unigram_long

출력: results/<분야>/03_full_period_review/
실행: python -m analysis_codes.n_05_review_docs   (인자 없이 실행하면 기본 분야 config.DEFAULT_DOMAIN)
"""
from pathlib import Path
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DEFAULT  # noqa: E402

PARTITION_SOURCE = DEFAULT.full_output
SHARED_SOURCE = DEFAULT.shared_source
OUT = DEFAULT.full_review


def main(partition_source: Path | None = None, shared_source: Path | None = None, out: Path | None = None):
    # 인자를 안 주면 기본 분야(config.DEFAULT) 경로. 웹앱·pipeline.py는 분야별 경로를 넘긴다.
    PARTITION_SOURCE = Path(partition_source) if partition_source else globals()['PARTITION_SOURCE']
    SHARED_SOURCE = Path(shared_source) if shared_source else globals()['SHARED_SOURCE']
    OUT = Path(out) if out else globals()['OUT']
    OUT.mkdir(parents=True, exist_ok=True)
    partition = pd.read_parquet(PARTITION_SOURCE / 'rising_unigram_clusters.parquet', columns=['cluster', 'term'])
    corpus = pd.read_parquet(SHARED_SOURCE / 'filtered_corpus.parquet')
    corpus = corpus[corpus.language.eq('en')].drop_duplicates('id')
    mapping = pd.read_csv(SHARED_SOURCE / 'term_normalization_dict.csv')
    mapping = dict(zip(mapping.variant, mapping.canonical))
    long = pd.read_parquet(SHARED_SOURCE / 'unigram_long.parquet', columns=['id', 'term'])
    long = long[long.id.isin(corpus.id)].copy()
    long['term'] = long.term.map(lambda t: mapping.get(t, t))
    long = long[long.term.isin(partition.term)].drop_duplicates(['id', 'term'])
    assert partition.term.is_unique
    terms = sorted(partition.term)
    tcat = pd.Categorical(long.term, categories=terms)
    dcat = pd.Categorical(long.id)
    mat = sparse.csr_matrix((np.ones(len(long)), (dcat.codes, tcat.codes)), shape=(len(dcat.categories), len(terms)))
    co = (mat.T @ mat).tocoo()
    mask = (co.row < co.col) & (co.data >= 2)
    edges = pd.DataFrame({'a': np.array(terms)[co.row[mask]], 'b': np.array(terms)[co.col[mask]], 'documents': co.data[mask].astype(int)})
    group = partition.set_index('term').cluster.to_dict()
    edges['cluster_a'] = edges.a.map(group)
    edges['cluster_b'] = edges.b.map(group)
    counts = long.groupby('term').size()
    strength = pd.concat([edges[['a', 'documents']].rename(columns={'a':'term'}), edges[['b', 'documents']].rename(columns={'b':'term'})]).groupby('term').documents.sum()
    inside = edges[edges.cluster_a.eq(edges.cluster_b)]
    istrength = pd.concat([inside[['a', 'documents']].rename(columns={'a':'term'}), inside[['b', 'documents']].rename(columns={'b':'term'})]).groupby('term').documents.sum()
    stats = partition.copy()
    stats['document_count'] = stats.term.map(counts).fillna(0).astype(int)
    stats['internal_strength'] = stats.term.map(istrength).fillna(0)
    stats['total_strength'] = stats.term.map(strength).fillna(0)
    stats['internal_weight_fraction'] = stats.internal_strength.div(stats.total_strength.replace(0, np.nan))
    stats = stats.sort_values(['cluster','document_count','term'], ascending=[True,False,True])
    stats.to_csv(OUT / 'terms.csv', index=False, encoding='utf-8-sig')
    edges.sort_values('documents', ascending=False).to_csv(OUT / 'edges.csv', index=False, encoding='utf-8-sig')
    long['cluster'] = long.term.map(group)
    papers = long.groupby(['cluster','id']).term.agg(lambda s: sorted(s)).reset_index(name='matched_terms')
    papers['matched_count'] = papers.matched_terms.str.len()
    papers = papers.merge(corpus[['id','year','journal','display_name']], on='id', validate='many_to_one')
    papers['url'] = papers.id
    papers['matched_terms'] = papers.matched_terms.map(lambda s: ' | '.join(s))
    papers.to_csv(OUT / 'all_papers.csv', index=False, encoding='utf-8-sig')
    selected = []
    for cid, sub in papers.groupby('cluster'):
        ordered = sub.sort_values(['matched_count','year','id'], ascending=[False,False,True])
        chosen = set()
        covered = set()
        target = set(stats[stats.cluster.eq(cid)].head(60).term)
        for k in range(min(12, len(ordered))):
            candidates = ordered[~ordered.id.isin(chosen)].copy()
            candidates['gain'] = candidates.matched_terms.map(lambda s: len((set(s.split(' | ')) & target) - covered))
            row = candidates.sort_values(['gain','matched_count','year','id'], ascending=[False,False,False,True]).iloc[0]
            chosen.add(row.id)
            covered.update(row.matched_terms.split(' | '))
            rec = row.drop(labels=['gain']).to_dict()
            rec['selection'] = 'coverage_of_top60_terms'
            selected.append(rec)
        for label, condition in [('early_2000_2004', sub.year.between(2000,2004)), ('recent_2021_2025',sub.year.between(2021,2025))]:
            for _, row in sub[condition & ~sub.id.isin(chosen)].sort_values(['matched_count','year','id'], ascending=[False,False,True]).head(3).iterrows():
                chosen.add(row.id)
                rec = row.to_dict()
                rec['selection'] = label
                selected.append(rec)
    sample = pd.DataFrame(selected)
    sample.to_csv(OUT / 'selected_papers.csv', index=False, encoding='utf-8-sig')
    lines = ['# Saved cluster evidence (FDR + association strength)', '', 'Selection is purposive, not a representative sample. Top 60 terms / 12 coverage titles / up to 3 early and 3 recent titles are display budgets, not analytical thresholds.', '', 'Internal weight fraction measures structural concentration among Rising terms only; it is not semantic specificity.', '']
    for cid, sub in stats.groupby('cluster'):
        lines += [f'## Cluster {cid}: {len(sub)} terms; {len(papers[papers.cluster.eq(cid)])} matching papers', '', 'Top 45 by document count:']
        lines += [f'- {r.term}: {r.document_count}; internal weight fraction {r.internal_weight_fraction:.3f}' for r in sub.head(45).itertuples()]
        lines += ['', 'Top 20 internal pairs:']
        lines += [f'- {r.a} + {r.b}: {r.documents}' for r in inside[inside.cluster_a.eq(cid)].sort_values('documents', ascending=False).head(20).itertuples()]
        lines += ['', 'Selected titles:']
        lines += [f'- {r.year} | [{r.display_name}]({r.url}) | {r.matched_terms} | {r.selection}' for r in sample[sample.cluster.eq(cid)].itertuples()]
        lines.append('')
    (OUT / 'evidence.md').write_text('\n'.join(lines), encoding='utf-8')
    manifest = {'source_partition': str(PARTITION_SOURCE), 'source_shared': str(SHARED_SOURCE),
        'source_sha256': {f: hashlib.sha256((SHARED_SOURCE/f).read_bytes()).hexdigest() for f in ['filtered_corpus.parquet','unigram_long.parquet','term_normalization_dict.csv']} | {'rising_unigram_clusters.parquet': hashlib.sha256((PARTITION_SOURCE/'rising_unigram_clusters.parquet').read_bytes()).hexdigest()},
        'terms': len(stats), 'clusters': int(stats.cluster.nunique()), 'english_documents': len(corpus), 'sample_titles': len(sample)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[evidence] terms={len(stats)} clusters={stats.cluster.nunique()}")
    assert not sample.duplicated(['cluster','id']).any()

    interp_path = OUT / 'interpretations.json'
    if not interp_path.exists():
        print(f"[중단] {interp_path} 이(가) 없습니다. evidence.md/terms.csv/edges.csv를 근거로 이 파일을 먼저 작성한 뒤 다시 실행하세요.")
        return
    interp_data = json.loads(interp_path.read_text(encoding='utf-8'))
    interpretations = interp_data['clusters'] if isinstance(interp_data, dict) else interp_data
    conclusion_bullets = interp_data.get('conclusion', []) if isinstance(interp_data, dict) else []
    pair_lookup = {tuple(sorted((r.a,r.b))):r.documents for r in edges.itertuples()}
    titles = corpus.set_index('id')
    n_clusters = int(stats.cluster.nunique())
    review = [f'# Rising {n_clusters}개 군집 검토표 (FDR 보정 + association strength 정규화)', '',
        '작성일: 2026-09-09. 이 문서는 두 가지를 함께 적용한 결과다 — 다중검정 보정(Benjamini-Hochberg FDR, Rising 용어 1,462개 → 1,080개로 축소)과, 동시출현 네트워크의 엣지 가중치 정규화(association strength 방식, 아래 "자료와 검토 범위"에서 이유를 설명한다). 이전 버전(FDR 미적용, 원시 동시출현 가중치 그대로 사용, 5개 군집, modularity 0.192)은 reports/rising_cluster_review/에 그대로 남겨뒀다. 이번 버전은 modularity 0.300, 군집 8개다.', '',
        '## 누가 어떤 근거로 이름을 정했는가', '',
        '군집에 붙은 잠정 이름과 세부 주제는 Claude가 실제 용어·동시출현 횟수·논문 제목을 직접 읽고 제안한 해석이다. Louvain 알고리즘이 이름을 만들거나 세부 주제를 검증해준 것이 아니다. 분야 전문가(수자원 전문가 2인)가 아래 근거를 직접 확인해 이름과 군집 분할 수준을 최종 판단해야 하며, 이 초안은 얼마든지 수정될 수 있다.', '',
        '- 계산으로 나온 것: 군집 소속, 고유 문서 수, 용어별 문서 빈도, 단어쌍 동시출현 횟수',
        '- 사람(Claude)이 해석한 것: 연구주제 이름, 세부 주제 후보, 더 세분화해야 하는지에 대한 판단',
        '- "빈도 상위 용어" 목록은 그 군집의 논문에 자주 등장하는 용어를 보여준다. "내부 연결 비중 상위 용어" 목록은 보조 지표로, 그 용어가 맺은 전체 연결 중 같은 군집 안으로 향한 연결의 비율이 높은 용어를 보여준다. 다만 등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 이 비율이 쉽게 100%에 가까워지므로, 이 목록만으로 군집을 대표하는 용어라고 단정하면 안 된다.',
        '- 아래 세부 주제들은 실제 논문 제목과 동시출현 빈도를 근거로 묶어본 사례다. 주제끼리 겹칠 수 있고, 군집 안의 모든 용어·논문에 빠짐없이 하위 라벨을 붙인 결과는 아니다.', '',
        '## 자료와 검토 범위', '',
        f'- 영어 문서 {len(corpus):,}건 중 FDR 보정을 통과한 Rising 용어 {len(stats):,}개를 대상으로 총 {n_clusters}개 군집으로 나눴다.',
        '- **엣지 가중치를 association strength로 정규화한 이유**: 두 용어의 연결 강도를 원시 동시출현 횟수로 그대로 쓰면, "water"처럼 그냥 등장 빈도 자체가 높은 범용어가 실제 의미적 연관과 무관하게 거의 모든 용어와 강하게 연결된 것처럼 나온다. Van Eck & Waltman(2009, *Journal of the American Society for Information Science and Technology*, 60(8))이 여러 정규화 방식을 비교 검증해, 두 용어가 각자 등장하는 빈도를 감안했을 때 우연한 수준보다 얼마나 더 자주 같이 나오는지를 재는 association strength(동시출현 횟수를 두 용어 각각의 문서 등장 횟수를 곱한 값의 제곱근으로 나눈 값)가 이 문제에 가장 적합하다고 결론지었고, 계량서지 네트워크 시각화의 표준 도구인 VOSviewer도 이를 기본 정규화 방식으로 쓴다. 이 분석도 같은 방식을 적용했다. 연결로 인정하는 기준(동시출현 2건 이상)은 이전 버전과 동일하다.',
        f'- **논문 제목 표본({len(sample)}개)을 고른 방법**: 군집별로 문서 수가 많은 상위 60개 용어를 그 군집의 "대표 어휘"로 삼고, 이 대표 어휘를 가장 폭넓게 새로 포함하는 논문 제목을 최대 12개까지 순서대로 골랐다(이미 고른 논문이 보여준 단어는 제외하고, 아직 안 나온 대표 어휘를 가장 많이 포함하는 논문을 다음으로 고르는 방식 — 적은 개수만 읽어도 군집의 다양한 하위주제를 훑어볼 수 있게 하기 위함이다). 여기에 옛날 논문(2000~2004년)과 최근 논문(2021~2025년) 중 최대 3개씩을 추가해, 시간에 따라 내용이 달라지는지도 함께 볼 수 있게 했다. 12개·3개라는 숫자는 검토자가 읽을 분량을 정한 것일 뿐 통계적 기준값이 아니다.',
        '- **이 표본은 무작위 표본이 아니다.** "이 군집을 가장 잘 보여줄 만한" 논문을 의도적으로 골라낸 것이므로, 표본에 나온 논문의 비율을 군집 전체의 비율로 일반화해서는 안 된다. 코퍼스 자체가 최근으로 갈수록 논문 수가 많아지는 구조라, 표본에서도 최근 논문 비중이 자연히 커진다.',
        '- 문서 수는 OpenAlex ID 기준으로 집계했다. 여러 군집의 용어를 함께 포함한 논문은 두 군집 모두에서 중복으로 잡힌다.', '',
        '## 요약', '', '| 군집 ID | 용어 수 | 용어가 등장한 문서 수 | 잠정 주제명(해석) | 1차 판단(해석) |', '|---|---:|---:|---|---|']
    for item in interpretations:
        cid = item['cluster']
        sub = stats[stats.cluster.eq(cid)]
        review.append(f"| {cid} | {len(sub)} | {len(papers[papers.cluster.eq(cid)]):,} | {item['label']} | {item['judgment']} |")
    for item in interpretations:
        cid = item['cluster']
        sub = stats[stats.cluster.eq(cid)]
        review += ['', f"## 군집 {cid}: {item['label']}", '', '**빈도 상위 용어(계산값):** ' + ', '.join(f'{r.term}({r.document_count:,}건)' for r in sub.head(15).itertuples()), '', '**내부 연결 비중 상위 용어(보조 지표):** ' + ', '.join(f'{r.term}({r.document_count}건, {r.internal_weight_fraction:.1%})' for r in sub.sort_values(['internal_weight_fraction','document_count'],ascending=False).head(10).itertuples()), '', '등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.', '']
        for theme in item['themes']:
            review += [f"### 세부 주제 후보: {theme['name']}", '']
            for a,b in theme['pairs']:
                number = pair_lookup.get(tuple(sorted((a,b))), 0)
                review.append(f'- 동시출현: `{a}` + `{b}` = {number:,}건' if number else f'- `{a}` + `{b}`: 저장 기준(2건 이상)의 연결 없음. 이 조합 자체를 정량 근거로 삼지 않는다.')
            for wid in theme['papers']:
                pid = 'https://openalex.org/' + wid
                assert pid in titles.index
                r = titles.loc[pid]
                hit = papers[papers.cluster.eq(cid) & papers.id.eq(pid)]
                assert len(hit) == 1
                title = str(r.display_name).replace('<','&lt;').replace('>','&gt;')
                review.append(f"- 근거 제목 ({r.year}, {r.journal}): [{title}]({pid})\n  - 이 군집에 속한 추출 용어: {hit.iloc[0].matched_terms}")
            review.append('')
        review += [f"**판단:** {item['judgment']}", '', f"**해석의 한계/점검 사항:** {item['caution']}", '']
    review += ['## 검토 결론과 다음 작업 제안', '']
    review += [f'- {b}' for b in conclusion_bullets]
    review += ['', '## 확인 파일', '',
        '- [전체 용어와 계산 지표](terms.csv)', '- [군집 내부·외부 연결 전체](edges.csv)', '- [선정 논문 제목과 선정 방식](selected_papers.csv)', '- [군집 용어가 등장한 전체 문서](all_papers.csv)', '- [상위 단어·연결·제목 모음](evidence.md)', '- [해석 초안의 구조화된 기록](interpretations.json)', '- [입력 파일 해시와 실행 규모](manifest.json)', '',
        '재현 명령: `python -m analysis_codes.n_04_cluster_network` 후 `python -m analysis_codes.n_05_review_docs`']
    (OUT / 'review.md').write_text('\n'.join(review), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
