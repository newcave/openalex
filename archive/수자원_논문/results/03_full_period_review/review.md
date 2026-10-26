# Rising 8개 군집 검토표 (FDR 보정 + association strength 정규화)

작성일: 2026-09-09. 이 문서는 두 가지를 함께 적용한 결과다 — 다중검정 보정(Benjamini-Hochberg FDR, Rising 용어 1,462개 → 1,080개로 축소)과, 동시출현 네트워크의 엣지 가중치 정규화(association strength 방식, 아래 "자료와 검토 범위"에서 이유를 설명한다). 이전 버전(FDR 미적용, 원시 동시출현 가중치 그대로 사용, 5개 군집, modularity 0.192)은 reports/rising_cluster_review/에 그대로 남겨뒀다. 이번 버전은 modularity 0.300, 군집 8개다.

## 누가 어떤 근거로 이름을 정했는가

군집에 붙은 잠정 이름과 세부 주제는 Claude가 실제 용어·동시출현 횟수·논문 제목을 직접 읽고 제안한 해석이다. Louvain 알고리즘이 이름을 만들거나 세부 주제를 검증해준 것이 아니다. 분야 전문가(수자원 전문가 2인)가 아래 근거를 직접 확인해 이름과 군집 분할 수준을 최종 판단해야 하며, 이 초안은 얼마든지 수정될 수 있다.

- 계산으로 나온 것: 군집 소속, 고유 문서 수, 용어별 문서 빈도, 단어쌍 동시출현 횟수
- 사람(Claude)이 해석한 것: 연구주제 이름, 세부 주제 후보, 더 세분화해야 하는지에 대한 판단
- "빈도 상위 용어" 목록은 그 군집의 논문에 자주 등장하는 용어를 보여준다. "내부 연결 비중 상위 용어" 목록은 보조 지표로, 그 용어가 맺은 전체 연결 중 같은 군집 안으로 향한 연결의 비율이 높은 용어를 보여준다. 다만 등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 이 비율이 쉽게 100%에 가까워지므로, 이 목록만으로 군집을 대표하는 용어라고 단정하면 안 된다.
- 아래 세부 주제들은 실제 논문 제목과 동시출현 빈도를 근거로 묶어본 사례다. 주제끼리 겹칠 수 있고, 군집 안의 모든 용어·논문에 빠짐없이 하위 라벨을 붙인 결과는 아니다.

## 자료와 검토 범위

- 영어 문서 91,125건 중 FDR 보정을 통과한 Rising 용어 1,080개를 대상으로 총 8개 군집으로 나눴다.
- **엣지 가중치를 association strength로 정규화한 이유**: 두 용어의 연결 강도를 원시 동시출현 횟수로 그대로 쓰면, "water"처럼 그냥 등장 빈도 자체가 높은 범용어가 실제 의미적 연관과 무관하게 거의 모든 용어와 강하게 연결된 것처럼 나온다. Van Eck & Waltman(2009, *Journal of the American Society for Information Science and Technology*, 60(8))이 여러 정규화 방식을 비교 검증해, 두 용어가 각자 등장하는 빈도를 감안했을 때 우연한 수준보다 얼마나 더 자주 같이 나오는지를 재는 association strength(동시출현 횟수를 두 용어 각각의 문서 등장 횟수를 곱한 값의 제곱근으로 나눈 값)가 이 문제에 가장 적합하다고 결론지었고, 계량서지 네트워크 시각화의 표준 도구인 VOSviewer도 이를 기본 정규화 방식으로 쓴다. 이 분석도 같은 방식을 적용했다. 연결로 인정하는 기준(동시출현 2건 이상)은 이전 버전과 동일하다.
- **논문 제목 표본(141개)을 고른 방법**: 군집별로 문서 수가 많은 상위 60개 용어를 그 군집의 "대표 어휘"로 삼고, 이 대표 어휘를 가장 폭넓게 새로 포함하는 논문 제목을 최대 12개까지 순서대로 골랐다(이미 고른 논문이 보여준 단어는 제외하고, 아직 안 나온 대표 어휘를 가장 많이 포함하는 논문을 다음으로 고르는 방식 — 적은 개수만 읽어도 군집의 다양한 하위주제를 훑어볼 수 있게 하기 위함이다). 여기에 옛날 논문(2000~2004년)과 최근 논문(2021~2025년) 중 최대 3개씩을 추가해, 시간에 따라 내용이 달라지는지도 함께 볼 수 있게 했다. 12개·3개라는 숫자는 검토자가 읽을 분량을 정한 것일 뿐 통계적 기준값이 아니다.
- **이 표본은 무작위 표본이 아니다.** "이 군집을 가장 잘 보여줄 만한" 논문을 의도적으로 골라낸 것이므로, 표본에 나온 논문의 비율을 군집 전체의 비율로 일반화해서는 안 된다. 코퍼스 자체가 최근으로 갈수록 논문 수가 많아지는 구조라, 표본에서도 최근 논문 비중이 자연히 커진다.
- 문서 수는 OpenAlex ID 기준으로 집계했다. 여러 군집의 용어를 함께 포함한 논문은 두 군집 모두에서 중복으로 잡힌다.

## 요약

| 군집 ID | 용어 수 | 용어가 등장한 문서 수 | 잠정 주제명(해석) | 1차 판단(해석) |
|---|---:|---:|---|---|
| 0 | 317 | 28,674 | 수처리 공정·미생물·수환경 오염 | 이전(FDR 미적용) 버전의 군집0과 핵심 구성이 거의 동일하게 유지됨 — 방법 변경(FDR·association strength)에 강건해 보이나, 세부 기술별 비교를 위해서는 여전히 세분화 우선 검토 대상 |
| 1 | 291 | 45,705 | 지하수 동태와 중국 지역 사례·동위원소 추적 | 이전 버전에서 별개로 흩어져 있던 지하수 일반 논의·중국 지역연구·동위원소 추적이 하나의 대형 군집으로 묶였다 — 대상(중국/고원/일반 지하수)과 방법(동위원소)이 혼재해 세분화 후보 |
| 2 | 278 | 44,865 | AI·데이터 기반 수문 관측·예측과 저수지 운영 | 관측·예측 방법론(기계학습·원격탐사)이 여러 연구대상(홍수·가뭄·저수지 운영)에 공통 적용되며 하나의 대형 군집으로 강하게 연결됨 — 방법과 대상이 교차하므로 세분화 후보 |
| 3 | 133 | 31,626 | 포괄적 물관리·도시 물시스템과 정책 | 용어 'water'의 내부연결비중이 0.179로 8개 군집 중 가장 낮다 — 즉 'water'는 이 군집 밖과도 광범위하게 연결되는 범용어이며, 이 군집은 대상(도시/관개/공급/거버넌스)과 방법이 가장 혼재된 '포괄적 물관리' 우산 군집이다. 세분화 우선순위가 가장 높음 |
| 4 | 39 | 4,143 | 지중유체(CO2) 저장·다공질매체 유동과 상수관망 센서 | 서로 다른 두 응용분야(CO2 지중저장의 다공질매체 유동 vs 상수관망 누수감지 센서)가 '압력(pressure)'이라는 공유 물리량과 '센서' 계측기술을 매개로 하나의 군집에 묶였다. 실제 연구공동체는 다를 가능성이 높아 세분화 검토 1순위 |
| 5 | 16 | 2,159 | 연안대수층 해수침투와 해수면 상승 | 해수침투·해수면상승이라는 뚜렷한 단일 주제로 응집된, 8개 군집 중 가장 주제적으로 선명한 군집. 이전(원시가중치) 버전에서 지하수 일반 논의에 섞여 있던 것과 달리 이번엔 완전히 독립된 군집으로 분리됨 |
| 6 | 4 | 121 | (잠정 보류) 생지화학적 hot spot·건물 배관 수질 — 서로 다른 두 주제의 우연한 결합 가능성 | 4개 용어 중 실제로 강하게 연결된 쌍은 plumbing+premise(14건)와 hot+spots(14건) 두 개뿐이며, 이 둘을 잇는 hot+plumbing 연결은 2건(저장 임계값의 경계)에 불과하다. 즉 서로 무관한 두 주제(생지화학적 hot spot/hot moment 개념과 건물 premise plumbing 수질)가 약한 다리 하나로 묶였을 가능성이 높다 — 하나의 군집으로 해석하지 않는 것이 안전 |
| 7 | 2 | 32 | 탈질산 암모늄화(DNRA) 질소순환 경로 | 2개 용어(dissimilatory, dnra)로만 이뤄진 매우 작지만 주제적으로는 명확한 군집 — DNRA(탈질산 암모늄화)라는 구체적인 질소순환 미생물 경로를 가리키는 전문용어 쌍이 그대로 하나의 군집이 됨 |

## 군집 0: 수처리 공정·미생물·수환경 오염

**빈도 상위 용어(계산값):** wastewater(2,869건), performance(1,700건), process(1,522건), potential(1,330건), dissolved(1,318건), microbial(1,294건), matter(1,174건), membrane(1,165건), role(1,143건), insights(1,076건), plant(1,021건), novel(929건), high(927건), enhanced(892건), production(875건)

**내부 연결 비중 상위 용어(보조 지표):** vulgaris(19건, 100.0%), algal-bacterial(15건, 100.0%), inhibits(15건, 100.0%), sulfidated(21건, 95.0%), cao(17건, 91.1%), mixotrophic(17건, 90.6%), molecules(18건, 89.5%), candidatus(22건, 87.8%), tetracycline(47건, 87.2%), singlet(28건, 87.1%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: 막여과·막오염 및 역삼투

- 동시출현: `fouling` + `membrane` = 284건
- 동시출현: `osmosis` + `reverse` = 197건
- 근거 제목 (2019, Water Research): [Mitigated membrane fouling and enhanced removal of extracellular antibiotic resistance genes from wastewater effluent via an integrated pre-coagulation and microfiltration process](https://openalex.org/W2943511235)
  - 이 군집에 속한 추출 용어: antibiotic | enhanced | extracellular | fouling | genes | membrane | process | resistance | wastewater
- 근거 제목 (2015, Water Research): [Fouling characteristics of reverse osmosis membranes at different positions of a full-scale plant for municipal wastewater reclamation](https://openalex.org/W2215081142)
  - 이 군집에 속한 추출 용어: fouling | full-scale | membranes | osmosis | plant | reclamation | reverse | wastewater

### 세부 주제 후보: 항생제 내성 유전자

- 동시출현: `antibiotic` + `resistance` = 243건
- 동시출현: `genes` + `resistance` = 203건
- 근거 제목 (2021, Water Research): [Insight into using a novel ultraviolet/peracetic acid combination disinfection process to simultaneously remove antibiotics and antibiotic resistance genes in wastewater: Mechanism and comparison with conventional processes](https://openalex.org/W4200346070)
  - 이 군집에 속한 추출 용어: antibiotic | antibiotics | conventional | disinfection | genes | insights | novel | process | remove | resistance | simultaneously | wastewater
- 근거 제목 (2018, Water Research): [Tertiary treatment of urban wastewater by solar and UV-C driven advanced oxidation with peracetic acid: Effect on contaminants of emerging concern and antibiotic resistance](https://openalex.org/W2901039749)
  - 이 군집에 속한 추출 용어: advanced | antibiotic | concern | driven | emerging | resistance | solar | wastewater

### 세부 주제 후보: 세포외고분자물질(EPS)과 소독 부산물

- 동시출현: `extracellular` + `polymeric` = 117건
- 동시출현: `byproducts` + `disinfection` = 111건
- 근거 제목 (2025, Water Research): [Insights into in-situ free nitrous acid induced extracellular polymeric substances changes and membrane fouling mitigation in a nitritation membrane bioreactor](https://openalex.org/W4408210789)
  - 이 군집에 속한 추출 용어: extracellular | fouling | in-situ | induced | insights | membrane | mitigation | nitritation | nitrous | polymeric | substances
- 근거 제목 (2025, Water Research): [Linking oxygen-induced oxidative stress to resource recovery by enhancing the production of extracellular polymeric substances in activated sludge microbial communities](https://openalex.org/W4412468162)
  - 이 군집에 속한 추출 용어: communities | enhancing | extracellular | microbial | polymeric | production | recovery | stress | substances

### 세부 주제 후보: 신흥·미량오염물질의 발생과 거동

- 동시출현: `microplastics` + `occurrence` = 25건
- 근거 제목 (2025, Water Research): [Molecular fractionation on ferrihydrite eroded the disinfection byproduct formation potential of dissolved organic matter derived from microplastics and biochar](https://openalex.org/W4408361109)
  - 이 군집에 속한 추출 용어: biochar | byproduct | disinfection | dissolved | matter | microplastics | molecular | potential
- 근거 제목 (2020, Water Research): [Occurrence and transformation of gabapentin in urban water quality engineering: Rapid formation of nitrile from amine during drinking water chlorination](https://openalex.org/W3041819629)
  - 이 군집에 속한 추출 용어: occurrence | rapid | transformation

**판단:** 이전(FDR 미적용) 버전의 군집0과 핵심 구성이 거의 동일하게 유지됨 — 방법 변경(FDR·association strength)에 강건해 보이나, 세부 기술별 비교를 위해서는 여전히 세분화 우선 검토 대상

**해석의 한계/점검 사항:** 수처리라는 공통 맥락은 존재하나 막공정/항생제내성/EPS·소독부산물/신흥오염물질이 혼재. 용어 수는 436→317개로 줄었지만(FDR 보정으로 완만한 변화 용어 다수 제외) 상위 용어·쌍의 구성은 이전과 크게 다르지 않다.


## 군집 1: 지하수 동태와 중국 지역 사례·동위원소 추적

**빈도 상위 용어(계산값):** groundwater(5,693건), impact(4,256건), dynamics(3,279건), china(2,792건), control(1,991건), area(1,773건), changes(1,643건), response(1,581건), region(1,558건), different(1,443건), storage(1,391건), variations(1,338건), patterns(1,291건), evapotranspiration(1,284건), implications(1,201건)

**내부 연결 비중 상위 용어(보조 지표):** bars(32건, 100.0%), erie(24건, 86.7%), xerophytic(17건, 86.4%), patches(25건, 84.6%), qilian(23건, 80.0%), inner(47건, 79.0%), tianshan(23건, 78.7%), thermokarst(21건, 77.8%), dongting(21건, 75.0%), permafrost(142건, 73.8%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: 고원·건조지역 지하수-지표수 변동

- 동시출현: `plateau` + `tibetan` = 281건
- 동시출현: `loess` + `plateau` = 250건
- 근거 제목 (2025, Journal of Hydrology): [Stable isotope evidence for the moisture sources and runoff components in the southwestern Loess Plateau, China](https://openalex.org/W4408359553)
  - 이 군집에 속한 추출 용어: china | evidence | isotope | loess | plateau | sources | stable
- 근거 제목 (2025, Water Resources Research): [Evaporation From Soil and Thermokarst Lakes Contribute More to Local Precipitation Than Plant Transpiration in Permafrost Regions of the Central Qinghai—Tibet Plateau](https://openalex.org/W4415724302)
  - 이 군집에 속한 추출 용어: contribute | evaporation | lakes | permafrost | plateau | qinghai | region | thermokarst | tibet

### 세부 주제 후보: 중국 지역별 지하수 사례연구

- 동시출현: `china` + `groundwater` = 292건
- 동시출현: `china` + `impact` = 256건
- 근거 제목 (2021, Journal of Hydrology): [Spatiotemporal evolution characteristics of land subsidence caused by groundwater depletion in the North China plain during the past six decades](https://openalex.org/W3181597087)
  - 이 군집에 속한 추출 용어: caused | china | decades | evolution | groundwater | past | spatiotemporal | subsidence
- 근거 제목 (2023, Journal of Hydrology): [Investigating the spatial–temporal changes of flood events across the Yangtze River Basin, China: Identification, spatial heterogeneity, and dominant impact factors](https://openalex.org/W4367671986)
  - 이 군집에 속한 추출 용어: across | changes | china | dominant | factors | heterogeneity | impact | investigating | spatial-temporal | yangtze

### 세부 주제 후보: 동위원소 기반 수문 기원 추적

- 동시출현: `isotopes` + `stable` = 257건
- 동시출현: `isotope` + `stable` = 211건
- 근거 제목 (2021, Journal of Hydrology): [Temporal variations and evaporation control effect of the stable isotope composition of precipitation in the subtropical monsoon climate region, Southwest China](https://openalex.org/W3149731500)
  - 이 군집에 속한 추출 용어: china | composition | control | evaporation | isotope | monsoon | region | southwest | stable | subtropical | variations
- 근거 제목 (2023, Hydrological Processes): [Using stable water isotopes to evaluate water flow and nonpoint source pollutant contributions in three southern Ontario agricultural headwater streams](https://openalex.org/W4313480414)
  - 이 군집에 속한 추출 용어: agricultural | contribution | headwater | isotopes | source | stable

### 세부 주제 후보: 육상 저수량(TWS)·증발산 변동

- 동시출현: `groundwater` + `storage` = 151건
- 근거 제목 (2025, Water Resources Research): [Controls of Climate Seasonality and Vegetation Dynamics on the Seasonal Variability of Terrestrial Water Storage Under Diverse Climate Regimes](https://openalex.org/W4407654330)
  - 이 군집에 속한 추출 용어: control | diverse | dynamics | regimes | seasonal | seasonality | storage | terrestrial | vegetation
- 근거 제목 (2025, Water Resources Research): [Spatiotemporal Variations of Terrestrial Water Storage and Driving Factors in the Water Towers of Northwest China Based on GRACE and Multi‐Source Data Sets](https://openalex.org/W4414575258)
  - 이 군집에 속한 추출 용어: china | driving | factors | grace | northwest | spatiotemporal | storage | terrestrial | variations

**판단:** 이전 버전에서 별개로 흩어져 있던 지하수 일반 논의·중국 지역연구·동위원소 추적이 하나의 대형 군집으로 묶였다 — 대상(중국/고원/일반 지하수)과 방법(동위원소)이 혼재해 세분화 후보

**해석의 한계/점검 사항:** china가 top15 용어(2,792건)에 들 만큼 지역 사례연구 비중이 크다. 연안대수층 해수침투(과거 같은 군집에 있었던 하위주제)는 이번에는 별도 군집(5)으로 완전히 분리됐다 — 이것이 association strength 정규화의 효과인지, 우연한 재배치인지는 이 표만으로 단정할 수 없다.


## 군집 2: AI·데이터 기반 수문 관측·예측과 저수지 운영

**빈도 상위 용어(계산값):** hydrological(3,774건), approach(3,760건), flood(3,563건), network(3,092건), precipitation(2,460건), reservoir(2,158건), streamflow(2,096건), prediction(1,985건), drought(1,756건), moisture(1,706건), framework(1,388건), learning(1,374건), forecasting(1,359건), optimization(1,315건), monitoring(1,251건)

**내부 연결 비중 상위 용어(보조 지표):** ensemble-based(30건, 93.8%), post-processing(49건, 90.1%), multi-step-ahead(22건, 88.9%), ahead(25건, 88.2%), breach(38건, 84.7%), k-means(16건, 84.4%), generative(43건, 84.4%), smoother(36건, 83.0%), merging(77건, 82.9%), informed(36건, 82.7%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: 기계학습·딥러닝 기반 수문 예측

- 동시출현: `learning` + `machine` = 744건
- 동시출현: `deep` + `learning` = 432건
- 근거 제목 (2024, Journal of Hydrology): [Novel time-lag informed deep learning framework for enhanced streamflow prediction and flood early warning in large-scale catchments](https://openalex.org/W4391541938)
  - 이 군집에 속한 추출 용어: deep | early | flood | framework | informed | large-scale | learning | prediction | streamflow | warning
- 근거 제목 (2024, Journal of Hydrology): [Real-time short-range flood forecasting based on a watershed scale 2-D hydrodynamic model and high-resolution precipitation forecast ensemble](https://openalex.org/W4405442097)
  - 이 군집에 속한 추출 용어: ensemble | flood | forecast | forecasting | high-resolution | hydrodynamic | precipitation | real-time

### 세부 주제 후보: 저수지·수력 운영 최적화

- 동시출현: `operation` + `reservoir` = 552건
- 근거 제목 (2019, Journal of Hydrology): [Operation rule derivation of hydropower reservoir by k-means clustering method and extreme learning machine based on particle swarm optimization](https://openalex.org/W2949533264)
  - 이 군집에 속한 추출 용어: clustering | extreme | hydropower | k-means | learning | machine | operation | optimization | particle | reservoir | swarm
- 근거 제목 (2001, Journal of Hydrology): [Reservoir optimization using sampling SDP with ensemble streamflow prediction (ESP) forecasts](https://openalex.org/W1993195626)
  - 이 군집에 속한 추출 용어: ensemble | forecast | optimization | prediction | reservoir | streamflow

### 세부 주제 후보: 원격탐사·자료동화 기반 관측

- 동시출현: `remote` + `sensing` = 494건
- 근거 제목 (2016, Hydrology and earth system sciences): [Improved large-scale hydrological modelling through the assimilation of streamflow and downscaled satellite soil moisture observations](https://openalex.org/W2139389869)
  - 이 군집에 속한 추출 용어: assimilation | hydrological | improved | large-scale | moisture | observations | satellite | streamflow
- 근거 제목 (2020, Water Resources Research): [Improving Global Monthly and Daily Precipitation Estimation by Fusing Gauge Observations, Remote Sensing, and Reanalysis Data Sets](https://openalex.org/W3008241583)
  - 이 군집에 속한 추출 용어: global | improving | monthly | observations | precipitation | reanalysis | remote | sensing

### 세부 주제 후보: 홍수·가뭄 극한현상 예측

- 동시출현: `flash` + `flood` = 216건
- 동시출현: `flood` + `inundation` = 210건
- 근거 제목 (2025, Journal of Hydrology): [A hybrid framework for real-time flash flood forecasting in small ungauged catchments: integrating hydrodynamic simulations with LSTM networks](https://openalex.org/W4411275863)
  - 이 군집에 속한 추출 용어: flash | flood | forecasting | framework | hybrid | hydrodynamic | integrating | lstm | network | real-time
- 근거 제목 (2022, Journal of Hydrology): [Nonstationary analysis of hydrological drought index in a coupled human-water system: Application of the GAMLSS with meteorological and anthropogenic covariates in the Wuding River basin, China](https://openalex.org/W4220678505)
  - 이 군집에 속한 추출 용어: coupled | drought | human-water | hydrological | index | meteorological | nonstationary

**판단:** 관측·예측 방법론(기계학습·원격탐사)이 여러 연구대상(홍수·가뭄·저수지 운영)에 공통 적용되며 하나의 대형 군집으로 강하게 연결됨 — 방법과 대상이 교차하므로 세분화 후보

**해석의 한계/점검 사항:** 이전 버전에서 저수지 운영 최적화는 별도 군집(수자원관리)에 속했으나, 이번에는 예측·관측 군집에 흡수됐다. operation+reservoir(552건) 연결이 learning+machine(744건) 등 예측 관련 연결보다 약해 이 군집 안에서도 비중이 다르다.


## 군집 3: 포괄적 물관리·도시 물시스템과 정책

**빈도 상위 용어(계산값):** water(16,355건), system(5,889건), urban(2,230건), risk(1,372건), environmental(1,070건), assessing(1,060건), review(945건), irrigation(826건), energy(791건), combined(696건), supply(682건), stormwater(540건), critical(498건), strategies(450건), sustainable(442건)

**내부 연결 비중 상위 용어(보조 지표):** equity(28건, 73.6%), perception(24건, 72.5%), advancements(15건, 69.7%), qmra(18건, 67.4%), water-energy-food(29건, 66.7%), investment(41건, 65.6%), epidemiology(59건, 64.8%), adoption(28건, 64.7%), wastewater-based(61건, 63.7%), sanitation(37건, 62.7%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: 도시 물 공급·인프라 안전

- 동시출현: `supply` + `water` = 571건
- 동시출현: `system` + `urban` = 299건
- 근거 제목 (2025, Journal of Hydrology): [Beyond water supply augmentation: Environmental benefits of infrastructure investment for a regional water supply system](https://openalex.org/W4408422825)
  - 이 군집에 속한 추출 용어: benefits | beyond | environmental | infrastructure | investment | supply | system | water
- 근거 제목 (2025, Journal of Water Resources Planning and Management): [Assessing Water Supply Systems in Remote Alaska: Challenges in Piped Infrastructure, Data, and Governance in Norton Sound](https://openalex.org/W7128195115)
  - 이 군집에 속한 추출 용어: assessing | challenges | governance | infrastructure | supply | system | water

### 세부 주제 후보: 도시 우수·그린 인프라

- 동시출현: `stormwater` + `urban` = 189건
- 근거 제목 (2025, Water Research): [Characterizing sustainability of green stormwater infrastructure technologies using life cycle assessment and life cycle costing: A systematic review](https://openalex.org/W4414751804)
  - 이 군집에 속한 추출 용어: cycle | green | infrastructure | review | stormwater | sustainability | systematic
- 근거 제목 (2015, Journal of Water Resources Planning and Management): [Energy-Water Nexus: Potential Energy Savings and Implications for Sustainable Integrated Water Management in Urban Areas from Rainwater Harvesting and Gray-Water Reuse](https://openalex.org/W2132719911)
  - 이 군집에 속한 추출 용어: energy | harvesting | nexus | rainwater | sustainable | urban | water

### 세부 주제 후보: 관개용수·식량-에너지-물 넥서스

- 동시출현: `irrigation` + `water` = 313건
- 근거 제목 (2014, Water Resources Management): [Groundwater Use for Irrigation and its Productivity: Status and Opportunities for Crop Intensification for Food Security in Bangladesh](https://openalex.org/W1970392486)
  - 이 군집에 속한 추출 용어: crop | food | irrigation | opportunities | security
- 근거 제목 (2022, Water Resources Management): [Measurement of Resource Environmental Performance of Crop Planting Water Consumption Based on Water Footprint and Data Enveloped Analysis](https://openalex.org/W4206547731)
  - 이 군집에 속한 추출 용어: consumption | crop | environmental | footprint | planting | resource | water

### 세부 주제 후보: 물관리 거버넌스·정책 리뷰

- 동시출현: `risk` + `water` = 294건
- 동시출현: `review` + `water` = 251건
- 근거 제목 (2017, Water Resources Management): [Assessing the Governance Capacity of Cities to Address Challenges of Water, Waste, and Climate Change](https://openalex.org/W2620078724)
  - 이 군집에 속한 추출 용어: assessing | capacity | challenges | cities | governance | water
- 근거 제목 (2025, Water Research): [A critical review of sulfide and methane control in urban sewer systems using nitrogen compounds](https://openalex.org/W4407630399)
  - 이 군집에 속한 추출 용어: critical | review | sewer | system | urban

**판단:** 용어 'water'의 내부연결비중이 0.179로 8개 군집 중 가장 낮다 — 즉 'water'는 이 군집 밖과도 광범위하게 연결되는 범용어이며, 이 군집은 대상(도시/관개/공급/거버넌스)과 방법이 가장 혼재된 '포괄적 물관리' 우산 군집이다. 세분화 우선순위가 가장 높음

**해석의 한계/점검 사항:** review, assessing, perspective 같은 논문 장르성 용어가 상위 빈도에 다수 포함돼 있어, 주제적 응집이라기보다 '정책·리뷰형 논문'이라는 형식적 공통점이 섞여 있을 가능성이 있다.


## 군집 4: 지중유체(CO2) 저장·다공질매체 유동과 상수관망 센서

**빈도 상위 용어(계산값):** co(616건), pressure(524건), permeability(461건), fluid(337건), pore-scale(270건), saline(262건), sensor(212건), note(210건), geological(202건), technical(202건), leakage(183건), brine(127건), boltzmann(116건), sensors(107건), model-based(103건)

**내부 연결 비중 상위 용어(보조 지표):** cosmic(15건, 59.3%), ray(16건, 56.4%), sequestration(73건, 49.7%), geometries(20건, 43.8%), trapping(102건, 40.3%), neutron(64건, 39.1%), note(210건, 33.8%), boltzmann(116건, 32.0%), cosmic-ray(56건, 31.7%), technical(202건, 30.5%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: CO2 지중저장과 다공질매체 유동

- 동시출현: `co` + `sequestration` = 38건
- 동시출현: `permeability` + `pore-scale` = 20건
- 근거 제목 (2025, Advances in Water Resources): [Pore-scale relative permeability and saturation analysis under wide-ranging injection velocity and wettability during primary CO2 injection for geological carbon sequestration](https://openalex.org/W4408011597)
  - 이 군집에 속한 추출 용어: co | geological | permeability | pore-scale | sequestration
- 근거 제목 (2025, Advances in Water Resources): [Microfluidic study of CO2 diffusive leakage through microfractures in saline aquifers for CO2 sequestration](https://openalex.org/W4408891263)
  - 이 군집에 속한 추출 용어: co | leakage | saline | sequestration

### 세부 주제 후보: 상수관망 누수 탐지·센서 배치

- 동시출현: `leak` + `localization` = 22건
- 동시출현: `pressure` + `sensor` = 14건
- 근거 제목 (2025, Journal of Water Resources Planning and Management): [Pressure Sensor Placement in Water Distribution Systems Using Leak Resolution–Specific Surrogates for Leak Localization](https://openalex.org/W4410308745)
  - 이 군집에 속한 추출 용어: leak | localization | placement | pressure | sensor
- 근거 제목 (2025, Journal of Water Resources Planning and Management): [Pressure Sensor Placement for Pipe Burst Detection and Localization in Water Distribution System](https://openalex.org/W4410410981)
  - 이 군집에 속한 추출 용어: burst | localization | placement | pressure | sensor

### 세부 주제 후보: 우주선 중성자 기반 토양수분 관측

- 동시출현: `cosmic-ray` + `neutron` = 36건
- 근거 제목 (2024, Hydrology and earth system sciences): [Technical Note: Revisiting the general calibration of cosmic-ray neutron sensors to estimate soil water content](https://openalex.org/W4392272335)
  - 이 군집에 속한 추출 용어: cosmic-ray | neutron | note | sensors | technical
- 근거 제목 (2025, Hydrology and earth system sciences): [Data-driven scaling methods for soil moisture cosmic ray neutron sensors](https://openalex.org/W4411405743)
  - 이 군집에 속한 추출 용어: cosmic | neutron | ray | sensors

**판단:** 서로 다른 두 응용분야(CO2 지중저장의 다공질매체 유동 vs 상수관망 누수감지 센서)가 '압력(pressure)'이라는 공유 물리량과 '센서' 계측기술을 매개로 하나의 군집에 묶였다. 실제 연구공동체는 다를 가능성이 높아 세분화 검토 1순위

**해석의 한계/점검 사항:** 'note'+'technical' 동시출현(163건)은 'Technical Note'라는 논문 유형(article type) 표기가 제목에 그대로 들어간 경우가 많아 생긴 것으로 보이며, 실제 주제적 연결이 아닐 수 있다 — 이전 HESS Opinions 접두어 사례와 유사한 제목형식 인공물 가능성이 있어 추가 점검이 필요하다.


## 군집 5: 연안대수층 해수침투와 해수면 상승

**빈도 상위 용어(계산값):** coastal(1,197건), seawater(355건), intrusion(332건), salt(316건), saltwater(136건), rise(110건), tides(51건), sea-level(42건), tide(41건), mangrove(36건), tolerance(32건), offshore(27건), subterranean(25건), cutoff(22건), lab(18건)

**내부 연결 비중 상위 용어(보조 지표):** sea-level(42건, 48.0%), saltwater(136건, 46.6%), intrusion(332건, 43.4%), cutoff(22건, 33.3%), rise(110건, 31.0%), offshore(27건, 30.3%), seawater(355건, 25.6%), tides(51건, 22.2%), lab(18건, 20.0%), tide(41건, 15.8%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: 해수면 상승과 연안대수층 염수침투

- 동시출현: `intrusion` + `seawater` = 162건
- 동시출현: `rise` + `sea-level` = 37건
- 근거 제목 (2025, Journal of Hydrology): [Impact of inland boundary conditions on seawater intrusion modeling in stratified coastal aquifers under sea-level rise](https://openalex.org/W4413864249)
  - 이 군집에 속한 추출 용어: coastal | intrusion | rise | sea-level | seawater
- 근거 제목 (2020, Journal of Hydrology): [Assessment of the impact of sea-level rise on seawater intrusion in sloping confined coastal aquifers](https://openalex.org/W3012325324)
  - 이 군집에 속한 추출 용어: coastal | intrusion | rise | sea-level | seawater

### 세부 주제 후보: 조석·해안 지하수위 변동

- 동시출현: `coastal` + `tides` = 15건
- 근거 제목 (2019, Journal of Hydrology): [Combined effect of cutoff wall and tides on groundwater flow and salinity distribution in coastal unconfined aquifers](https://openalex.org/W2994134937)
  - 이 군집에 속한 추출 용어: coastal | cutoff | tides
- 근거 제목 (2018, Water Resources Research): [Analysis of Tide and Offshore Storm‐Induced Water Table Fluctuations for Structural Characterization of a Coastal Island Aquifer](https://openalex.org/W2793456952)
  - 이 군집에 속한 추출 용어: coastal | offshore | tide

**판단:** 해수침투·해수면상승이라는 뚜렷한 단일 주제로 응집된, 8개 군집 중 가장 주제적으로 선명한 군집. 이전(원시가중치) 버전에서 지하수 일반 논의에 섞여 있던 것과 달리 이번엔 완전히 독립된 군집으로 분리됨

**해석의 한계/점검 사항:** 규모가 작아(16개 용어, 2,159건) 다른 시드·해상도에서도 이 분리가 안정적으로 유지되는지는 추가 확인이 필요하다.


## 군집 6: (잠정 보류) 생지화학적 hot spot·건물 배관 수질 — 서로 다른 두 주제의 우연한 결합 가능성

**빈도 상위 용어(계산값):** hot(93건), plumbing(27건), spots(16건), premise(15건)

**내부 연결 비중 상위 용어(보조 지표):** spots(16건, 56.0%), premise(15건, 29.2%), plumbing(27건, 20.3%), hot(93건, 7.6%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: 건물 배관(premise plumbing) 수질·Legionella

- 동시출현: `plumbing` + `premise` = 14건
- 근거 제목 (2025, Journal of Water Resources Planning and Management): [Comparing the Hydraulic Performance of Cold-Water Premise Plumbing Systems Designed Based on Traditional and Modern Peak Demand Estimates](https://openalex.org/W4410641939)
  - 이 군집에 속한 추출 용어: plumbing | premise
- 근거 제목 (2024, Water Research): [Efficacy of chlorine-based disinfectants to control Legionella within premise plumbing systems](https://openalex.org/W4399020319)
  - 이 군집에 속한 추출 용어: plumbing | premise

### 세부 주제 후보: 생지화학적 hot spot·hot moment

- 동시출현: `hot` + `spots` = 14건
- 근거 제목 (2025, Hydrological Processes): [High‐Frequency Soil Redox Data Provide Novel Insights Into Biogeochemical Hot Spots and Hot Moments](https://openalex.org/W7116918939)
  - 이 군집에 속한 추출 용어: hot | spots
- 근거 제목 (2021, Hydrological Processes): [Evaluation of riverbed magnetic susceptibility for mapping biogeochemical hot spots in groundwater‐impacted rivers](https://openalex.org/W3153566715)
  - 이 군집에 속한 추출 용어: hot | spots

**판단:** 4개 용어 중 실제로 강하게 연결된 쌍은 plumbing+premise(14건)와 hot+spots(14건) 두 개뿐이며, 이 둘을 잇는 hot+plumbing 연결은 2건(저장 임계값의 경계)에 불과하다. 즉 서로 무관한 두 주제(생지화학적 hot spot/hot moment 개념과 건물 premise plumbing 수질)가 약한 다리 하나로 묶였을 가능성이 높다 — 하나의 군집으로 해석하지 않는 것이 안전

**해석의 한계/점검 사항:** 군집 크기가 작을수록(4개 용어) 이런 우연한 결합이 발생하기 쉽다. 전문가 검토 시 이 군집은 사실상 두 개의 별개 발견으로 보고하는 편을 권장한다.


## 군집 7: 탈질산 암모늄화(DNRA) 질소순환 경로

**빈도 상위 용어(계산값):** dissimilatory(24건), dnra(16건)

**내부 연결 비중 상위 용어(보조 지표):** dnra(16건, 40.0%), dissimilatory(24건, 24.2%)

등장 문서 수가 적은 용어는 연결이 한두 개만 있어도 내부 연결 비중이 쉽게 높게 나올 수 있다. 그러니 이 목록을 군집의 대표 용어 순위로 읽지 않는 것이 좋다.

### 세부 주제 후보: DNRA와 탈질의 경쟁·상호작용

- 동시출현: `dissimilatory` + `dnra` = 8건
- 근거 제목 (2024, Water Research): [Fimbriimonadales performed dissimilatory nitrate reduction to ammonium (DNRA) in an anammox reactor](https://openalex.org/W4403097909)
  - 이 군집에 속한 추출 용어: dissimilatory | dnra
- 근거 제목 (2023, Water Research): [Double-edged sword effects of dissimilatory nitrate reduction to ammonium (DNRA) bacteria on anammox bacteria performance in an MBR reactor](https://openalex.org/W4321350737)
  - 이 군집에 속한 추출 용어: dissimilatory | dnra

**판단:** 2개 용어(dissimilatory, dnra)로만 이뤄진 매우 작지만 주제적으로는 명확한 군집 — DNRA(탈질산 암모늄화)라는 구체적인 질소순환 미생물 경로를 가리키는 전문용어 쌍이 그대로 하나의 군집이 됨

**해석의 한계/점검 사항:** 군집이 작아(32건) 이것이 '부상하는 세부 연구주제'인지 단순히 특정 전문용어 표기 관행인지는 문서 수 추이(cluster_trends_fdr_assoc.parquet)를 함께 봐야 판단할 수 있다.

## 검토 결론과 다음 작업 제안

- FDR 보정(Rising 1,462→1,080개)과 co-occurrence 엣지 가중치의 association strength 정규화(Van Eck & Waltman 방식)를 함께 적용한 결과, modularity가 0.192에서 0.300으로 실측 개선되었고 군집 수는 5개에서 8개(대형 4개 + 소형 4개)로 늘었다. resolution 파라미터만 조정해서는(0.5~10.0 범위 실측) modularity가 오히려 최대 0.192에서 더 떨어졌다 — 개선은 엣지 가중치 정규화에서 왔다.
- 가장 뚜렷한 구조 변화는 연안대수층 해수침투(군집5, 16개 용어)가 지하수 일반 논의(군집1)에서 완전히 분리되어 독립 군집이 된 것이다. 허브 용어(coastal, groundwater 등)의 과잉연결을 정규화가 눌러 실제 하위주제를 드러낸 사례로 해석되지만, 표본 크기가 작아(2,159건) 안정성 확인이 필요하다.
- 반대로 CO2 지중저장(다공질매체 유동)은 지하수·연안 주제와 다시 묶이지 않고, 오히려 상수관망 누수감지 센서 기술과 새로 묶였다(군집4) — '압력/센서'라는 공유 계측기술이 매개가 된 것으로 보이나 실제 연구공동체는 다를 가능성이 높아 세분화 우선순위 1위다.
- 군집3(포괄적 물관리, water 내부연결비중 0.179로 8개 군집 중 최저)과 군집2(관측·예측·저수지운영이 뭉친 대형 군집, 278개 용어)는 여전히 대상과 방법이 혼재해 세분화 후보다.
- 군집6('hot'+'plumbing'+'spots'+'premise')은 실제로는 서로 무관한 두 주제가 동시출현 2건짜리 약한 다리로 우연히 묶인 것으로 보이며, 전문가 검토 시 두 개의 별개 발견으로 분리해 보고하는 편이 안전하다.
- 군집4의 'note'+'technical' 동시출현(163건)은 'Technical Note'라는 논문 유형(article type) 표기가 제목에 그대로 들어간 결과일 가능성이 있다 — 이전 HESS Opinions 접두어 사례와 같은 유형의 제목형식 인공물이므로 실제 주제 연결로 오인하지 않도록 확인이 필요하다.
- 이 결과는 Rising 용어들의 구조를 검토한 것이며, 8개 군집 각각의 연도별 문서비중 추이는 analysis/output_fdr_assoc/cluster_trends_fdr_assoc.parquet에 별도로 계산돼 있어 '부상'을 주장하려면 이 추이를 함께 제시해야 한다.
- 다음 단계: 이 문서를 수자원 전문가 2인에게 전달해 (1) 8개 잠정 이름과 세부 주제가 타당한지, (2) 군집2·3·4의 세분화가 실제로 필요한지, (3) 군집6을 두 개로 분리해야 하는지 검증받는다.

## 확인 파일

- [전체 용어와 계산 지표](terms.csv)
- [군집 내부·외부 연결 전체](edges.csv)
- [선정 논문 제목과 선정 방식](selected_papers.csv)
- [군집 용어가 등장한 전체 문서](all_papers.csv)
- [상위 단어·연결·제목 모음](evidence.md)
- [해석 초안의 구조화된 기록](interpretations.json)
- [입력 파일 해시와 실행 규모](manifest.json)

재현 명령: `C:\Users\user\anaconda3\envs\openalex\python.exe -m analysis.run_fdr_assoc_pipeline` 후 `python -m analysis.review_rising_clusters_fdr_assoc`