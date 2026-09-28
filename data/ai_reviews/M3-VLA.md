# Robust Bimanual Vision-Language-Action Models via Embarrassingly Simple Modality Masking (M3)

> **한 줄 요약**: 쿼리 기반 VLA(VLA-Adapter, 0.5B)를 학습할 때 어텐션 마스크로 **두 손목 뷰를 함께**, **언어 전체를**, **액션 쿼리 일부를** 확률적으로 가리는(ego 뷰는 항상 유지) 학습 전용 기법 M3. 구조·추론 변경 없이 RoboTwin 2.0 10개 과제 Clean 평균 41.0 → 62.7, Clean2Rand 3.7 → 15.1, 실물 Agilex 3개 장기 과제 full-task 44.4 → 69.4(clean) / 12.5 → 61.1(OOD).

- **arXiv**: 2608.22419v1 (2026-08-23, cs.RO)
- **소속**: Southeast University, Shanghai Innovation Institute 외 (Wuhan Univ., Zhejiang Univ., USTC, SJTU, CityU HK)
- **프로젝트**: m3vla.github.io
- **백본**: Prismatic VLM(DINOv2 + SigLIP) + Qwen2.5-0.5B, VLA-Adapter식 action expert

---

## 1. 배경 및 동기

쿼리 기반 VLA(OpenVLA-OFT, VLA-Adapter)는 학습 가능한 액션 쿼리로 action chunk를 한 번의 forward로 디코딩해 지연이 낮아 양팔 제어에 매력적이다. 그러나 저자들은 복잡한 양팔 과제에서 행동이 불연속적으로 튀고 실패하는 현상을 관찰했고, 이때 어텐션이 과제와 무관한 영역(특히 쉬고 있는 팔의 손목 카메라 속 두드러진 물체)으로 퍼져 있음을 발견했다. 모든 뷰가 항상 보이는 학습에서는 뷰 간 허위 상관(spurious cross-view correlation)을 학습해 분포 이동에 취약해진다는 가설이다.

## 2. 문제 정의

- 입력: K개 카메라 시각 토큰 V_t, 언어 토큰 T, N_q개 액션 쿼리 Q.
- 출력: H-step action chunk A_t = g_θ(V_t, T, Q), L1 회귀로 학습.
- 목표: 아키텍처·추론을 바꾸지 않고, 부분 관측 하 학습으로 어떤 단서가 섭동 하에서도 신뢰 가능한지 스스로 고르게 만드는 것.

## 3. 방법: Modality Masking Mechanism (M3)

세 가지 양팔 특화 설계 원칙:
1. **Ego 뷰는 항상 유지** — 안정적 공간 기준. 실제로 ego를 가리면 37.0%로 크게 떨어진다(Table 3a).
2. **두 손목 뷰는 함께 가림** — 한쪽만 가리면 남은 손목 뷰와 ego 사이의 허위 매칭이 여전히 가능. u_v ~ Bernoulli(1−p_v).
3. **액션 쿼리 부분집합 마스킹** — dropout처럼 쿼리 간 공적응을 막아 상보적 표현을 유도. 최소 1개는 유지, 남은 쿼리는 1/(1−p_q)로 재스케일.
- 언어도 u_ℓ ~ Bernoulli(1−p_ℓ)로 통째로 가림.
- 구현: 토큰 가시성 벡터 m으로 만든 가산 마스크(보이는 토큰끼리만 attend)를 causal mask와 합산. 임베딩 자체는 건드리지 않는다.
- 손실은 기존 L1 그대로, 입력만 M3로 마스킹.

## 4. 구현 세부

- VLM: Prismatic(DINOv2+SigLIP) + Qwen2.5-0.5B, action expert는 VLA-Adapter의 bridge attention 구조, 고유수용 상태는 기본 key/value.
- AdamW, lr 2e-4, 10k step(5k에서 ×0.1), LoRA rank 64, 호라이즌별로 동일 하이퍼파라미터.
- action chunk 25 step(RoboTwin 2.0/OpenVLA-OFT 관례).
- 최적 쿼리 마스크 비율 0.1.
- 시뮬레이션은 SAPIEN 렌더링 때문에 RTX 4090 사용(OpenVLA-OFT 7B는 24GB 한계로 OOM 빈발).

## 5. 실험 설정

- **RoboTwin 2.0**: SimpleVLA-RL을 따라 10개 과제(단/중/장기 호라이즌 분류), 과제당 clean 시연 50개로 과제별 학습, 100개 held-out 장면 평가. Clean(깨끗한 장면)과 Clean2Rand(깨끗한 데이터로 학습, 배경·조명·방해물 무작위 장면 평가) 두 설정.
- **기준선**: RDT-1B*, π0*(대규모 사전학습), ACT, DP, VLA-Adapter(동일 백본·예산), 부록에서 OpenVLA-OFT.
- **실물**: Agilex Cobot, 3개 장기 과제(병 치우기+팔간 전달, 그릇 쌓고 선반 올리기, 채소 담고 접시 가운데 맞추기), 각 800 제어 step 이상, 과제당 시연 50개. 3라운드 × (clean 16 + OOD 8) trial.

## 6. 주요 결과 – 시뮬레이션

**Table 1 – RoboTwin 2.0 Clean (%)**

| 과제 | RDT* | π0* | ACT | DP | Adapter | **M3** |
|---|---|---|---|---|---|---|
| Click Bell | 80 | 44 | 58 | 54 | 84 | **97** |
| Grab Roller | 74 | 96 | 94 | 98 | 88 | 96 |
| Place Phone Stand | 15 | 35 | 2 | 13 | 10 | **55** |
| Place Bread Basket | 10 | 17 | 6 | 14 | 11 | **22** |
| Place A2B Right | 1 | 27 | 0 | 13 | 4 | **28** |
| Place Shoe | 35 | 28 | 5 | 23 | 34 | **63** |
| Stack Blocks Two | 21 | 42 | 25 | 7 | 78 | **83** |
| Handover Block | 45 | 45 | 42 | 10 | 27 | **74** |
| Put Bottles Dustbin | 21 | 54 | 27 | 22 | 60 | **81** |
| Block Rank Size | 0 | 7 | 0 | 1 | 14 | **28** |
| **평균** | 30.2 | 39.5 | 25.9 | 25.5 | 41.0 | **62.7** |

**Table 2 – Clean2Rand 평균**: RDT 8.8, π0 12.9, ACT 2.9, DP 0.0, Adapter 3.7, **M3 15.1**. Put Bottles Dustbin +38, Grab Roller +26가 가장 큰 향상, Block Rank Size는 모든 방법이 0 근처.

**부록 Table 8 – OpenVLA-OFT로 이식**: 평균 32.2 → 53.5(+21.3).

## 7. 주요 결과 – 실물 및 절제

**부록 Table 9 – 실물 full-task 성공률 (%)**

| 과제 | Adapter clean | M3 clean | Adapter OOD | M3 OOD |
|---|---|---|---|---|
| Bottle Cleanup | 41.7 | **66.7** | 16.7 | **58.3** |
| Stack & Shelf | 27.1 | **58.3** | 12.5 | **54.2** |
| Veggie Centering | 64.6 | **83.3** | 8.3 | **70.8** |
| 평균(본문) | 44.4 | **69.4** | 12.5 | **61.1** |

**절제 (Place Phone Stand / Place Shoe / Handover Block 평균)**
- 뷰 마스킹: ego 가림 37.0, 한쪽 손목 32.7, 양 손목 64.0.
- 쿼리 마스크 비율: 0.0(Adapter) 23.7 → 0.1에서 64.0 최고 → 0.9에서 43.7.
- 구성요소: L 29.0, V 34.0, Q 33.3, V+L 36.7, V+Q 60.3, V+L+Q 64.0 → **시각+쿼리 결합**이 이득의 대부분.
- 범용 정규화 대비: token dropout 31.8, modality dropout 24.1, visual aug 22.3, region aug 23.0 — 구조화되지 않은 마스킹/증강으로는 재현되지 않음.

## 8. Related Work 상의 위치

- ModDrop, Sensor Dropout 등 멀티모달 드롭아웃 계열이지만, **양팔·다중 뷰 구조를 반영한 규칙(ego 고정, 손목 공동 마스킹, 쿼리 부분 마스킹)**이 핵심 차별점이며, 일반 modality dropout은 실험적으로 효과가 없었다.
- RDT-1B, π0처럼 대규모 사전학습으로 도메인 격차를 줄이는 대신, 과제별 clean 데이터만으로 Clean2Rand에서 그 이상을 달성.
- OTTER, ReconVLA, DTP 등 어텐션 품질 관점의 VLA 분석과 맥을 같이 한다.

## 9. 강점

1. **극도로 단순**: 추가 파라미터·추론 비용 0, 어텐션 마스크 몇 줄로 구현.
2. **절제가 설계 원칙을 직접 검증**: ego 고정, 손목 공동 마스킹, 쿼리 비율, 범용 드롭아웃 대비 모두 수치로 비교.
3. **백본 이식성**: VLA-Adapter와 구조가 다른 OpenVLA-OFT에서도 +21.3.
4. **실물 OOD 강건성**: 방해물 조건에서 Adapter 12.5 → 61.1로 격차가 특히 크며, 단계별(phase) 성공률까지 공개.

## 10. 약점 및 한계

1. **절대 수준의 강건성은 여전히 낮다**: Clean2Rand 15.1%는 "격차를 줄였을 뿐 닫지 못했다"(저자 표현). 10개 중 여러 과제가 한 자릿수.
2. **시뮬레이션은 단일 실행**: 공식 프로토콜을 따른다고 하나 seed 분산이 없어, 일부 과제의 작은 차이는 해석이 어렵다.
3. **기준선의 공정성**: RDT*/π0*는 논문이 직접 돌린 수치인지 인용값인지, 같은 예산 조건인지가 본문에서 명확하지 않다. Adapter 기준선 자체가 Clean2Rand에서 3.7%로 매우 약해 상대 이득이 부풀려 보일 수 있다.
4. **분석은 상관적**: 어텐션 확산과 불연속 행동의 연관은 post-hoc 통계로, 인과를 보이진 않는다.
5. 회귀(L1) 쿼리 기반 헤드에만 검증 — diffusion/flow 헤드에서의 효과는 미래 과제로 남김.
6. 과제별(task-specific) 학습만 다뤄 멀티태스크·대규모 사전학습 단계에서의 효과는 미확인.

## 11. 재현 및 확장 아이디어

- π0/π0.5, GR00T 같은 flow-matching VLA의 VLM prefix에 동일 마스킹 적용.
- 마스킹 확률을 과제 단계(접근/전달/배치)나 불확실성에 따라 적응적으로 조절.
- 멀티태스크 사전학습 단계에서의 M3 효과, domain randomized 데이터와의 결합.
- 깊이·촉각 등 추가 모달리티로 원칙("안정적 기준 모달리티는 유지, 상관된 국소 모달리티는 공동 마스킹") 일반화.

## 12. 총평

"양팔 다중 뷰 VLA가 쉬는 팔의 손목 카메라에 속는다"는 구체적 실패 모드를 짚고, 그에 맞춘 구조화된 마스킹 규칙 세 가지로 추가 비용 없이 큰 이득을 얻었다. 범용 dropout/augmentation이 효과가 없다는 대조 실험이 특히 설득력 있다. 다만 약한 기준선 위에서의 상대 향상, 단일 실행, 여전히 낮은 Clean2Rand 절대치는 한계로 남는다.

**한 문장 요약**: ego 뷰 고정·양 손목 공동 마스킹·액션 쿼리 부분 마스킹이라는 학습 전용 트릭만으로 0.5B 쿼리 기반 VLA의 RoboTwin 2.0 평균을 41.0 → 62.7, 실물 OOD를 12.5 → 61.1로 끌어올린 실용적 정규화 기법.

<!-- VERIFIED: pdf -->
