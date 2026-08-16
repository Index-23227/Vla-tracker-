# LIRA: Local Cross-Layer Information Routing for Vision-Language-Action Decoding

- arXiv: [2608.07596](https://arxiv.org/abs/2608.07596) (2026-08-06)
- 저자: Zhewei Zhang*, Puyue Wang*, Guanren Qiao*, Yijie Weng, Jiawei Hu, Guo Li, Lujia Wang, Junyan Wang, Tao Gu, Hongliang Lu†, Guiliang Liu, Hong Jia, Xinhu Zheng‡ (*동일 기여, †프로젝트 리드, ‡교신저자)
- 소속: v1 PDF에 소속 기관이 표기되어 있지 않음
- 코드: 논문 채택 시 코드와 체크포인트 공개 예정

## 1. 한 줄 요약

VLA에서 "VLM의 어느 층 표현을 액션 디코더로 흘려보낼 것인가"라는 **VLM-to-action 인터페이스**를 깊이 인지(depth-aware) 라우팅 문제로 재정의하고, 각 디코더 블록에 **자기 층 주변의 국소 윈도우**를 열어주는 파라미터 0 증가 기법(LIRA)을 제안한다.

## 2. 문제 정의

기존 VLA의 조건화 방식은 세 부류다. (a) 최종 층 특징만 노출(OpenVLA-OFT), (b) 디코더 블록 하나를 VLM 층 하나에 1:1로 고정(VLA-Adapter), (c) 전 층을 전역 집계. (a)는 계층 정보를 버리고, (b)는 인접 층의 상보적 증거에 접근하지 못하며, (c)는 깊이의 국소 구조를 뭉갠다. 논문의 가설은 "인접한 VLM 층들은 같은 디코더 깊이에 대해 서로 보완적인 태스크 증거를 제공한다"는 것이다.

## 3. 핵심 아이디어

VLA-Adapter를 베이스로, VLM 입력에 학습 가능한 **LIRA Query 토큰 M개**를 붙이고 층별 특징 R^(l)을 뽑는다. i번째 Parallel Fusion Block(PFB)에 대해

- W_r(i) = { l : |l − i| ≤ r }
- R̃^(i) = Concat_tok { R^(l) : l ∈ W_r(i) } (층 오름차순 토큰 축 결합)

기본값 r=1, 즉 자기 층 ±1의 3층 윈도우(경계에서는 2층으로 클리핑)를 쓴다. task-token 특징 T^(i)는 기존대로 깊이 정렬을 유지해 의미 앵커 역할을 하고, R̃^(i)가 국소 이웃의 보완 정보를 공급한다. 토큰 축 concat이라 기존 cross-attention이 그대로 소비할 수 있고 **추가 projection·학습 파라미터가 전혀 없다**.

## 4. 아키텍처

- 입력: 3인칭 이미지, 그리퍼 뷰 이미지, 언어 지시, 고유수용 상태(learnable projection φ_p)
- VLM: Prismatic 스타일 + Qwen2.5-0.5B 백본, LoRA rank 64
- 디코더: L개의 PFB, Action Query 토큰으로 초기화, 각 PFB는 Bridge Attention(self + cross)으로 T^(i), R̃^(i), E_p를 융합
- 출력: 마지막 PFB 뒤 output head가 horizon H의 **연속 액션 청크**를 회귀 (action_head_category = regression)
- 학습 손실: L_act = E_D[ (1/H) Σ_h ‖â_{t+h} − a_{t+h}‖₁ ] (L1)
- 내부 PFB는 3M개, 경계 PFB는 2M개의 LIRA Query 특징을 받음 (M=64)

## 5. 실험 설정

LIBERO(4 suite, suite당 500 trial), LIBERO-Plus(10,030개 섭동 태스크, 태스크당 1 rollout, 공식 프로토콜), CALVIN ABC→D(1,000 시퀀스), 실물 Franka 실험. 학습: H100 4장, lr 1e-4, batch 16, M=64, r=1. 통제 비교 대상은 동일 레시피의 VLA-Adapter이며 VLA-Adapter-Pro는 강한 참조군.

## 6. 주요 결과 — LIBERO (Table 2, %)

| Method | Params | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| VLA-Adapter | 0.5B | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| VLA-Adapter-Pro | 0.5B | 99.6 | 99.6 | 98.2 | 96.4 | 98.5 |
| **LIRA** | 0.5B | 99.6 | 99.8 | 98.5 | **97.6** | **98.9** |

통제 베이스라인 대비 평균 +1.6점, LIBERO-Long에서 95.0 → 97.6으로 가장 크게 개선된다(다단계 태스크에서 이득이 큼).

## 7. 주요 결과 — CALVIN ABC→D (Table 3)

| Method | 1 | 2 | 3 | 4 | 5 | Avg. len |
|---|---|---|---|---|---|---|
| OpenVLA-OFT (7B) | 96.9 | 92.0 | 85.7 | 80.4 | 72.9 | 4.28 |
| VLA-Adapter (0.5B) | **99.1** | 94.6 | 88.8 | 82.8 | 76.5 | 4.42 |
| VLA-Adapter-Pro | 98.5 | 95.0 | 90.5 | 85.3 | 80.0 | 4.50 |
| **LIRA** | 98.8 | **95.3** | **91.0** | **85.8** | **80.5** | **4.52** |

시퀀스가 길어질수록 격차가 벌어진다(5연속 76.5 → 80.5). 단일 태스크 성공률만 VLA-Adapter가 근소 우위.

## 8. 주요 결과 — LIBERO-Plus 강건성 (Table 4, %)

Zero-shot transfer:

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Overall** |
|---|---|---|---|---|---|---|---|---|
| OpenVLA (7B) | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT (7B) | 56.4 | 31.9 | 79.5 | 88.7 | **93.3** | 75.8 | 74.2 | 69.6 |
| VLA-Adapter (0.5B) | 36.2 | 37.9 | 74.6 | 70.6 | 76.1 | 58.0 | 69.7 | 59.1 |
| **LIRA (0.5B)** | **74.9** | **51.8** | **82.0** | **96.9** | 90.2 | **80.5** | **78.1** | **78.0** |

LIBERO-Plus 파인튜닝 후: LIRA 82.9 Overall (VLA-Adapter 81.0, OpenVLA-OFT 79.5). Overall은 7개 카테고리 단순 평균이 아니라 10,030 태스크 전체에 대한 태스크 가중 값임에 주의. **이 논문의 하이라이트는 여기다 — 동일 아키텍처에서 라우팅만 바꿔 zero-shot 강건성 +18.9점**.

## 9. 어블레이션

- **조건화 소스** (Fig. 3): task-token 단독, LIRA Query 단독보다 둘의 결합이 최고.
- **Query 예산** (Table 5, LIBERO-Long): last-layer-only는 256 토큰에서 최고 92.8, LIRA는 64 토큰으로 97.6. LIRA를 256으로 늘리면 93.0으로 오히려 하락 → 이득의 원천은 토큰 수가 아니라 국소 층간 라우팅.
- **라우팅 토폴로지** (Table 6, LIBERO-Long / CALVIN-5): Flashback 94.5/75.7, Column-wise 91.1/73.8, Q-Former 집계 89.2/78.4, **Local window 97.6/80.5**.
- **윈도우 폭** (Table 7): matched-only 95.4/77.9, 2층 96.2/79.3, **3층 97.6/80.5**, 4층 96.4/79.8, 전역 집계 93.7/76.9 → 국소성 사전지식(locality prior)이 검증됨.

## 10. 자원 프로파일 (Table 8)

| Metric | OpenVLA-OFT | LIRA | 상대 |
|---|---|---|---|
| 백본 크기 ↓ | 7B | 0.5B | 1/14× |
| 학습 메모리 ↓ | 62 GB | 12.8 GB | 0.21× |
| 처리량 ↑ | 71.4 Hz | 186.3 Hz | 2.61× |
| LIBERO 평균 ↑ | 97.1% | 98.9% | +1.8점 |

처리량은 8차원 액션 청크 기준, 학습 메모리는 batch size 8 기준. 백본과 학습 프로토콜이 달라 저자들도 "시스템 수준 비교"로 한정한다.

## 11. 실물 로봇 평가 (Table 9, 10회 시도)

Franka Research 3 (7-DoF) + Franka Hand, Orbbec Gemini 336L 2대, SpaceMouse 텔레오퍼레이션 시연.

| Task | OpenVLA-OFT | VLA-Adapter | LIRA |
|---|---|---|---|
| Move | 8/10 | 9/10 | **10/10** |
| Pick-and-Place | 6/10 | 7/10 | **9/10** |
| Collection (step1/step2) | 8/10 · 5/10 | 9/10 · 6/10 | **10/10 · 8/10** |
| Transfer (step1/step2) | 6/10 · 4/10 | 8/10 · 6/10 | 8/10 · **7/10** |

특히 서브골 간 정보 전달이 필요한 2단계 태스크에서 우위. 저자 스스로 시도 횟수 10회는 예비적(preliminary) 결과라고 명시.

## 12. 평가 및 한계

**강점.** (1) 변경 지점이 극도로 좁고(라우팅 윈도우 하나) 추가 학습 파라미터가 0이라 통제 실험으로서 설득력이 높다. (2) 표준 벤치마크 포화 구간(LIBERO 97→99)의 소폭 개선보다, LIBERO-Plus zero-shot +18.9점이라는 강건성 축의 큰 이동이 훨씬 흥미롭다. (3) 윈도우 폭 어블레이션이 1층·2층·3층·4층·전역을 모두 훑어 "국소성"이라는 주장 자체를 직접 검증한다. (4) 0.5B에 186.3 Hz로 배포 친화적.

**한계.** (1) 검증이 Qwen2.5-0.5B 기반 Prismatic VLM 한 종류, PFB 기반 디코더 한 종류에 국한되어 다른 백본/디코더로의 이식성은 미검증(저자도 명시). (2) 왜 국소 윈도우가 특히 분포 이동에서 크게 이득인지에 대한 메커니즘 분석(층별 표현 유사도, 어텐션 분포 등)은 제시되지 않고 성능 표로만 논증한다. (3) 실물 평가 태스크 4개·시도 10회로 통계적 결론을 내기 어렵다. (4) 토큰 축 concat이므로 내부 PFB의 cross-attention 키/값 길이가 3배가 되는데, 이 비용이 처리량에 미치는 영향을 VLA-Adapter와 직접 비교한 수치는 없다(비교는 OpenVLA-OFT 상대로만 제시). (5) 접촉이 많거나 힘 제어가 필요한 조작, 크로스 임바디먼트는 다루지 않는다.

**한 줄 코멘트.** 새 손실도 새 모듈도 없이 "어느 층을 볼지"만 바꿔 강건성 벤치마크를 크게 움직인, 인터페이스 설계 축의 깔끔한 ablation 논문. VLA-Adapter 계열을 쓰고 있다면 즉시 시도해볼 만한 변경이다.

<!-- VERIFIED: pdf -->
