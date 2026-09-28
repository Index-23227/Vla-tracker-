# What Makes an Efficient VLA? Navigating Action-Head Design, Scaling, and Latency (EffVLA)

> **한 줄 요약**: SigLIP2 + Qwen2.5 백본과 DROID→LIBERO 파이프라인을 고정하고 63개 셀을 학습해, action head의 네 축(디코더·손실·초기화·추론 패스)과 세 모듈 규모를 RTX 5090 지연과 짝지어 하나씩 분리한 통제 연구. 가장 큰 레버는 구조가 아니라 **"언어 백본 마지막 층을 head로 복사하는 초기화(VLM-INIT)"**이며, 이 규칙들로 고른 3.75B 모델 **EffVLA**는 LIBERO 98.2%, LIBERO-Plus 79.8%(7축 중 6축 1위), 청크당 39.2 ms.

- **arXiv**: 2609.13984 (PDF 스탬프: v2, 2026-09-23, cs.RO)
- **소속**: 중국과학원 자동화연구소(CASIA), UCAS, AI Lab The Yangtze River Delta, Li Auto, BUPT, UCL, University of Edinburgh
- **학회**: Not stated in the paper
- **코드**: https://github.com/MindVLA-Team/EFFVLA
- **백본**: SigLIP2-So400m(256 px) + Qwen2.5-3B, action head ≈0.35B

---

## 1. 배경 및 동기

VLA는 비전 인코더·언어 백본·action head의 조합인데, π0(flow-matching expert), OpenVLA-OFT(MLP 회귀), OpenVLA/FAST(자기회귀 토큰) 등 각 시스템이 서로 다른 백본·데이터·벤치마크에서 성능을 보고해 **어떤 설계 선택이 얼마나 기여하는지** 통제된 비교가 없었다. 또한 실시간 제어에서는 지연이 정확도만큼 중요하지만 거의 보고되지 않는다. 저자들은 설계 선택마다 동일 하드웨어 지연을 짝지은 통제 실험으로 이 공백을 메우려 한다.

## 2. 문제 정의

- 고정: 백본 계열(SigLIP2, Qwen2.5), 파이프라인(DROID로 action head 사전학습 → LIBERO에서 전 모듈 100k step 파인튜닝), 평가(LIBERO-Plus 4 suite × 7 섭동축, 셀당 약 1000 rollout), 지연(RTX 5090, bf16, batch 1, 청크 8, torch.compile, 100회 평균).
- 변수: action head의 네 축 — **디코더**(MLP / KV-share 트랜스포머 블록 / AR 트렁크), **손실**(L1 / flow matching / NTP), **초기화**(random / VLM-INIT), **추론 패스**(1 / 4) — 과 모듈 규모 V, L, A ∈ {s, m, l, g}.
- 목표: "무엇을 만들지, 어디를 키울지, 얼마나 크게 할지"에 대한 규칙과 그 규칙이 선택하는 모델.

## 3. 방법: 여덟 가지 substrate

| Substrate | 디코더 | 손실 | 초기화 | 패스 | LIBERO-Plus |
|---|---|---|---|---|---|
| FAST | AR 트렁크 | NTP | – | 다수 | 57.7 |
| OFT | MLP | L1 | – | 1 | 79.0 |
| OFT-TF | TF 블록 | L1 | random | 1 | 72.7 |
| OFT-TF-R4 | TF 블록 | L1 | random | 4 | 74.8 |
| **EffVLA** (OFT-TF-VLMINIT) | TF 블록 | L1 | VLM-INIT | 1 | **79.8** |
| OFT-TF-VLMINIT-R4 | TF 블록 | L1 | VLM-INIT | 4 | 79.6 |
| π0 | DiT 블록 | FM | random | 4 | 77.4 |
| π0-vlminit | TF 블록 | FM | VLM-INIT | 4 | 75.2 |

- VLM-INIT: Qwen2.5-3B 마지막 D개 층의 텐서를 head의 D개 층에 그대로 복사(hidden·head 수 일치). L1 쌍에서는 random과 VLM-INIT이 **구조가 완전히 같고 가중치만 다르다**.
- KV-share: 트랜스포머 블록 head가 언어 백본의 KV 캐시를 직접 읽어 중복 prefill이 없다.
- EffVLA = 4층 트랜스포머 head(Qwen2.5-3B 마지막 4층 복사) + 단일 패스 L1, 총 ≈3.75B.

## 4. 세 가지 규칙(Findings)

1. **무엇을 만들지 — 초기화 우선, 단순하게**: random→VLM-INIT이 +7.1pt(지연 비용 없음). VLM-INIT 이후에는 MLP→TF +0.8(판별 불가), 4패스 −0.2, L1→FM −4.4. 반대로 random init에서는 FM +2.6, 4패스 +2.1 → **추가 표현력은 초기화가 없을 때만 보상적으로 작동**. 10개 L1 초기화 매칭 쌍 모두 양수(평균 +5.8).
2. **어디를 키울지 — head**: VLM-INIT 하에서 head s→l +9.4pt(2.4 ms), random-init L1 TF는 +3.1. 지연 1 ms당 head ≈4pt, 비전 ≈1pt, 언어 ≈0.15pt.
3. **얼마나 크게 — l에서 멈춤**: head 650M 78.3, 1B 비전 78.9, 7B 언어 77.8로 모두 l(79.8) 이하 → ≈3.75B(π 시리즈 규모)에서 포화.

## 5. 실험 설정

- 셀당 단일 seed(앵커 EffVLA–OFT 쌍만 5 seed), LIBERO-Plus 셀당 ≈1000 rollout(이항 표준오차 ≈1.4pt).
- 교차 벤치마크: SimplerEnv Bridge(WidowX) 4과제 × 24 trial.
- 실물: SO-ARM101 6-DOF 저가 팔, Stage A는 커뮤니티 공개 코퍼스, Stage B는 5개 분류 과제 텔레옵 시연 병합, 과제당 10 trial.

## 6. 주요 결과

**Table 4 – 공개 기준선 비교 (기준선 수치는 ABot-M0 논문에서 인용)**

| Method | LIBERO Avg | BG | Init | Cam | Lang | Noise | Layout | Light |
|---|---|---|---|---|---|---|---|---|
| π0 | 94.4 | 81.4 | 6.0 | 13.8 | 58.8 | 79.0 | 68.9 | 85.0 |
| OpenVLA-OFT | 97.1 | 93.3 | 31.9 | 56.4 | 79.5 | 75.8 | 74.2 | 88.7 |
| RIPT-VLA | 97.5 | 91.6 | 31.2 | 55.2 | 77.6 | 73.5 | 74.2 | 88.4 |
| ABot-M0 | 98.6 | 91.6 | 67.9 | 60.4 | 86.4 | 86.4 | 82.6 | 96.2 |
| **EffVLA** | 98.2 | 96.7 | 69.8 | 68.3 | 87.5 | 66.1 | 84.8 | 97.4 |

- LIBERO 세부: Spatial 98.6 / Object 99.2 / Goal 99.0 / Long 96.0.
- LIBERO-Plus 7축 중 6축 1위, 카메라 +7.9(vs ABot-M0), 로봇 초기 상태 +37.9(vs OpenVLA-OFT). 센서 노이즈만 약함(66.1).
- 지연: EffVLA 39.2 ms, OFT 40.1, π0 53.7, FAST ≈195 ms.
- SimplerEnv Bridge: EffVLA 61.5, π0 60.5, OFT 59.4, random-init TF 54.2 (초기화 효과 +7.3 재현).
- SO-ARM101: 40/50 (1개 물체 10/10·9/10, 2개 9/10·7/10, 3개 5/10).

## 7. 절제 및 분석

- **메커니즘 증거(상관적)**: 가중치 공간 CKA — 초기화 head는 학습 중 백본과의 유사도가 약 0.88→0.76, random은 0.27→0.24. 어텐션 — 초기화 head가 지시문 토큰에 약 4배 많은 질량(예시 롤아웃 0.223 vs 0.030). 저자는 이를 "호환성(compatibility)" 해석으로 제시하되 인과는 입증되지 않았다고 명시.
- **DROID 사전학습 제거**: L1 계열은 1–6pt 하락에 그치지만 flow matching π0는 15–22pt 하락(l/l/l: 77.4→55.8) → L1 계열이 짧은 Stage B만으로 새 embodiment에 이식 가능한 근거.
- **5-seed 검증**: EffVLA 79.0±1.2 vs OFT 77.9±1.5, 5 seed 모두 EffVLA 우위(+0.5~+1.7).
- 레시피 이탈 비용(Table 10): random init −7.1, FM −2.4(+14.5 ms), MLP −0.8, head s −9.4, 언어 s −1.7(−11.4 ms), 비전 s −3.4.

## 8. Related Work 상의 위치

- RoboVLMs, OpenVLA-OFT의 설계 절제, TinyVLA의 소형화 연구, 동시대 VLANeXt·StarVLA-α와 같은 계열이지만, **action head 초기화를 독립 축으로 분리**하고 **모든 선택에 실측 지연을 짝지은** 점이 차별점.
- "파인튜닝이 사전학습 특징을 왜곡한다"(Kumar et al.), surgical fine-tuning(Lee et al.)의 관점을 VLA head 초기화로 연결.
- EfficientNet/MnasNet식 하드웨어 인식 설계를 모듈형 VLA로 가져옴.

## 9. 강점

1. **통제의 엄밀성**: 구조 동일·가중치만 다른 random vs VLM-INIT 쌍, 축별 분리, 10개 매칭 쌍의 방향 일관성, 5-seed 앵커 검증.
2. **실무적 규칙**: "초기화 → head 확장 → 3.75B에서 멈춤"이라는 명확한 가이드와 레시피 이탈 비용표.
3. **정직한 해석**: CKA가 activation 수준 정렬이 아님, 인과 미입증, 실물은 비교가 아닌 타당성 결과임을 반복해서 명시.
4. 단순한 단일 패스 L1 head로 LIBERO 상위권·LIBERO-Plus 대부분 축 1위를 더 낮은 지연으로 달성.

## 10. 약점 및 한계

1. 대부분 셀이 **단일 seed**이고 차이가 1–2pt인 경우가 많아, 작은 격차는 판별 불가(저자도 인정).
2. LIBERO-Plus는 단기 시뮬레이션 — 포화 지점(Finding 3)이 장기·고정밀·모바일 과제에서 유지될지 불명. 센서 노이즈 축에서는 MLP head가 더 강함.
3. Table 4 기준선 수치는 ABot-M0 논문에서 가져온 것으로, 동일 파이프라인 재현이 아니다.
4. flow matching의 VLM-INIT 비교는 블록 타입(DiT vs plain)도 바뀌어 순수 초기화 효과가 아니며, 디노이징 스텝 수는 4로 고정.
5. KV-share 인터페이스를 고정했으므로 다른 VLM–정책 연결 방식에서도 규칙이 유지되는지 미검증. SimplerEnv 재현은 96 trial로 통계력이 낮고 학습 데이터 세부가 부족.
6. 실물 실험은 저가 단일 팔 5과제, 다른 head와의 하드웨어 비교 없음.

## 11. 재현 및 확장 아이디어

- EffVLA vs 구조 동일 OFT-TF(random) 쌍의 다중 seed 대응 비교(저자가 제시한 우선 후속 실험).
- RoboCasa·장기 모바일 조작 등 더 어려운 벤치마크에서 포화 지점 재측정.
- 복사할 층의 위치/개수, 부분 복사, activation-level CKA로 호환성 가설 검증.
- 다른 연결 방식(cross-attention, prefix hidden-state 읽기)과 다른 백본(PaliGemma, Qwen3-VL)으로 규칙 일반화.

## 12. 총평

"VLA action head는 무엇이 좌우하는가"에 대해 가장 통제된 답 중 하나를 제공한다: 복잡한 디코더·flow matching·추가 패스보다 **백본과 호환되는 초기화**가 결정적이고, 그 뒤에야 head 용량 확장이 보상된다. 결과물인 EffVLA는 단일 패스 L1 head라는 단순함으로 LIBERO 98.2, LIBERO-Plus 79.8을 39 ms에 달성한다. 단일 seed 셀과 단기 벤치마크라는 한계는 있지만, VLA 설계에서 "표현력 추가" 관성에 반례를 제시한 유용한 연구다.

**한 문장 요약**: 63셀 통제·지연 짝 실험으로 "head는 백본 마지막 층으로 초기화하고, 단순 L1로, head를 키우되 3.75B에서 멈춰라"를 도출하고, 그 레시피 EffVLA로 LIBERO 98.2 / LIBERO-Plus 79.8 / 39.2 ms를 보인 연구.

<!-- VERIFIED: pdf -->
