# LEEVLA: Seeing What Matters in Latent Environment Evolution for Vision-Language-Action

**arXiv**: [2607.08182](https://arxiv.org/abs/2607.08182) · **발표일**: 2026-07-09 · **소속**: 중국과학원 선양자동화연구소(SIA, CAS), 중국과학원대학(UCAS), MBZUAI
**저자**: Qi Lyu, Baicheng Liu, Xudong Wang, Jiahua Dong, Lianqing Liu, Zhi Han
**코드**: https://github.com/LyuQi127/LEEVLA

---

## 1. 문제 정의 (Problem Statement)

기존 VLA 모델은 두 가지 구조적 한계를 갖는다. (1) 모든 시각 토큰을 균일하게 취급하여 학습 시 감독 신호(gradient)가 정적 배경이나 과제와 무관한 물체로 희석된다. (2) 서브골 이미지, 세그멘테이션, 깊이 등 사람이 선택한(human-selected) 특정 컨텍스트 단서에 기반한 명시적 추론은 탐색 공간을 제한하고, 보조 모델의 성능에 종속되며, 알려지지 않았지만 과제에 중요한(unknown but task-relevant) 요인을 놓친다. LEEVLA는 "어디를 볼 것인가(where)"와 "잠재 표현이 어떻게 진화할 것인가(how)"를 학습 시점에만 명시적으로 지도하는 프레임워크를 제안한다.

## 2. 핵심 기여 (Key Contributions)

1. **DGDP (Drift-Guided Dynamic Prioritization)**: 동적 위치 우선화(DPP)와 의미 드리프트 가이던스(SDG)를 결합하여, 동적으로 활성화되고 명령어와 관련된 영역을 자동 발견 — "어디를 볼 것인가"를 지도.
2. **SFFG (Structured Feature Flow Generation)**: 프로토타입-주변부(P2P) 예측과 상호 이웃 대조(MC) 손실로 잠재 공간에서의 구조화된 특징 진화를 학습 — "어떻게 진화할 것인가"를 지도.
3. LIBERO와 CALVIN에서 SOTA 달성. DGDP/SFFG는 학습 시에만 사용되어 추론 비용 증가가 전혀 없음.

## 3. 방법론 (Methodology)

**전체 구조**: 3인칭 + 손목 카메라 이미지를 DINOv2 + SigLIP 비전 인코더로 패치 특징화하고, 프로젝터를 거쳐 LLM에 입력한다. 병렬 연속 액션 예측기(MLP 헤드)가 LLM 마지막 레이어 hidden state로부터 연속 액션 청크 a_{1:T}를 직접 회귀한다. 미래 특징 디코더 D가 시점 t의 임베딩으로부터 t+T 시점의 특징을 예측한다.

**DGDP**:
- *DPP*: 패치별 동적 점수 θ_i = 1 − cos(v_{t,i}, v_{t+T,i}) — 시간에 따른 특징 변화가 큰 영역에 높은 가중치 (식 1).
- *SDG*: 언어 토큰과의 최대 내적으로 명령 관련도 r_{t,i}를 구하고 (식 2), 드리프트 Δ_i = clip((r_{t+T,i} − r_{t,i})/τ, −δ, δ)를 [-1,1]로 정규화한 후 ω_i = exp(Δ̃_i/τ)로 변환 (식 3-4).
- 최종 우선화 가중치 β_i = σ(ω_i · θ_i) (식 5, σ는 sigmoid). 배경으로 의미가 드리프트하는 패치는 억제, 명령 관련 방향으로 진화하는 동적 패치는 강조.

**SFFG**:
- *P2P 예측*: 다중 뷰 미래 특징을 클러스터링(식 6)한 후, 각 클러스터 내에서 중심(프로토타입)에서 주변부 순으로 정렬(식 7)하여 코사인 임베딩 손실로 예측 (식 8). 전역 변조 인자 α=1로 배경의 약한 기여를 보존. 좌상→우하 평탄(flat) 토큰 예측이 의미 단위를 분절시키는 문제를 해결.
- *MC 손실*: 1차(K=10) + 2차(M=5) k-NN 이웃 중 상호(reciprocal) 이웃만 양성으로 채택하여 InfoNCE 대조 손실 적용 (식 9-10). 비대칭·허위 이웃 링크를 필터링해 잠재 공간 위상 일관성 유지.

**학습 목표**: L_total = λ1·L_action(L1) + λ2·L_P2P + λ3·L_MC (식 12).

## 4. 아키텍처 상세 (Architecture Details)

- **LEEVLA-large (7B)**: OpenVLA-7B 초기화 (Llama-2-7B + DINOv2/SigLIP), Open X-Embodiment 혼합 데이터로 추가 사전학습, 50k-150k 스텝 학습, LR 5e-4.
- **LEEVLA-mini (0.5B)**: LIBERO-90으로 사전학습된 miniVLA 초기화, 20k-50k 스텝, LR 2e-5.
- AdamW 옵티마이저, 8× A100 80GB.
- 액션 헤드: OpenVLA-OFT 스타일의 병렬 디코딩 + L1 회귀 MLP (연속 액션 청크).
- 추론 시 토큰 스트림 불변 — DGDP/SFFG는 학습 전용 보조 감독.

## 5. 실험 설정 (Experimental Setup)

- **LIBERO**: Spatial/Object/Goal/Long 4개 스위트, 명령 10개 × 50 에피소드 × 3 시드로 성공률 측정.
- **CALVIN ABC-D**: 연속 5개 과제 성공률과 평균 성공 길이(Avg. Len).
- **실세계**: UR5 6-DoF 로봇팔, 3개 과제(놓기, 버튼 누르기, 서랍 닫기), 각 설정 20회 연속 시행.
- 비교군: Octo, UniACT, Seer, DreamVLA, FLOWER (소형); OpenVLA, CoT-VLA, π0, π0.5, OpenVLA-OFT, UniVLA, MemoryVLA (대형).

## 6. 주요 결과 (Main Results)

**LIBERO (Table 2)**:
| 모델 | Spatial | Object | Goal | Long | 평균 |
|---|---|---|---|---|---|
| OpenVLA-OFT (7B) | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0.5 (3B) | 97.0 | 99.0 | 98.0 | 96.0 | 97.5 |
| **LEEVLA-large (7B)** | **98.8** | **99.0** | **98.6** | **96.4** | **98.2** |
| FLOWER (1B) | 97.1 | 96.7 | 95.6 | 93.5 | 95.7 |
| **LEEVLA-mini (0.5B)** | **98.6** | **99.0** | **97.0** | **95.5** | **97.5** |

**CALVIN ABC-D (Table 1)**: LEEVLA-large 평균 길이 **4.34** (과제 1-5: 98.8/94.5/87.3/80.6/72.7) — OpenVLA-OFT 4.10, π0.5 4.02 대비 우위.

**실세계 (Table 3)**: 평균 성공률 78.5% vs OpenVLA 40% (Place 70/30, Press 80/55, Drawer 65/40, Long 60/35).

## 7. 어블레이션 분석 (Ablation Study)

LIBERO-Goal에서 LEEVLA-mini로 점진적 어블레이션 (Table 5): 베이스라인 94.8% → +P2P 95.2% (+0.4) → +MC 95.6% (+0.8) → +DPP 96.3% (+1.5, 최대 단일 기여) → +SDG **97.0%**. SFFG(P2P+MC)와 DGDP(DPP+SDG)는 상호 보완적.
- 특징 재정렬 효과 (Table 6): reorder 미적용 94.7 → 적용 95.2.
- 전역 인자 α 효과 (Table 7): α 미사용 95.8 → 사용 97.0 — β만 쓰면 배경이 과도하게 버려져 공간 문맥 상실.
- 복잡도 (Table 4): 학습 시 추가 오버헤드는 P2P 0.237GB/2ms, MC 0.652GB/1200.91ms, DPP 0.726GB/9.6ms, SDG 1.052GB/22.68ms. 추론은 OpenVLA-OFT와 동일 (15.639GB / 124ms 수준).

## 8. 관련 연구와의 비교 (Comparison with Related Work)

- **명시적 추론 VLA** (DreamVLA, CoT-VLA): 서브골 이미지·깊이 등 외부 조건에 의존 → LEEVLA는 잠재 공간에서 직접 추론하여 보조 모델·픽셀 재구성 불필요.
- **토큰 선택/압축** (OTTER, Compressor-VLA): 추론 시 토큰 스트림을 변경 → LEEVLA는 토큰 스트림을 유지하고 학습 시 가중치만 조정.
- **월드 모델** (DreamerV3, GAIA-1, UWM, DreamZero): 비디오 생성 기반 제어와 달리, LEEVLA는 잠재 미래 예측을 액션 정책의 보조 학습 감독으로만 사용.

## 9. 강점 (Strengths)

1. 추론 비용 제로 — 모든 제안 모듈이 학습 전용이라 배포 시 OpenVLA-OFT와 동일한 지연/메모리.
2. 소형(0.5B)과 대형(7B) 두 스케일 모두에서 일관된 개선 — mini가 1B FLOWER를 능가.
3. LIBERO + CALVIN + 실로봇(UR5)까지 폭넓은 검증과 체계적 어블레이션 (컴포넌트별, 재정렬, α 인자).
4. 사람이 고른 외부 조건(깊이·세그멘테이션) 불필요 — 보조 파이프라인 의존성 제거.
5. DGDP 가중치의 정성적 시각화(Fig. 3)로 해석 가능성 제공.

## 10. 한계 (Limitations)

1. MC 손실의 학습 지연이 크다 (배치당 약 1.2초) — 대규모 사전학습에 부담.
2. 하이퍼파라미터 다수 (τ, δ, τ_c, K=10, M=5, λ1-3, α) — 튜닝 민감도 분석 부족.
3. 실세계 평가가 UR5 단일 플랫폼, 3-4개 과제, 20회 시행으로 제한적.
4. 미래 특징 예측 지평 T의 선택 근거와 민감도가 본문에 명확하지 않음 (보충자료 의존).
5. 클러스터링 품질에 의존 — 어수선한 장면에서 허위 클러스터가 생기면 P2P 순서가 노이즈화될 수 있음.

## 11. 향후 연구 방향 (Future Directions)

- MC 손실의 효율화(근사 k-NN 등)를 통한 대규모 사전학습으로의 확장.
- 흐름 매칭/확산 액션 헤드(π0류)와의 결합 — 현재는 L1 회귀 헤드에 국한.
- 이동 조작(mobile manipulation), 장기 지평 과제, 교차 실체(cross-embodiment)로의 일반화 검증.
- DGDP 가중치를 추론 시 동적 토큰 프루닝에 재활용하는 효율화 연구.

## 12. 총평 (Overall Assessment)

LEEVLA는 "어디를 볼 것인가(DGDP)"와 "잠재 표현이 어떻게 진화하는가(SFFG)"를 학습 시 감독으로 분리해 명시화한, 개념적으로 깔끔하고 실용적인 VLA 학습 프레임워크다. 외부 조건 파이프라인 없이 잠재 공간 미래 예측만으로 LIBERO 98.2%(7B) / 97.5%(0.5B), CALVIN 4.34라는 SOTA급 결과를 달성했고, 추론 비용이 전혀 늘지 않는다는 점이 배포 관점에서 매력적이다. 어블레이션이 컴포넌트별 기여를 설득력 있게 분해하지만, MC 손실의 학습 비용과 제한적인 실세계 검증은 후속 확인이 필요하다. OpenVLA-OFT 계열 회귀 정책에 저비용으로 얹을 수 있는 보조 감독이라는 점에서 실무적 파급력이 크다.

<!-- VERIFIED: pdf -->
