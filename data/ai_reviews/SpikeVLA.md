# SpikeVLA: Vision-Language-Action Models with Spiking Neural Networks

> **한 줄 요약**: SigLIPv2 → SpikeSigLIP(훈련 불필요 ANN-to-SNN 변환), LLaMA-8B → 멀티모달 스파이킹 LLM, PPO 학습 완전 스파이킹 액션 정책까지 VLA의 세 구성요소를 모두 스파이킹 신경망(SNN)으로 대체한 **최초의 spiking VLA** 아키텍처. R2R-CE Val-Unseen에서 NaVILA 수준의 내비게이션 성능(SR 53.3 vs 53.9)을 유지하면서 이론적 에너지 소비를 141.25 J → 49.09 J(약 34%), GPU 메모리를 16.1 GB → 6.2 GB로 절감한 ICML 2026 논문.

---

## 1. 배경 및 동기

- NaVid, NaVILA, UniNaVid 등 최근 내비게이션 VLA는 대규모 transformer 기반 멀티모달 추론 + ANN 정책으로 강력한 성능을 내지만, dense 연산으로 인한 추론 지연·에너지 비용이 마이크로 로봇, 다족 보행 로봇, 심우주 탐사 로봇 같은 저전력·실시간 플랫폼 배포를 제약.
- COSMO(선택적 memorization), VL-Nav(효율적 공간 추론) 등 효율 지향 설계가 있었으나 여전히 연속·dense 연산 패러다임 내부에 머물러 추가 효율 개선 여지가 제한적.
- SNN은 정보가 변할 때만 연산이 트리거되는 event-driven 희소 스파이크 메커니즘으로 근본적으로 다른 에너지 인식 연산 패러다임을 제공 — 저자들은 이를 VLA 전체 스택에 적용하는 것을 목표로 함.
- 핵심 질문: **"VLA의 시각 인코딩·멀티모달 추론·연속 제어 전부를 스파이킹 연산으로 재구성하면서 경쟁력 있는 성능을 유지할 수 있는가?"**

## 2. 방법론 심층 분석

### 2.1 전체 구조 (Fig. 2)
세 모듈의 end-to-end spiking VLA:
- **Spike-V**: 현재 프레임 + 과거 프레임을 event-driven 스파이킹 시각 토큰으로 인코딩.
- **Spike-L**: 시각·텍스트 토큰을 융합하고 token-level(채널 단위) event-driven 희소화로 연산량 절감, 내비게이션 명령 생성.
- **Spike-A**: 융합 표현/관측을 완전 스파이킹 정책으로 연속 행동에 매핑 (저수준 보행 제어).

### 2.2 Spike-V: 스파이킹 시각 인코더 (Sec. 3.2)
- **Differential Spiking Neurons**: 적은 timestep T에서 고충실도 표현을 위해 연속 활성값을 시간에 따른 증분 갱신으로 표현하는 differential coding 채택. δ^l[t] = ā^l[t−1] + θ^l z^l[t], ā^l[t]는 시점 t까지 δ의 평균 (식 1). 막전위 갱신에 recurrent auxiliary state c_r[t]를 도입해 입력 변화·스파이크 방출 기반으로 입력 전류를 증분 보정 (식 2).
- **Linear-Layer Conversion**: MLP·attention projection 등 선형 연산을 시간 증분 구동 event-driven 실행으로 변환. bias 누적 방지를 위해 bias를 다음 유닛의 초기 막전위로 이동 (식 3).
- **Nonlinear-Layer Conversion**: LayerNorm, GELU, Softmax 등 단일 입력 비선형 연산자는 differential graded unit으로 증분 출력 ∆F^l[t] = F(c^l[t]) − F(c^l[t−1]), x^l[t] = t·∆F^l[t] (식 4). attention 내 곱셈·행렬곱 등 2입력 연산자는 두 상태 c_a, c_b를 유지하는 동적 매핑으로 처리 (식 5). **추가 fitting/재학습 불필요(training-free 변환)**.
- 결과물이 **SpikeSigLIP**: SigLIPv2(패치 임베딩 + 27 encoder blocks + attention pooling head) 구조를 유지한 채 주요 dense 연산자에 스파이킹 wrapper 적용.

### 2.3 Spike-L: 멀티모달 스파이킹 LLM (Sec. 3.3)
- LLaMA-8B 기반. 실 비디오 + 시뮬레이션 데이터 + 보조 내비게이션 + VQA 데이터로 SFT하여 embodied navigation 기초 모델 구축 후, spike encoding + Integrate-and-Fire 뉴런 통합.
- **통합 토큰 표현**: 히스토리 프레임 V_h ∈ R^(196×t)×d, 현재 프레임 V_c ∈ R^196×d, 텍스트 T를 공유 잠재 공간에 사영해 연결 (식 6).
- **스파이킹 동역학**: 각 토큰이 leaky IF 동역학(leak α, 발화 후 과분극 β, 임계값 V_th)으로 진화 (식 7). 학습 안정화를 위해 L개의 fine-grained step을 병합한 **multi-level spike token** s_i[t′] ∈ {0,…,L} 사용 (식 8).
- **Differential Temporal Sparsity Allocation**: 채널 중요도에 따라 정보량 많은 채널은 긴 spiking horizon(T_c), 덜 중요한 채널은 1-step spike로 인코딩 (식 9). 모달리티 간 채널 중요도까지 고려한 확장 (식 10) + per-token zero-point shift z^(l−1) = min(h^(l−1))로 음수 활성값의 spike-rate 인코딩 지원. 2차(second-order) 최적화로 가중치 정밀 조정.

### 2.4 Spike-A: 스파이킹 액션 정책 (Sec. 3.4)
- **인코딩**: 연속 관측을 population coding (Tang et al., 2021)으로 스파이크화. **Laplacian 커널** 기반 population 응답 A_E(s) = Φ_LoG(s; µ, σ), µ·σ는 학습 가능한 수용장 파라미터 (식 11). soft-reset IF 동역학으로 결정론적 스파이크 생성 (식 12–14).
- **디코딩**: T step 동안 출력 뉴런 스파이크를 누적, 발화율 평균에 학습된 가중치·bias를 적용해 연속 행동 생성 (식 15).
- **학습**: actor-critic PPO (clipped surrogate objective, 식 16) + surrogate-gradient 시공간 역전파(STBP). Gaussian 정책 N(µ_t, σ_t)에서 행동 샘플링. 인코더/디코더 파라미터는 population별 독립 갱신.

## 3. 데이터 전략

- **Spike-L SFT**: 실 비디오, 시뮬레이션 데이터, 보조 내비게이션 태스크, VQA 데이터셋 통합 (NaVILA류 데이터 블렌드와 유사한 구성).
- **Spike-V 캘리브레이션**: ImageNet calibration으로 ANN-to-SNN 변환 품질 확인 (Table 7: T=2→16에서 최종층 feature MSE 2.46→0.12로 감소, 에너지는 0.31→11.15 J 증가).
- **Spike-A**: VLN-CE-Isaac / Isaac 시뮬레이션에서 Unitree Go2 보행 제어 RL 학습, 다양한 지형(험지·경사·장애물, Fig. 5) 검증.

## 4. 시스템/학습 세부사항

- Spike-V 기본 timestep: 전체 시스템 비교 기준 T=8/16 (Table 8), feature 정합은 T=16에서 우수.
- Spike-A 기본 설정: **T=3, population size P=5, Actor [128,128] (baseline 표기는 [256,128]∗), Critic [512,256,128]** — 메모리 2.35 MB, 에너지 0.31 µJ(T=5 기준)/0.10 µJ(T=3), ACEs 10^6 단위.
- 에너지 산정: 45nm 공정 가정 E_MAC=4.6 pJ, E_AC=0.9 pJ의 **이론적 에너지 모델** (식 17–19: SOPs = r·T·FLOPs, 첫 층만 MAC, 이후 층은 AC). ACE(Arithmetic Computation Effort) 지표 병용 (식 20).

## 5. 실험 설계 및 평가 프로토콜

- **벤치마크**: VLN-CE R2R Val-Unseen(미학습 장면 일반화), VLN-CE RxR Val-Unseen(장기 지시 추종), VLN-CE-Isaac(사실적 동역학·주행 가능성 제약 하 end-to-end, 1,077 에피소드, Unitree Go2).
- **지표**: NE↓, OS↑, SR↑, SPL↑, nDTW↑ + 저수준 제어(선속도/각속도 추적 오차) + 자원 효율(Mem, Eng(J), ACEs).
- **설정**: RGB-only, waypoint supervision 없음 — panoramic/depth/odometry를 쓰는 waypoint 기반 방법 대비 더 엄격한 조건.

## 6. 실험 결과 심층 분석

### 6.1 R2R-CE Val-Unseen (Table 1)
| Method | Obs. | NE↓ | OS↑ | SR↑ | SPL↑ | Mem(MB)↓ | Eng(J)↓ | ACEs(10^12)↓ |
|---|---|---|---|---|---|---|---|---|
| NaVid | RGB | 5.47 | 49.0 | 37.0 | 35.0 | 14232 | 157.29 | 4376.68 |
| UniNaVid | RGB | 5.58 | 53.3 | 47.0 | 42.7 | 14232 | 157.29 | 4376.68 |
| NaVILA | RGB | 5.28 | 61.5 | **53.9** | **49.3** | 16120 | 141.25 | 3930.21 |
| **SpikeVLA** | RGB | 5.38 | **63.4** | 53.3 | 47.9 | **6249** | **49.09** | **1196.16** |

- OS는 NaVILA를 상회(+1.9), SR/SPL은 근소 열세(−0.6/−1.4)이나 에너지 약 1/3, 메모리 약 39%, ACEs 약 30%.
- 파노라마+depth를 쓰는 AO-Planner(SR 47.0/SPL 33.0)보다 RGB-only 조건에서 핵심 지표 우위.

### 6.2 RxR-CE Val-Unseen (Table 3)
- SpikeVLA: NE 6.20 / SR 51.9 / SPL 45.3 / nDTW 60.4 vs NaVILA: 6.12 / 52.3 / 46.1 / 61.0 — 언어 다양성이 높은 장기 태스크에서도 근접, 자원 이점 동일.

### 6.3 VLN-CE-Isaac (Table 2)
- SpikeVLA: NE 6.02 / OS 53.6 / SR 32.7 / SPL 28.5 vs NaVILA-R(동일 조건 재현): 6.29 / 52.1 / 36.5 / 29.5. NE·OS 우위, SR −3.8. 에너지 44.23 J vs 141.25 J.

### 6.4 저수준 정책 (Table 4)
- 선속도 오차 0.42(NaVILA 0.23보다 열세) / 각속도 오차 0.29(0.38보다 우위). Spike-A 에너지 **0.31 µJ vs 5.80 µJ (약 1/19)**, ACEs 5.53 vs 161.48 M.

### 6.5 INT4 양자화 비교 (Table 5)
- NaVILA INT4: SR 48.2 / SPL 43.6로 뚜렷한 성능 저하 + 에너지 72.49 J. SpikeVLA는 SR 53.3 / 49.09 J로 **정확도·에너지 모두 우위** — 정밀도 저하(양자화)와 event-driven 희소 연산(SNN)의 메커니즘 차이를 강조.

## 7. Ablation 분석

- **모듈별 변환 효과 (Fig. 3, Table 8)**: Spike-V/L/A를 점진 변환할수록 성능은 완만히 감소, 에너지·연산은 단조 감소. w/o V(6248 MB, 60.64 J), w/o L T=16(16126 MB, 129.70 J) 대비 full 모델 T=16(6249 MB, 49.09 J). Spike-L이 메모리·에너지 절감의 핵심, Spike-V는 경로 효율·성공률 및 연산 효율 개선. 흥미롭게 SpikeVLA(w/o L, T=16)는 R2R SR 54.7/SPL 49.85로 NaVILA 원본을 상회.
- **인코딩 커널 (Table 6, Fig. 4)**: Laplacian 커널이 reward 26.72 / MEL 983.94로 Gaussian RBF(23.10), IMQ(22.73), Triangular(25.15) 대비 우수. ℓ1-거리 지수 감쇠가 국소성(가우시안의 제곱 감쇠)·경성 절단(삼각)·과도한 장거리 영향(IMQ) 사이의 균형점 — 접지 충격·지형 교란의 과도 동역학에 안정적 population 발화 제공. 단 ANN(reward 33.45) 대비 reward 갭 존재.
- **Timestep T (Table 9)**: T=2→5에서 선속도 오차 0.44→0.35, 각속도 0.55→0.45 개선, 에너지 0.07→0.31 µJ 증가. baseline T=3.
- **Population size P (Table 10)**: P=2→5에서 선속도 오차 0.77→0.35 대폭 개선, 메모리 0.98→2.35 MB.
- **Actor/Critic 차원 (Tables 11–12)**: 확대해도 정확도 이득 미미(각속도 오차 되레 증가), 자원만 증가 — 과적합 징후. 소형 정책으로 충분.

## 8. 관련 연구 비교

- **vs NaVid/UniNaVid/NaVILA**: 동일한 RGB-only·no-waypoint VLN-CE 설정에서 성능 동급, 자원 소비 1/3 수준. NaVILA의 hierarchical (VLA + 저수준 RL 보행 정책) 프레임을 스파이킹 도메인으로 이식한 구도.
- **vs 효율 지향 VLA (COSMO, VL-Nav)**: dense 연산 내 최적화가 아닌 연산 패러다임 자체의 전환.
- **vs SNN 선행 연구**: ANN-to-SNN 변환(Bu et al. 2025, Huang et al. 2024/2025 — 본 논문의 differential coding은 Huang et al. 2025 계열), 스파이킹 transformer(Spike2Former), 스파이킹 LM(SpikingBERT, SpikeLM 계열), 스파이킹 RL 정책(PopSAN/Tang et al. 2021의 population coding 차용). SpikeVLA는 이들을 VLA 파이프라인 전체로 통합한 최초 사례.

## 9. 한계 및 미해결 문제

- **에너지 수치가 이론치**: 45nm 공정 가정의 analytic 모델(식 17–19). 실제 뉴로모픽 칩(Loihi 등) 실측 에너지·지연은 미검증 — 저자 스스로 향후 과제로 명시. GPU에서는 SNN의 event-driven 이점이 온전히 실현되지 않음.
- **실물 로봇 실험 부재**: 시뮬레이션(VLN-CE, Isaac)과 GPU 배포 검증에 국한. 뉴로모픽 하드웨어 + 사족보행 로봇 통합은 진행 중이라고만 언급.
- **manipulation 미검증**: 제목·초록은 일반 VLA를 표방하지만 실험은 내비게이션(VLN-CE)과 보행 제어에 한정. LIBERO/CALVIN류 조작 벤치마크 결과 없음.
- **저수준 제어 정밀도 손실**: 선속도 추적 오차 0.42 vs NaVILA 0.23 — 스파이크 이산화의 대가. Spike-A 단독 reward도 ANN 대비 낮음(26.72 vs 33.45).
- Spike-L 변환의 학습 비용, timestep 수·추론 지연(wall-clock latency) 수치 미보고.

## 10. 총평

- VLA 3대 구성요소(시각 인코더·멀티모달 LLM·액션 정책)를 모두 SNN으로 대체하고도 강력한 ANN baseline(NaVILA)과 동급의 VLN 성능을 유지하며 이론 에너지를 1/3로 줄인, "spiking VLA"라는 새 설계 공간을 연 논문.
- 특히 INT4 양자화 대비 정확도·에너지 동시 우위 실험(Table 5)은 SNN 접근의 차별성을 설득력 있게 보여줌.
- 다만 에너지 주장 전체가 이론 모델에 의존하고 실물·뉴로모픽 검증이 없다는 점에서, 현 시점 기여는 "실측된 저전력 시스템"보다는 "변환 방법론 + 시뮬레이션 검증"에 가까움. ICML 논문으로서 방법론적 신규성(differential coding 기반 training-free ViT 변환, 모달리티 인식 temporal sparsity allocation, Laplacian population coding 정책)은 충실.

## 11. 남은 질문 / 후속 연구 방향

- Loihi 2, SpiNNaker 등 실제 뉴로모픽 칩에서의 실측 에너지·지연·성능은? GPU 시뮬레이션 대비 어느 정도의 이론-실측 갭이 있는가?
- 동일한 변환 파이프라인이 manipulation VLA(OpenVLA, π0류)의 연속 제어·고주파 액션 청크에도 적용 가능한가? diffusion/flow-matching 액션 헤드의 스파이킹 변환은?
- Spike-L의 token-level sparsity allocation이 긴 히스토리(장기 비디오 컨텍스트)에서 어떻게 스케일하는가?
- w/o L(T=16) 구성이 NaVILA를 능가한 결과(Table 8)는 SpikeSigLIP 자체의 정규화 효과인가, 분산인가?

## 12. 예상 날카로운 질문 모음

> ❓ **"에너지 절감이 전부 이론치인데, GPU에서 실제로 빠르거나 저전력인가?"**
> **답변**: 논문은 45nm 가정의 E_MAC/E_AC 기반 이론 모델(식 17–19)과 ACE 지표를 사용하며, GPU 메모리(16.1→6.2 GB)는 실측에 가깝지만 J 단위 에너지는 analytic 추정. 저자들도 뉴로모픽 칩 검증을 future work로 명시. GPU는 희소 event-driven 연산을 가속하지 못하므로 실질 이득은 전용 하드웨어에서만 온전히 실현.

> ❓ **"SR/SPL이 NaVILA보다 낮은데 '성능 유지'라 할 수 있나?"**
> **답변**: R2R-CE SR −0.6, SPL −1.4로 통계적으로 근소하며 OS·NE 일부 지표는 우위. 게다가 에너지 1/3·메모리 39% 조건임을 감안하면 Pareto 개선으로 볼 수 있음. 단, VLN-CE-Isaac SR은 −3.8로 갭이 더 큼.

> ❓ **"differential coding 변환은 기존 연구(Huang et al. 2025)와 무엇이 다른가?"**
> **답변**: 기존 differential coding ANN-to-SNN을 ViT의 2입력 비선형 연산자(attention 내 곱셈·행렬곱, Softmax)까지 확장한 실용화 + recurrent auxiliary state에 의한 입력 전류 증분 보정(식 2)이 차별점. 또한 이를 8B급 멀티모달 LLM과 RL 정책까지 포함한 전체 VLA 스택에 통합한 것이 본질적 기여.

> ❓ **"Laplacian 커널이 왜 quadruped 제어에 더 좋은가?"**
> **답변**: 지수적 ℓ1-거리 감쇠가 중거리 변화 민감도를 보존하면서 원거리 간섭을 억제 — 접지 충격·지형 교란 같은 과도 동역학에서 일관된 population 발화를 생성해 정책 최적화를 안정화 (Table 6: reward 26.72로 4개 커널 중 최고, 에너지 0.31 µJ).

> ❓ **"low-level에서 SpikeVLA 메모리가 오히려 크다(2.35 vs 1.20 MB). 왜?"**
> **답변**: population coding(P=5)이 입력 차원을 K배 확장하기 때문. 대신 에너지(0.31 vs 5.80 µJ)·ACEs(5.53 vs 161.48 M)에서 큰 폭 우위 — µJ급 정책에서는 메모리보다 연산 에너지가 지배적.

---

*리뷰 작성일: 2026-07-03. arXiv 2606.27807v1 PDF 전문 검토 기반.*

<!-- VERIFIED: pdf -->
