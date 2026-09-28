# DLS: Where Success Breaks — Failure-Boundary Learning for Robust Vision-Language-Action Models

> **한 줄 요약**: SFT는 성공 행동만 가르치고 "어디서 실패가 시작되는지"는 알려주지 않는다는 비대칭을, 디지털 트윈 on-policy 롤아웃으로 실패를 **발견(Discover)**, 시뮬레이터 privileged predicate로 실패 지점을 **국소화(Localize)**, velocity field에서 push-pull로 **형성(Shape)**하는 DLS 파이프라인으로 해결. π0.5 기준 실제 Franka 3과제 ID 84.4% / Unseen 72.2%로 PPO(81.1/67.8)와 2배 데이터 SFT(78.9/54.4)를 능가.

- **arXiv**: 2609.06114v1 (2026-09-05, cs.RO)
- **소속**: Showlab, National University of Singapore
- **백본**: π0.5, π0 (flow-matching VLA)
- **코드**: 공개 정보 없음

---

## 1. 배경 및 동기

VLA 하위 과제 적응은 대부분 전문가 시연 SFT에 의존한다. 그러나 모방 학습은 성공 행동으로 정책을 끌어당길 뿐, 폐루프 실행 중 작은 오차로 시연 분포를 벗어났을 때 무엇을 피해야 하는지 신호를 주지 않는다. 저자들은 이를 **Failure-Boundary Learning**, 즉 정책 자신의 폐루프 행동이 회복 가능한 편차에서 과제 실패로 넘어가는 경계를 학습하는 문제로 정식화한다.

## 2. 문제 정의 및 요구 조건

유용한 실패 신호의 세 요건:
1. **On-policy & scalable**: 수작업 실패 수집은 현재 정책이 만드는 오류를 포괄하지 못하고, 실제 로봇 상호작용은 비싸다.
2. **Progress-localized**: 어느 단계에서 실패했는지에 따라 교정 신호가 달라야 한다.
3. **Generation-aware**: flow 기반 VLA는 다단계 denoising으로 행동을 생성해 정확한 likelihood가 없으므로 표준 policy gradient가 어렵다.

## 3. Stage 1: Real-Grounded Behavioral Prior

- ManiSkill 기반 디지털 트윈(실제 hand-eye 보정 행렬로 카메라 설정)을 구축.
- 실제 시드 궤적 5개에 MimicGen을 적용해 1,000개 시뮬레이션 궤적 생성(카메라 교란 ±1°, ±2cm).
- 실제 50개 시연과 Sim-Real 혼합 SFT(α = 0.5)로 정책 사전 분포를 만든다.

## 4. Stage 2-1: On-policy Boundary Discovery + Semantic Progress Localization (SPL)

- Flow-SDE 샘플러로 확률적 탐색을 주입하며 128개 병렬 트윈에서 롤아웃, 각 step에서 무작위 denoising step 하나의 전이를 버퍼에 저장.
- SPL은 궤적을 Z개 순서 있는 phase로 사상하는 hybrid state transition system이다. phase 전이는 TCP 자세, 물체 자세, 파지 상태, 목표 정렬 등 privileged predicate로 판정.
- 상태 잠재값 Φ = W_{z−1} + w_z φ_z: 이전 phase를 완료해야만 다음 phase 크레딧을 얻는 **discrete gate**로 거리 기반 shortcut을 차단.
- 보상 맵: 성공이면 효율 패널티(η = 0.1) 반영한 1 근처 값, 완료된 phase에는 c_SPL, **실패 지점 phase에는 0**을 부여해 교정 신호를 실패 위치에 집중.

## 5. Stage 2-2: Directional Boundary Shaping (DBS)

- 현재 정책 속도와 old 정책 속도의 차이 Δv로 두 거울 분기 v± = v_old ± βΔv를 만들고, 각 분기가 유도하는 전이 평균과 실제 다음 상태 간 Mahalanobis 오차 E±를 계산.
- 부호 있는 라벨 y ∈ [−1,1]로 Softplus(½ y (E+ − E−))를 최소화 → y>0이면 성공 방향으로 당기고, y<0이면 실패 방향에서 밀어낸다.
- likelihood도 critic도 필요 없다. 실제 데이터 SFT 정규화(λ)로 망각 방지, EMA로 old 정책 동기화, 매 반복 버퍼 비움(strict on-policy).

## 6. 실험 설정

- 과제: Pick & Place(30×30cm 무작위, cm 정밀 배치), Sweep(비파지 밀기), Plugin(~3mm 여유 삽입).
- 하드웨어: Franka Research 3, 전면 RealSense D435 단일 RGB, 10 Hz, chunk H = 8, 과제당 실제 시연 50개.
- 평가: 조건당 30회. ID(980 Lux, 표준 초기 위치) / Unseen(식탁보·배경 변경, 560 Lux 단측 조명, 물체 위치 무작위).
- Baseline: Real-only SFT(N=50, 100), Sim-Real SFT, +GRPO, +PPO. 학습은 H200 4장.

## 7. 주요 결과 (Table 1)

| 백본 | 방법 | Avg ID | Avg Unseen |
|---|---|---|---|
| π0.5 | Real-only (N=50) | 67.8 | 52.2 |
| π0.5 | Real-only (N=100) | 78.9 | 54.4 |
| π0.5 | Sim-Real SFT | 56.7 | 41.1 |
| π0.5 | + GRPO | 74.4 | 64.4 |
| π0.5 | + PPO | 81.1 | 67.8 |
| π0.5 | **DLS** | **84.4** | **72.2** |
| π0 | + PPO | 77.8 | 65.6 |
| π0 | **DLS** | **78.9** | **71.1** |

- π0.5 DLS 과제별: Pick & Place 90.0/80.0, Sweep 73.3/66.7, Plugin 90.0/70.0.
- 실제 시연 50개로 100개 SFT보다 Unseen에서 +17.8%p.
- PPO 대비 격차가 ID→Unseen에서 커진다(π0.5: +3.3→+4.4, π0: +1.1→+5.5).
- Stage 1 없이 바로 RL하면 성공률이 거의 0으로 붕괴한다고 보고.

## 8. Ablation

- **SPL vs Binary reward (Fig. 4, 시뮬레이션)**: SPL이 세 과제 모두 더 빠르게 수렴하고 최종 성공률이 높다(그림 기반).
- **SPL vs 대안 진행도 보상 (Table 2, Unseen, 3 seed)**: Step-count penalty 60.0/53.3/56.7, Distance-based potential 66.7/50.0/63.3, SPL 80.0/66.7/70.0 (Pick&Place/Sweep/Plugin).
- **SFT 정규화 λ (Fig. 5)**: 역U자, λ ∈ [0.5, 1.0]에서 최적, λ ≤ 0.2에서 급락.
- **데이터 효율 (Fig. 6)**: Plugin에서 모든 시연 수에서 DLS가 Real-only SFT를 앞섬.

## 9. 강점

- "성공만 가르치는 SFT"의 구조적 비대칭을 명확한 개념(failure boundary)으로 정리했다.
- 시뮬레이터 privileged 상태를 **학습 측 라벨 생성기**로 쓰고 배포 정책은 RGB+proprio만 쓰는 깔끔한 분리.
- Critic·likelihood 없이 velocity field에서 직접 대조 형성 — flow VLA RL의 실용적 대안.
- 두 백본(π0, π0.5), 세 성격이 다른 과제에서 일관된 개선, 특히 분포 이동 조건에서의 이득.

## 10. 약점 및 한계

- 과제마다 디지털 트윈과 물체 에셋, phase predicate를 수작업으로 정의해야 한다(저자도 인정).
- 실제 평가가 조건당 30회로 작고, 주 결과(Table 1)는 seed 분산 없이 단일 값.
- 시뮬레이션 벤치마크(LIBERO 등) 결과가 없어 다른 방법과 직접 비교가 어렵다.
- 과제가 3개의 비교적 짧은 단일 과제로 장기·다단계 과제로의 확장성은 미검증.

## 11. 재현 및 확장 아이디어

- ManiSkill + MimicGen + openpi 조합으로 재구성 가능하나 코드 미공개.
- SPL의 phase predicate를 VLM 기반 판정으로 대체해 트윈 없이 실제 롤아웃에도 적용.
- DBS 목적을 LIBERO/RoboTwin 같은 표준 시뮬레이션 벤치마크의 RL 후학습에 적용해 GRPO류 방법과 비교.
- 3DGS 기반 real-to-sim 재구성과 결합해 에셋 준비 비용 절감.

## 12. 총평

DLS는 flow 기반 VLA의 사후 적응을 "더 많은 시연 맞추기"가 아닌 "실패 경계 학습"으로 재정의하고, 디지털 트윈·진행도 국소화·critic-free 형성을 하나의 파이프라인으로 엮었다. 소규모 실제 평가라는 한계가 있지만, 50개 시연으로 100개 SFT와 PPO를 넘고 특히 Unseen 조건에서 격차가 커지는 결과는 설득력 있다. 트윈 구축 비용이 해결된다면 실제 배포 지향 VLA 후학습의 유용한 레시피가 될 것이다.

<!-- VERIFIED: pdf -->
