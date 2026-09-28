# HaWMPO: Hallucination-Aware World Model-based Policy Optimization for Generalist Robot Policy

> **한 줄 요약**: World model 상상 rollout으로 VLA를 GRPO 후학습할 때, action-conditioned Hallucination-Aware Model(HAM)이 생성 영상 chunk의 신뢰도를 점수화하고 이를 Reward-Soft로 보상에 반영해 환각 rollout의 영향을 억제. One-shot SFT OpenVLA-OFT 기준 LIBERO 3-suite 평균 48.7% → 63.7%, G1 실로봇 2태스크 67.5% → 80.0%.

---

## 1. 배경 및 동기

- 실로봇 온라인 RL은 비용·샘플 효율·안전 문제가 크다 → world model을 가상 시뮬레이터로 쓰는 RL 후학습(WMPO, WoVR, World-Env 등)이 대안으로 부상.
- 그러나 생성형 world model은 엄밀한 물리 시뮬레이터가 아니며 long-horizon rollout에서 오차 누적·환각이 발생. 정책이 실제 환경에서 유효한 행동 대신 world-model 편향을 exploit하게 됨(WoVR가 지적).
- 목표: rollout의 신뢰도를 명시적으로 추정해 정책 업데이트에서 신뢰도 낮은 chunk의 영향을 줄이는 것.

## 2. 방법론 심층 분석

### 2.1 파이프라인 (Algorithm 1)
- 구성: VLA 정책, action-conditioned world model(frozen), HAM, reward model.
- 정책이 action chunk를 제안 → world model이 미래 8프레임 생성 → reward model이 성공 여부 평가 → HAM이 환각 점수 산출 → Reward-Soft로 보상 조정 → GRPO 업데이트.

### 2.2 Hallucination-Aware Model (HAM)
- 조건 프레임과 VLA action을 입력으로 받아 생성 영상 chunk의 환각 정도를 회귀. 관측된 미래 프레임이 필요 없어 상상 rollout 중에도 사용 가능.
- 학습 타깃은 여러 지표(이미지 품질, DINO 유사도, trajectory 일관성, depth 불일치 등)를 결합한 composite proxy.

### 2.3 Reward-Soft + GRPO
- 최종 보상 = 원 보상을 환각 점수와 penalty 계수 α로 soft하게 감쇠(Eq. 1). 그룹 단위 reward normalization 후 GRPO loss와 KL 정규화(Eq. 7)로 VLA를 end-to-end 업데이트.
- α가 너무 크면 유용한 가상 경험을 버리고, 너무 작으면 환각 rollout이 정책을 오염.

## 3. 데이터 전략

- 초기 정책: 태스크당 expert trajectory 1개만 사용하는 one-shot SFT.
- World model: WoVR의 action-conditioned Wan2.2-TI2V-5B를 계승, RL 전에 1회 학습 후 고정(PACE 전략 미사용).
- 실로봇: G1 로봇의 Tissue-to-Box, Headphone-on-Stand 두 태스크에 대해 동일 파이프라인 적용.

## 4. 시스템/학습 세부사항

| 항목 | 값 |
|---|---|
| Base policy | OpenVLA-OFT (one-shot SFT) |
| World model | Wan2.2-TI2V-5B action-conditioned, VAE + 30-block DiT(hidden 3072, 24 heads) |
| WM 입출력 | 5 conditioning frames(참조 1 + 최근 4) + 8-step action chunk → 8 미래 프레임, 256×256 |
| RL | GRPO, 200 steps |
| Optimizer | Adam, backbone lr 2e-5, value lr 3e-3, wd 0.01, grad clip 1.0 |
| 하드웨어 | 8× H200 |

## 5. 실험 설계 및 평가 프로토콜

- LIBERO Spatial/Goal/Object 3 suite(Long 제외), 모든 방법 200 RL step 후 평가.
- Baselines: OpenVLA-OFT base, WMPO, WoVR*(PACE 없이 재현, world model 1회 학습).
- α는 suite별로 선택(Object/Spatial 0.3, Goal 1.0)해 Table 1 보고.
- 실로봇: G1, 태스크당 20 trials, base SFT 정책 및 WoVR와 비교.
- **주의**: one-shot SFT + 3-suite만 사용하는 비표준 프로토콜이므로 표준 LIBERO 리더보드 수치와 직접 비교 불가.

## 6. 실험 결과 심층 분석 (PDF Table 1, 2 직접 인용)

| Method | Spatial | Goal | Object | Avg |
|---|---|---|---|---|
| OpenVLA-OFT-base | 61.5 | 48.2 | 36.3 | 48.7 |
| WMPO | 67.8 | 54.6 | 48.0 | 56.8 |
| WoVR* | 69.2 | 64.0 | 49.6 | 60.9 |
| **HaWMPO** | **77.2** | 61.6 | **52.2** | **63.7** |

- Base 대비 +15.0, WMPO 대비 +6.9, WoVR* 대비 +2.8 pp. Goal에서는 WoVR*(64.0)보다 낮음.

| Real (G1) | Tissue-to-Box | Headphone | Avg |
|---|---|---|---|
| Base VLA | 75% (15/20) | 60% (12/20) | 67.5 |
| WoVR | 80% (16/20) | 65% (13/20) | 72.5 |
| **HaWMPO** | **85% (17/20)** | **75% (15/20)** | **80.0** |

- HAM 환각 검출(Table 3, 50 LIBERO chunk 사람 라벨): AUROC 0.9375, AP 0.8952 — 미래 프레임을 참조하는 DINO(0.9062)/Depth(0.8941)보다 높고 MUSIQ(0.3056)보다 크게 우수.

## 7. Ablation 분석

- HAM 제거(Table 4, α=0.3 통일): Spatial은 80/120/200 step에서 +11.6/+9.8/+8.0 pp 향상, Object는 +2.4/−1.4/+2.6, Goal은 −0.2/−1.8/−2.4.
- Goal의 열세는 α 선택 문제로 설명: α=1에서 66.2%로 HAM 제거(64.0%)를 앞섬(Fig. 4).
- α 민감도(Fig. 4): α=0.1은 모든 suite에서 가장 느리게 개선.

## 8. 관련 연구 비교

- WMPO/World-Env: world model 기반 GRPO/PPO 후학습이지만 rollout 신뢰도를 명시적으로 모델링하지 않음.
- WoVR: 환각 문제를 지적하고 PACE 전략을 제안 — HaWMPO는 PACE와 직교하는 reward-level 보정.
- 실로봇 온라인 RL(예: HIL-SERL, RL-100 계열)과 달리 물리 상호작용을 최소화.

## 9. 한계 및 미해결 문제

- LIBERO-Long 미평가, one-shot 설정의 절대 성능(63.7%)은 여전히 낮음.
- Goal suite에서 개선 불안정, α를 suite별로 튜닝해야 함.
- 실로봇은 2태스크 × 20 trials로 저자도 "preliminary evidence"라고 명시.
- HAM 학습 타깃이 composite proxy에 의존, 사람 라벨 검증은 50 chunk 규모로 작음. 유의성 검정 없음.

## 10. 총평

World-model RL의 핵심 병목인 환각을 "신뢰도 점수 → 보상 감쇠"라는 단순하고 plug-in 가능한 형태로 다룬 실용적 기여. 실험 규모는 작고 Goal suite 결과가 혼재하지만, 방향성(Spatial에서 일관된 이득, 실로봇 개선)은 명확하다.

## 11. 🔥 예상 날카로운 질문 모음

1. 보상 감쇠 대신 환각 chunk를 advantage 계산에서 제외(hard masking)하면 어떤가?
2. HAM 자체가 world model과 같은 편향을 공유할 가능성은? 정책이 HAM을 exploit할 수 있는가?
3. 200 step 이후 더 긴 학습에서도 격차가 유지되는가?
4. Full-data SFT 정책(성능 ~97%)에서도 이득이 있는가?
5. 실로봇 태스크에서 world model은 어떤 데이터로 학습됐는가?

## 12. 재현성 및 후속 연구 제안

- 코드 공개 언급 없음. 하이퍼파라미터와 world model 구조는 Appendix에 상세.
- 후속: (1) world model 재학습(PACE 등)과의 결합, (2) 불확실성 기반 적응적 rollout 길이, (3) LIBERO-Long·RoboTwin 등 장기 태스크 검증, (4) HAM의 cross-domain 일반화.

<!-- VERIFIED: pdf -->
