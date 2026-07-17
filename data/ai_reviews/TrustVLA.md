# TrustVLA: Mechanism-Guided Inference-Time Defense Against Vision-Language-Action Backdoors

**arXiv**: [2607.12571](https://arxiv.org/abs/2607.12571) · **발표일**: 2026-07-14 · **소속**: 우한대학교(Wuhan University) / D-Robotics / 화중과기대(HUST) / 중국과학원 자동화연구소(CAS)
**저자**: Pinhan Fu, Xianda Guo, Xuetao Li, Wenke Huang, Ruilin Wang, Weiheng Zhao, Wei Sui, Mang Ye
**코드**: 미공개

> **한 줄 요약**: 재학습 없이 frozen VLA 체크포인트 위에서 동작하는 **추론 시점(inference-time) 백도어 방어 프레임워크**. Dirichlet evidence 프레임워크를 per-token/per-layer 인식적 불확실성 모니터링으로 확장하여 트리거된 관측을 탐지하고, 반사실적(counterfactual) support 국소화 + localized inpainting으로 behavior를 복구한다. OpenVLA/LIBERO에서 평균 VLA-ASR을 100% → 7.0%(BadVLA) / 2.2%(INFUSE)로 낮추면서 clean 성능을 보존.

---

## 1. 문제 정의 (Problem Statement)

VLA 모델은 최종 사용자가 감사(audit)할 수 없는 파이프라인(model supply-chain, Training-as-a-Service, downstream fine-tuning)을 통해 배포된다. 이런 파이프라인에서 공격자는 데이터나 파라미터를 오염(poison)시켜, 모델이 clean 관측에서는 정상적으로 동작하지만 작은 **시각적 트리거**가 나타나면 장기(long-horizon) 로봇 정책을 탈취하도록 만들 수 있다. 핵심 위협은 이 **clean-behavior / triggered-behavior 비대칭성**이다: 트리거된 예측 오류는 잘못된 라벨 하나가 아니라 수백 개 제어 스텝에 걸쳐 물리적 결과가 누적되는 행동 시퀀스이며, 실패가 관측 가능해지기 전에 이미 로봇이 재지향된다. 기존 방어는 부분적 답만 준다 — 입력 전처리(JPEG, Gaussian)는 이미지 디테일을 파괴하면서도 고수준 트리거 효과는 제거하지 못하고, fine-tuning/pruning은 clean 역량을 훼손하면서 백도어를 남기며, 출력 전용 모니터는 장기 제어에서 너무 늦게 반응한다. 더 근본적으로, 성공적인 시각-트리거 VLA 백도어가 **모델 내부에서 무엇을 하는지**에 대한 명확한 설명이 부재하다.

## 2. 핵심 기여 (Key Contributions)

1. **독립 설계 공격들에 걸친 메커니즘적 증거**: 주입 전략이 서로 다른 BadVLA(objective-decoupled poisoned fine-tuning)와 INFUSE(clean adaptation 후에도 잔존하는 module injection) 두 공격을 분석하여, 반복되는 두 내부 시그니처 — **Epistemic Homogenization**(공간적으로 이질적인 불확실성이 거의 균일한 evidence 상태로 압축)과 **Attention Reallocation**(결정 레이어에서 트리거 영역 토큰이 global 지배 없이 순위 상승) — 를 식별. 방어 목표를 고립된 이상 점수(anomaly score)가 아닌 **반증 가능한(falsifiable) 메커니즘 주장**으로 전환.
2. **메커니즘 기반 추론 시점 방어**: Dirichlet evidence를 trusted classification에서 per-token/per-layer VLA 모니터링으로 확장하고, 반사실적 support 국소화 + inpainting과 결합. 재학습, poisoned calibration, 트리거 메타데이터가 전혀 불필요.
3. **주장-분리형 평가(claim-separated evaluation)**: OpenVLA/LIBERO와 pi0.5 transfer 설정에서 탐지 / 복구 / clean false alarm / cross-attack / ablation을 분리 보고. Main row는 paired 500-episode clean/trigger 로그를 사용하고, oracle·clean fine-tuning 후 잔존·Fine-Pruning 진단은 부록으로 분리.

## 3. 방법론 (Methodology)

TrustVLA는 **책임 분리(division of labor)** 원칙으로 조직된 3개 모듈로 구성된다.

**(1) 불확실성 모니터 (Trusted-Evidence Uncertainty)**: Dirichlet evidence 프레임워크를 채택하되, 단일 최종 예측이 아니라 **모든 transformer 레이어에서 language-model head를 통해 Dirichlet evidence를 추출**하여 per-token/per-layer evidence trajectory를 얻는다. 총 evidence를 인식적 불확실성(epistemic uncertainty)으로 매핑하는 단조 사상 EU = V/(E + 2V)를 사용. Clean rollout이 정상적인 공간·레이어별 geometry를 정의하고(clean-calibrated operating region), 트리거된 상태는 evidence-evolution이 이 영역을 벗어날 때만 flag된다. 두 시그니처가 함께 나타나는 이유에 대한 stylized 설명: 트리거가 hidden state에 공유 shortcut 방향을 활성화하면 토큰별 evidence에 공통 성분이 더해져 불확실성이 균일한 저분산 상태로 압축된다(homogenization).

**(2) 반사실적 support 국소화 (Counterfactual Localization)**: 결정 레이어의 attention rank-promotion으로 후보 compact support를 **seed**한 뒤(attention을 인과적 설명으로 직접 신뢰하지 않음), 그 support를 임시로 masking했을 때 탐지에 쓰인 것과 **동일한** mechanism score가 clean-calibrated 영역으로 떨어지는지(score drop)로 검증한다. 이것이 **Definition 1 (Compact Causal Footprint)**의 세 조건이다 — compactness(|S| <= B, area budget), localizability(decision-layer attention 순위 상승 토큰으로 seed됨), score restoration(R(M_S(X)) <= tau_cal). 검증되는 support가 없으면 **fail-safe** 진입.

**(3) 국소 복구 (Localized Inpainting)**: 검증된 support S*를 지우고 주변 장면 구조를 보존하는 inpainting으로 관측을 복구한 뒤 다음 action query를 수행. local-mean/zero/blur masking은 복구 ablation으로 취급.

Clean 프레임은 탐지 비용만 지불하고, 후보 탐색·반사실 masking·inpainting은 탐지 게이트가 발화한 후에만 실행된다.

## 4. 아키텍처 상세 (Architecture Details)

- **적용 대상**: frozen open-source VLA 체크포인트. 주 평가는 **OpenVLA**(SigLIP/DINOv2 vision + LLaMA-2), cross-architecture transfer는 **pi0.5**(PaliGemma).
- **위협 모델**: 사용자는 추론 시점에 hidden state와 attention weight를 관찰하고, action query 전 관측을 수정하며, 타깃 환경에서 소량의 clean calibration 에피소드를 수집할 수 있다. 재학습·poisoning 데이터·트리거 좌표 접근은 **불가**.
- **하이퍼파라미터**: LIBERO 실험에서 area budget **B = 16 image tokens**(≈ 시각 토큰 그리드의 6%). 다른 backbone에서는 정의는 동일하나 grid geometry와 budget을 재보정.
- **추가 학습 파라미터**: 0 (frozen checkpoint, threshold만 clean 데이터로 동결).

## 5. 실험 설정 (Experimental Setup)

- **모델/벤치마크**: OpenVLA를 LIBERO Spatial/Object/Goal/LIBERO-10에서 평가, pi0.5를 LIBERO-style 및 REAL transfer 과제에서 테스트.
- **프로토콜**: 각 LIBERO 셀은 10개 태스크 × 50 rollout. Main row는 동일 체크포인트·동결 방어 설정 하의 **paired 500-episode clean/trigger 로그**를 요구(불완전 페어는 부록으로).
- **공격**: BadVLA(objective-decoupled poisoning)와 INFUSE(Stage-II clean fine-tuning 후 module injection).
- **베이스라인**: 입력 전처리(JPEG q=20, Gaussian ε=0.08), 가중치 감사(ΔW Auditing r=20%).
- **지표**: **VLA-ASR** = max(0, 1 − SR(w)/SR(w/o)) × 100% — 트리거된 태스크 열화로 잔존 백도어 효과를 측정(0%=무트리거 수준 회복, 100%=무트리거 성공 전부 상실). 그 외 DR = TP/(TP+FN), FAR = FP/(FP+TN). Wilson interval은 부록.

## 6. 주요 결과 (Main Results)

**OpenVLA/LIBERO 방어 (Table 1, VLA-ASR ↓)**:

| Attack | 방법 | Spatial | Object | Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|---|
| BadVLA | No defense | 100.0 | 100.0 | 100.0 | 100.0 | 100.0% |
| BadVLA | JPEG / Gaussian / ΔW | ~100 | ~100 | ~100 | ~100 | 99.9–100% |
| BadVLA | **TrustVLA** | **0.6** | **8.5** | **9.4** | **9.5** | **7.0% (↓93.0)** |
| INFUSE | No defense | 100.0 | 100.0 | 100.0 | 100.0 | 100.0% |
| INFUSE | ΔW Auditing | 20.9 | 37.3 | 98.9 | 83.2 | 60.1% |
| INFUSE | **TrustVLA** | **0.0** | **0.6** | **1.7** | **6.3** | **2.2% (↓97.8)** |

- BadVLA 하에서 TrustVLA-방어 **triggered-recovery SR(w)** = 97.6 / 90.6 / 88.6 / 84.0, defended clean **SR(w/o)** = 98.2 / 99.0 / 97.8 / 92.8 — 즉 복구가 정책을 전역적으로 억제해서가 아니라 트리거 rollout을 무트리거 baseline 쪽으로 되돌려 달성됨.
- INFUSE는 clean Stage-II fine-tuning 후에도 transfer되어 최종 triggered SR = 97.4 / 97.4 / 93.4 / 86.6.

## 7. 탐지 신뢰성 및 Cross-Architecture (Detection & Transfer)

- **탐지 (Table 2)**: 두 공격 모두 **0/2000 clean false alarm**. BadVLA 탐지 실패는 Goal(489/500 DR)에 국한, LIBERO-10은 500/500 탐지 → LIBERO-10의 잔존 gap은 탐지가 아니라 국소화·복구에 기인. INFUSE는 2000/2000 탐지.
- **Cross-architecture pi0.5 transfer (Table 3)**: TrustVLA가 VLA-ASR을 Spatial 0.44 / Object 6.04 / Goal 11.80 / LIBERO-10 24.06으로 낮추면서 triggered SR을 90.8 / 90.2 / 82.2 / 60.6로 복구(BadVLA no-defense는 99.4–100 ASR). LIBERO-10에서 잔존이 가장 큼.

## 8. Ablation 및 분석 (Ablations)

- **모듈 분리 ablation**: attention-only, score-drop-only, closure, oracle intervention을 분리 평가하여 attention seeding만으로는 불충분하고 반사실적 score-drop 검증이 필요함을 확인. Oracle masking은 국소화 headroom의 상한(upper bound)으로만 사용.
- **트리거-변형 강건성 (동일 backbone, Fig. 5)**: 다양한 트리거 외형에서 VLA-ASR ~0–9%, 절대 감소 86.0–100 %p.
- **Evidence-evolution footprint (Fig. 3)**: clean 에피소드는 clean-normalized 기준선 근처에, 트리거 에피소드는 벗어남 — BadVLA/INFUSE 양쪽에서 시그니처 재현.

## 9. 강점 (Strengths)

- **재학습 불필요·모델 불가지론**: frozen 체크포인트 위에서 동작, 추가 학습 파라미터 0, poisoned calibration/트리거 메타데이터 불필요 — 실제 배포 위협 모델에 부합.
- **메커니즘 기반 반증 가능성**: 단순 anomaly score가 아니라 Definition 1의 세 조건(compactness·localizability·score restoration)으로 방어 대상을 falsifiable하게 정의. Attention을 인과로 맹신하지 않고 반사실적 검증을 요구.
- **주장-분리형 평가의 엄밀성**: paired 500-episode 로그, false alarm과 recovery failure 분리, oracle을 상한으로만 사용 — 과대주장 회피.
- **clean 성능 보존**: 전역 억제가 아닌 국소 복구로 clean SR을 undefended 수준 근처로 유지.

## 10. 약점 및 한계 (Weaknesses & Limitations)

- **범위 제한**: localizable visual-triggered 공격만을 대상. Adaptive attacker, global filter-style 트리거, 태스크 객체와 분리 불가능한 semantic 트리거는 boundary case로, 복구를 보장하지 않고 보수적으로 탐지·fail-safe만 목표.
- **잔존 gap**: LIBERO-10(장기·복합)과 pi0.5 transfer(특히 LIBERO-10 24.06 ASR)에서 잔존 VLA-ASR이 상대적으로 큼 — 국소화·복구가 어려운 케이스 존재.
- **calibration 의존성**: 타깃 환경의 clean calibration 집합과 area budget B(=16)에 의존하며, backbone마다 grid geometry·budget 재보정 필요.
- **추론 비용**: 탐지 게이트 발화 시 후보 탐색·반사실 masking·inpainting 비용 발생(runtime 회계는 부록 Table 12) — 실시간 제어에서의 지연 영향은 부록에 국한.
- **평가 공격 수**: 두 공격(BadVLA, INFUSE)에 대해 검증 — 메커니즘 주장의 일반성은 추가 공격군에서 추가 검증 필요.

## 11. 의의 및 향후 방향 (Significance & Future Work)

TrustVLA는 VLA 보안 연구에서 "백도어가 모델 내부에서 무엇을 하는가"를 **trusted-prediction 언어(Dirichlet evidence)**로 재정식화하여, 재학습 없는 방어를 메커니즘적으로 근거 지은 첫 시도 중 하나다. Epistemic Homogenization + Attention Reallocation이라는 재현되는 내부 시그니처는 향후 방어·탐지 연구의 진단 좌표로 활용될 수 있다. 향후 방향: (i) adaptive/semantic/global 트리거로의 확장, (ii) 다중 트리거·비-시각 채널 트리거 대응, (iii) 국소화·inpainting 정확도 향상으로 LIBERO-10·transfer 잔존 gap 축소, (iv) 실시간 제어 루프에서의 지연 최적화.

## 12. 총평 (Overall Assessment)

TrustVLA는 embodied AI 보안이라는 시의성 높은 문제에, **frozen 체크포인트 · 재학습 불필요 · 메커니즘 기반 · 주장-분리형 평가**라는 잘 설계된 답을 제시한다. VLA-ASR을 100% → 7.0%/2.2%로 낮추면서 clean 성능을 보존하고, 0 false-alarm의 탐지 신뢰성을 보인 점은 강력하다. 방어 대상을 localizable visual trigger로 한정하고 LIBERO-10·pi0.5 transfer에서 잔존 gap이 남는다는 정직한 한계 명시가 오히려 신뢰도를 높인다. 새로운 VLA 정책이 아니라 **추론 시점 방어 래퍼(wrapper)**로서, VLA 배포 보안 파이프라인에 실용적으로 통합될 잠재력이 크다.

<!-- VERIFIED: pdf -->
