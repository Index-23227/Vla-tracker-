# LDiP: Large Discrete Policy — Advancing Explicit Behavior Modeling with Stochastic Iterative Scoring

> **한 줄 요약**: 연속 denoising 정책 대신, 시연을 K-Means로 모은 **대규모 이산 행동 vocabulary**(최대 16,384개)를 반복적으로 재채점·가지치기(16,384→512→16→argmax)하고, 학습 시 행동이 아닌 **점수에 Gumbel 노이즈**를 주는 stochastic scoring으로 표현력을 높인 완전 이산 정책. LDiP-VLA(Qwen3-VL-2B)는 NAVSIM v1 PDMS 92.1, 종단간 LDiP는 NAVSIM v2 EPDMS 88.1(ViT-L), Robomimic ToolHang 0.52·PushT 0.76으로 Diffusion Policy 초과.

- **arXiv**: 2609.07049v1 (2026-09-07, cs.RO)
- **소속**: Fudan University, NVIDIA
- **VLA 백본**: Qwen3-VL-2B (two-system, VLM 고정)
- **프로젝트**: https://zhenxinli.net/LargeDiscretePolicy/

---

## 1. 배경 및 동기

Diffusion/flow-matching 정책은 표현력이 높지만 생성 과정이 암묵적이라 해석이 어렵고, denoising 오차가 물리적으로 불가능한 행동을 만들 수 있다. 이산 정책(trajectory scorer)은 후보를 명시적으로 채점해 해석 가능하지만, 고정 vocabulary의 경직성이 문제로 여겨져 행동 수준 perturbation이나 별도 refinement 모듈로 보완해왔다. 저자들은 병목이 후보 부족이 아니라 **비슷하게 좋은 후보들 사이의 정밀한 순위 매기기**라고 본다.

## 2. 문제 정의

시연을 H-step 행동 chunk로 자르고 K-Means로 K개 중심 V = {v_i}를 만든다(각 중심은 시연 분포 안의 물리적으로 가능한 chunk). 정책은 관측(과 선택적 언어)을 조건으로 모든 후보에 점수를 매기고 하나를 고른다.

## 3. 아키텍처

- 각 후보 v_i를 action tokenizer φ(1D conv)로 토큰화해 Transformer decoder의 query로, 관측 인코더의 조건 토큰을 key/value로 사용.
- **cross-attention 전용 decoder**(후보 간 self-attention 제거로 대규모 vocabulary에서 효율적) + MLP 점수 head.
- 운전에서는 충돌, 주행 가능 영역 준수 등 PDM 하위 지표를 예측하는 추가 head(Hydra-MDP 방식).

## 4. Stochastic Iterative Scoring

- **Iterative scoring**: 1단계에서 전체를 채점 → TopK로 줄인 후보를 가중치 공유 decoder로 다시 채점, T단계 반복 후 argmax. 결정 과정(어떤 후보가 고려·배제되었는지)이 명시적이다.
- **Stochastic scoring**: 중간 단계 TopK 전에 s̃ = s + τ·Gumbel(0,1). 행동 자체는 건드리지 않아 vocabulary의 물리적 타당성을 유지하면서 근사 최적 후보 간 탐색을 유도.
- 추론: 운전은 결정적 점수, 조작은 노이즈 점수 선택이 더 좋았다.
- **학습**: 각 후보에 거리 기반 soft label q_i ∝ exp(−‖v_i − a*‖²/σ), 단계별 cross-entropy 합.

## 5. VLA 확장 (LDiP-VLA)

세 가지 설계를 비교:
1. **Action Group Token**: π-계열처럼 vocabulary를 그룹 토큰(예: 2,048)으로 VLM과 joint attention.
2. **Implicit Action Token**: 소수 학습 토큰만 VLM에 넣고 후보 점수로 선형 디코딩.
3. **Two-System**: GR00T-N1처럼 고정 VLM이 VL 토큰을 만들고 별도 action decoder가 채점 — 최종 채택.

## 6. 실험 설정

- NAVSIM navtrain 103K 장면으로 학습. 종단간 모델은 24×A100·20 epoch, VLA는 8×A100·8 epoch, lr 7.5e-5. vocabulary 16,384개 4초 10Hz 궤적.
- 평가: NAVSIM navtest v2(EPDMS, 종단간), v1(PDMS, VLA), HUGSIM 345개 3DGS 시나리오 zero-shot closed-loop(RC, HD-Score).
- 조작: Diffusion Policy 프로토콜(horizon 16, 관측 2, 실행 8), Robomimic Lift/Can/ToolHang + PushT, 마지막 10 체크포인트 평균 성공률.

## 7. 주요 결과

**NAVSIM v1, VLA (Table 2)**: LDiP-VLA **PDMS 92.1**(NC 99.0, DAC 98.1, TTC 98.6, C 98.2, EP 88.8) — DriveFine 91.8, SpanVLA/AdaThinkDrive 90.3, ReCogDrive 89.6, AutoVLA 89.1보다 높다. 2B급 VLM으로 8B급 모델들을 앞선다.

**NAVSIM v2, 종단간 (Table 1)**: EPDMS ResNet34 84.7(DiffusionDrive 84.2), ViT-L **88.1**(DriveSuprim 87.1), V2-99 87.8.

**HUGSIM (Table 3)**: controller 조정 버전 Overall RC 44.7 / HD-Score 36.7로 최고(ZTRS 42.0/32.9). Extreme 시나리오에서는 조정 후 오히려 하락.

**조작 (Table 4)**: Lift 0.99, Can 0.94(DP 1.00/0.98), ToolHang **0.52**(DP 0.47), PushT **0.76**(DP 0.66).

## 8. Ablation

- **구성요소 (Table 5, V2-99)**: Hydra-MDP-V16384 86.2 → +iterative 86.7 → +개선 tokenizer·cross-attn decoder 87.1 → +stochastic 87.8. 행동 noise injection(top-8)은 87.2로 점수 노이즈보다 낮다.
- **VLA 설계 (Table 6)**: SIS가 세 설계 모두 개선 — Group 87.9→88.8, Implicit 89.4→90.6, Two-System 90.8→92.1.
- **조작 (Table 7, epoch 50)**: one-shot 0.54/0.67 → iterative 0.56/0.69 → stochastic 0.60/0.78(ToolHang/PushT).
- **폭·깊이 (Figure 5)**: 3단계가 최적(4단계에서 소폭 하락), 중간 512·최종 16 후보가 최적.

## 9. 강점

- 이산 정책의 약점을 "vocabulary 확장"이 아닌 "채점 메커니즘 개선"으로 푼 명확한 관점.
- 행동을 교란하지 않는 score-space 노이즈로 물리적 타당성과 탐색을 동시에 확보.
- 운전(open-loop, closed-loop), 조작, VLA까지 동일 프레임워크로 일관된 성능.
- 반복 pruning 과정 시각화로 해석 가능성을 실제로 보여줌.

## 10. 약점 및 한계

- VLA 결과는 운전(NAVSIM)에 한정; 로봇 조작 VLA(LIBERO 등) 실험은 없다.
- 조작 실험은 저차원 시뮬레이션 몇 과제뿐이며 Lift/Can에서는 Diffusion Policy보다 낮다.
- vocabulary가 시연 분포에 묶여 있어 분포 밖 정밀 행동은 표현 불가; 고차원 행동 공간에서 vocabulary 크기가 급증할 수 있다.
- HUGSIM Extreme에서 성능 저하, 실세계 검증 부재(저자 인정).
- NAVSIM 비교가 대부분 보고치 인용이며 VLM 크기·학습 데이터가 서로 다르다.

## 11. 재현 및 확장 아이디어

- LIBERO/RoboTwin 등 조작 VLA 벤치마크에서 LDiP-VLA 평가.
- 계층적/분해형 vocabulary(팔·그리퍼 분리)로 고차원 행동 확장.
- 점수 기반이므로 RL 보상/critic으로 재순위하는 후처리나 RL fine-tuning과 결합.
- 온라인 vocabulary 확장(실패 사례 궤적 추가).

## 12. 총평

"denoising 없이도 충분히 표현력 있는" 이산 정책을 채점 방식만으로 구현한 설득력 있는 작업. 운전 VLA에서 소형 VLM으로 SOTA급 PDMS를 달성했고, 해석 가능성과 물리적 타당성이라는 이산 정책의 장점을 유지한다. 조작 VLA로의 확장은 아직 검증되지 않았다.

**한 문장 요약**: 행동을 만들지 말고 골라라 — 다만 여러 번, 약간의 무작위성과 함께.

<!-- VERIFIED: pdf -->
