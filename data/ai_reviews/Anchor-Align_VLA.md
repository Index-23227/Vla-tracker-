# Anchor-Align VLA: Representation Anchoring과 Language-Action Alignment를 통한 일반화 가능한 VLA 파인튜닝

*arXiv:2607.13429 · University of Illinois Urbana-Champaign / Texas A&M / UC Irvine · 2026-07-15*

## 1. 개요 (Overview)

Anchor-Align은 사전학습된 Vision-Language Model(VLM)을 로봇 시연 데이터로 파인튜닝하여 VLA 정책을 만드는 표준 레시피(behavior cloning, BC)의 근본적 한계를 다룬다. 저자들은 BC 파인튜닝이 시각적·의미론적 일반화를 지탱하는 사전학습 표현을 점진적으로 덮어써 버린다(catastrophic forgetting)는 점을 지적하고, 이를 완화하는 두 가지 손실 항 — Vision-Language Anchoring과 Language-Action Alignment — 을 표준 BC에 추가하는 파인튜닝 방법을 제안한다.

## 2. 문제 정의 (Problem Statement)

두 가지 실패 모드가 지목된다. (1) 표준 BC는 오직 action 예측 손실만 최적화하므로 VLM의 사전학습 표현을 보호하는 장치가 없어, 색·형태·공간 개념 같은 시각언어적 사전지식이 파인튜닝 과정에서 소실된다. 실제 xArm7 로봇에서 "green mug"로 학습된 정책은 "pink mug"를 집으라는 지시에도 90% 확률로 green mug로 향한다. (2) Co-training은 action head와 language head를 서로 다른 관측(로봇 데이터 vs. 일반 image-text)으로 감독하므로, 두 head가 같은 backbone을 공유하면서도 모순된 예측(action은 "right", language는 "left")을 낼 수 있는 language-action misalignment가 발생한다.

## 3. 핵심 방법 (Method)

**Vision-Language Anchoring**: 사전학습된 VLM의 동결 복사본(frozen anchor VLM)을 유지하고, 학습 가능한 VLA의 각 트랜스포머 디코더 레이어 hidden state를 anchor VLM의 대응 레이어로 distillation(MSE Anchor loss)한다. 모든 |D| 디코더 레이어에 대해 평균낸 anchoring 손실을 사용하며, 추가 데이터나 아키텍처 변경이 필요 없다.

**Language-Action Alignment**: 각 시연 궤적의 연속 action target을 프로그래밍적으로 이산 방향 라벨 W = {up, down, left, right, forward, backward} 중 하나로 변환한다. K-step chunk의 translational 성분을 배치 평균하여 라벨을 부여하고, 같은 로봇 관측에 대해 language head와 action head를 함께 감독(Align loss)하여 두 head를 정렬한다.

최종 목적함수는 표준 BC 손실 + Anchor 손실 + Align 손실이다.

## 4. 아키텍처 (Architecture)

기본 아키텍처는 Prismatic-Qwen2.5-0.5B VLM을 LoRA(rank r=64, 전 레이어)로 파인튜닝한다. 두 이미지에서 DINOv2와 SigLIP 특징을 패치별로 feature 차원 결합하여 이미지당 256패치, 총 512 vision 패치의 prefix를 구성한다. 입력은 vision + text + proprioceptive state이며, VLA-Adapter의 bridge attention으로 regression action head 잠재변수를 VLM 특징에 레이어별로 조건화한다. 아키텍처·action head에 대한 일반성을 확인하기 위해 Qwen2.5-VL backbone + flow-matching GR00T FM-DiT head를 쓰는 StarVLA 구성으로도 실환경 실험을 수행한다.

## 5. 학습 설정 (Training Setup)

LIBERO, LIBERO-PRO, LIBERO-Plus, CALVIN ABC->D 시뮬레이션 데이터와 실제 xArm7 로봇 시연으로 학습한다. Anchor-Align 학습은 co-training보다 36분 빠른 68분에 완료되며(본문 언급), 두 보조 손실은 regression과 flow-matching action head 모두에서 일관된 개선을 보인다.

## 6. 벤치마크 결과 (Benchmark Results)

**표준 LIBERO 4-suite (Table 7, App. C.1)**: Spatial 98.4 / Object 100.0 / Goal 97.2 / Long 90.8 — 모든 baseline(π0.5-KI, VLA-Adapter 등)을 상회하며 SOTA.

**CALVIN ABC->D (Table 2)**: 1/5 99.1, 2/5 95.8, 3/5 90.6, 4/5 84.7, 5/5 77.9, 평균 롤아웃 길이 4.5 — 모든 체인 길이에서 최고, 오류가 누적되는 긴 체인에서 이득이 커진다(+0.8@1 → +4.8@5).

**LIBERO-PRO / LIBERO-Plus (Table 1)**: LIBERO-PRO Mean 71.9(다음 baseline 61.0), LIBERO-Plus Mean 90.3(다음 85.1). 가장 어려운 position-swap 축에서 22.6%로, 다른 모든 baseline이 0%인 것과 대조된다.

## 7. Ablation 분석 (Ablations)

Anchoring만(Anchor VLA), Alignment만(Align VLA), 전체(Anchor-Align)를 LIBERO-PRO/Plus에서 비교(Table 3)하여 두 항이 상보적으로 기여함을 보인다. 또한 정규화 대조군(Table 4)으로 Shuffle(관측-라벨 매핑에 고정 순열 적용)과 Scatter를 두어, 개선이 단순히 보조 태스크 추가나 정규화 효과가 아니라 실제 language-action 정렬에서 비롯됨을 검증한다.

## 8. Language-Action Misalignment 진단 (Diagnosis)

Table 5는 LIBERO-PRO 롤아웃에서 두 head의 label 일치 비율(alignment)을 정량화한다. Co-training에서는 alignment가 task success와 무상관이지만, Anchor-Align은 alignment를 높여 task success를 61.0% → 71.9%로 끌어올린다. 이는 co-trained VLA의 language-action misalignment에 대한 최초의 직접 진단을 제공한다.

## 9. 실환경 실험 (Real-World)

xArm7 로봇에서 두 VLA 아키텍처에 걸쳐 real-robot success를 28% → 54%, 37% → 60%로 개선한다. compositional object layout, spatial rearrangement, semantic perturbation, cluttered scene 등 held-out perturbation regime에서 표준 BC가 학습 궤적을 재생하며 실패하는 반면 Anchor-Align은 일반화한다.

## 10. 강점 (Strengths)

추가 데이터·아키텍처 변경 없이 표준 BC에 손실 두 개만 더해 catastrophic forgetting과 misalignment를 동시에 완화한다. regression/flow-matching 두 head, 시뮬레이션/실환경 양쪽에서 일관되게 작동하고, position-swap 같은 극한 OOD에서 baseline 대비 압도적 격차를 보인다. 학습 오버헤드가 co-training보다 오히려 작다.

## 11. 약점 및 한계 (Weaknesses & Limitations)

표준 LIBERO는 이미 포화(saturated)되어 본 방법의 진짜 강점은 PRO/Plus/CALVIN 같은 스트레스 테스트에서 드러난다. 이산 방향 라벨이 6개(translational)로 제한되어 회전·그립 등 세밀한 action 의미는 정렬 대상에서 빠진다. 백본이 0.5B급 소형으로, 대형 VLM에서의 일반화 여부는 미검증이다. 코드 공개 여부는 확인되지 않았다.

## 12. 총평 (Assessment)

Anchor-Align은 특정 모델이라기보다 VLA 파인튜닝 레시피로, "사전학습 표현 보존"과 "효과적 action 학습"이 상충하지 않음을 실증한다. 판정: **ACCEPTED** — fine-tuning method + policy(VLA-Adapter/StarVLA 정책 위에서 검증). LIBERO/CALVIN 정량 결과가 견고하고, language-action misalignment 진단이라는 개념적 기여도 명확하다.

<!-- VERIFIED: pdf -->
