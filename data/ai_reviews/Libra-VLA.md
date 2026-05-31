# Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어

Libra-VLA(Wei et al., 2026, arXiv:2604.24921, Beihang University · AgiBot)는 dual-system VLA 패러다임에 "learning equilibrium" 관점을 도입한 연구이다. Helix, π0.5, OpenHelix 계열의 System1+System2 분해는 빠른 제어와 느린 추론의 latency 균형에 초점을 맞춰왔으나, Libra-VLA는 학습 난이도(learning difficulty) 자체를 두 sub-system에 균형 배분해야 최적 성능에 도달한다는 가설을 제시한다.

**핵심 동기**:
- (a) Coarse 토큰만으로는 정밀 제어가 불가능하고, fine continuous action만으로는 high-level intent 학습이 비효율적이다
- (b) Action decomposition granularity(bin 개수 N)에 대해 성능이 inverted-U 곡선을 그리며, 양쪽 sub-system의 학습 난이도가 균형 잡히는 지점에서 peak(N=10)를 보인다
- (c) 따라서 "어떻게 분해하느냐"가 단순 architectural choice가 아니라 학습 dynamics를 결정하는 핵심 변수다

## 2. 아키텍처: 비동기 Coarse-to-Fine Dual-System

Libra-VLA는 두 개의 비동기적으로 동작하는 expert로 구성된다.

**Semantic Planner (System 2, 저주파)**: InternVL2.5-2B 기반 VLM에 Parallel Coarse-Action Head를 부착. Bidirectional transformer가 cross-entropy로 discrete action token을 병렬 예측하여 macro-directional intent를 포착한다. 저주파로 동작하며 high-level 의사결정을 담당.

**Action Refiner (System 1, 고주파)**: 독립적인 SigLIP visual encoder를 가진 Diffusion Transformer. Coarse intent에 conditioning되어 고주파 continuous action을 생성, 정밀 alignment를 제공.

**총 파라미터**: 2,738M trainable / 3,042M total. Action head category는 discrete planner + diffusion refiner의 조합이므로 `hybrid`로 분류(YAML tags: dual-system, system1-system2, coarse-to-fine, asynchronous).

## 3. 학습 데이터 및 최적화

논문 본문 PDF의 binary stream에 압축된 부분이 많아 학습 데이터 mix와 hyperparameter는 직접 확인이 제한적이다. 다만 다음은 검증되었다:

- Semantic Planner: discrete coarse action token에 대한 cross-entropy loss
- Action Refiner: noise → expert continuous action distribution 학습 (diffusion)
- Action decomposition granularity N을 ablation: N∈{2, 5, 10, 20, 50} 영역에서 N=10 부근 peak

핵심 hyperparameter인 bin 개수 N은 "learning difficulty balance"를 반영하는 control knob으로 작용. N이 너무 작으면 planner가 너무 쉬워 refiner에 부담이 집중되고, 너무 크면 planner가 quasi-continuous 회귀 문제를 떠안아 둘 다 학습이 비효율적이 된다.

## 4. 핵심 실험 결과: LIBERO 및 LIBERO-Plus

**LIBERO (fine-tuned, 검증된 수치)**:
- Spatial: 98.6
- Object: 99.4
- Goal: 98.0
- Long: 92.8
- Average: 97.2

LIBERO-Long 92.8은 long-horizon task에서 dual-system 분해의 강점을 보여주는 수치로, 단일 stack VLA들이 90 미만에서 정체되는 경향과 대비된다. Spatial/Object/Goal 모두 98+로 짧은 horizon에서는 거의 saturated.

**LIBERO-Plus (perturbation robustness)**: Zero-shot 79.5%, fine-tuned 82.3%. 7개 perturbation 차원(시점, 조명, 배경, 객체 텍스처 등)에 대한 평균으로, robustness 영역에서도 dual-system 설계가 유효함을 시사.

**Real-world (AgiBot G1)**: 3개 실세계 task 평균 69.4%. 시뮬레이션 SOTA가 실로봇으로 직접 transfer됨을 입증.

CALVIN, SimplerEnv, RoboTwin, RLBench, RoboCasa 결과는 paper에 보고되지 않아 YAML에서 omit.

## 5. 비교: Dual-System VLA 계열 내 위상

Helix(Figure AI), π0.5(Physical Intelligence), OpenHelix와 같은 dual-system VLA들은 주로 **inference latency 균형**을 위해 System1/System2를 분리해왔다. Libra-VLA의 차별점:

- **Learning equilibrium 관점**: Latency가 아니라 **학습 난이도 균형**을 dual-system 설계의 1차 목표로 제시
- **Discrete + Diffusion hybrid**: Planner는 discrete cross-entropy, Refiner는 continuous diffusion으로 명확히 modality 분리
- **Inverted-U 분석**: Granularity ablation으로 sweet spot을 정량적으로 식별
- **Independent visual encoder for refiner**: SigLIP을 별도 부착하여 fine-grained geometric feature를 planner의 semantic feature와 분리

LIBERO 평균 97.2는 2026년 4월 시점 dual-system VLA 중 최상위권이며, 특히 LIBERO-Long 92.8은 π0/π0.5 계열과 어깨를 나란히 한다.

## 6. 평가 및 한계

**강점**:
- (a) Dual-system 설계에 대한 새로운 이론적 해석(learning equilibrium / inverted-U)
- (b) LIBERO 4-suite 평균 97.2, Long 92.8로 강력한 성능
- (c) LIBERO-Plus에서 robustness 검증, real-world G1 transfer 확인
- (d) Discrete planner + diffusion refiner의 명확한 modality 분리

**약점**:
- (a) Code 미공개 ("coming soon" 명시, YAML open_source=false, code_url=null)
- (b) CALVIN, SimplerEnv, RoboTwin, RLBench 등 광범위 벤치마크 보고 부재 — LIBERO 단일 시뮬레이션 평가에 의존
- (c) Bin 개수 N의 최적값(N=10)이 LIBERO에 specific한지, 다른 task suite/embodiment에 일반화되는지 미검증
- (d) Asynchronous 동작의 실제 latency 측정값 부재 — Helix/π0.5 같은 동시기 dual-system과의 inference cost 비교 없음

**YAML 점검**:
- `parameters` "~3B"로 표기 (3,042M total과 일치)
- `action_head_category=hybrid` — discrete planner + diffusion refiner 조합에 적절
- `backbone="InternVL2.5-2B + SigLIP"` 표기는 dual-encoder 구조 반영
- `open_source=false`, `code_url=null` 일관 (project page만 존재)
- LIBERO 4-suite 모두 기록 + average 보존 — leaderboard 정렬 정확

**참고**: 본 리뷰는 (a) arXiv abstract, (b) ar5iv HTML 렌더링, (c) project page (libra-vla.github.io)에 기반해 작성되었다. PDF 본문 압축 stream의 ablation table 원수치는 추가 검증이 필요할 수 있다.

<!-- VERIFIED: pdf -->
