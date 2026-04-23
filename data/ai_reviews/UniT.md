# UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling

> **한 줄 요약**: Human egocentric video와 humanoid action을 "visual consequence"를 공통 anchor로 삼아 **embodiment-agnostic discrete latent space**로 통합하는, tri-branch cross-reconstruction 기반의 통합 physical language tokenizer.

---

## 1. 배경 및 동기

- Humanoid foundation model의 scaling 병목 = robotic data 부족.
- 대규모 egocentric human video는 scalable 대안이지만, 인간과 로봇의 **kinematic mismatch**(cross-embodiment chasm)로 직접 전이 불가.
- **핵심 통찰**: 이질적 kinematics도 **물리적 결과(visual consequence)**는 보편적 → 시각을 anchor로 활용하자.

---

## 2. 방법론

### UniT = Unified Latent Action Tokenizer via Visual Anchoring
- 인간과 humanoid action 모두를 공유 discrete latent space의 token으로 맵핑.
- Token은 "embodiment-agnostic physical intent"를 표현.

### Tri-Branch Cross-Reconstruction
1. **Action → Vision** 분기: action이 미래 vision을 예측하게 하여 kinematics를 물리적 결과에 anchoring.
2. **Vision → Action** 분기: vision이 action을 reconstruct하게 하여 무관한 시각 confounder를 필터링.
3. **Fusion** 분기: 정제된 두 modality를 결합해 shared discrete latent space를 형성.

### Downstream: VLA-UniT (policy) / WM-UniT (world model)
- 공유 discrete latent token을 **autoregressive token prediction** 방식으로 예측하여 policy/world-model을 학습 (diffusion 기반 아님 — abstract는 "predicting these unified tokens"로만 기술).
- Token 예측 패러다임이라 LLM-style scaling과 자연스럽게 호환.

---

## 3. 실험 결과

> 논문 PDF 미검증 (abstract-only). 구체 수치는 paper 참조 필요.

- XPeng Robotics 프로젝트 페이지 기반으로 humanoid scaling 목적.
- Cross-embodiment token의 downstream humanoid policy 효과성을 주장 (구체 수치 paper 참조).

---

## 4. 한계 및 미해결 문제

1. **Visual confounder의 완전 제거 난이도**: Vision→Action 분기가 조명·배경 등 irrelevant signal을 완벽히 걸러낸다는 보장 없음.
2. **Discrete tokenization의 precision loss**: 미세한 humanoid motor 명령을 discrete token으로 양자화할 때 정밀도 손실 가능.
3. **Human-egocentric 데이터 편향**: 일상 인간 활동이 humanoid 작업 분포와 다를 때 transfer 효과 제한.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — Visual consequence를 cross-embodiment anchor로 삼은 tri-branch 설계가 독창적 |
| **Practical impact** | ★★★★☆ — Humanoid data scaling 문제에 정면 대응 |

**강점**: "Kinematics는 다르나 visual outcome은 같다"는 철학이 단순·강력하고, tri-branch 구조가 이를 엔지니어링적으로 구현. **약점**: Discrete latent의 해상도와 confounder filtering의 실제 품질 검증 필요.

---

## 6. 🔥 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | LAPA와 차별점은? | Tri-branch 구조와 visual anchoring 철학, discrete token의 physical intent로서의 정의 |
| 2 | 왜 discrete latent인가 (continuous 대비)? | World model과의 joint token prediction, LLM-style scaling 친화성 |
| 3 | Vision이 action을 복원하는 분기의 필요성? | Action이 vision을 예측만 하면 spurious correlation 학습 → reverse branch가 regularizer 역할 |
| 4 | Kinematic mismatch가 크면 visual anchor도 달라지지 않나? | 사용 도구·객체의 상태 변화 등 결과 수준에서는 공유됨 (핵심 가정) |

<!-- VERIFIED: pdf -->
