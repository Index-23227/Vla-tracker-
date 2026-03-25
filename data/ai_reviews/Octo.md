# Octo: An Open-Source Generalist Robot Policy

> **한 줄 요약**: Open X-Embodiment의 80만 trajectory에서 학습한 transformer 기반 generalist policy로, language/goal image 지시와 새로운 sensor/action space에 수 시간 만에 fine-tuning 가능한 **오픈소스 범용 로봇 정책의 선구적 연구**.

---

## 1. 배경 및 동기

- 로봇 학습의 **데이터 파편화**: 각 연구그룹이 자체 데이터로 자체 모델 학습 → 지식 공유 불가
- Open X-Embodiment 데이터셋의 등장으로 대규모 cross-embodiment 학습이 가능해짐
- **오픈소스** generalist policy의 필요성: 누구나 접근·fine-tuning 가능

---

## 2. 방법론

### Transformer Policy

- Observation tokenizer: 다양한 카메라·proprioception을 토큰화
- Task specification: Language command 또는 goal image
- Action head: Diffusion-based

### Flexible Architecture

새로운 sensory input과 action space에 대한 빠른 adaptation:
- New camera → observation tokenizer 추가
- New robot → action head 교체/fine-tuning
- **수 시간 내 adaptation** 가능

---

## 3. 실험 결과

| 설정 | SR (%) |
|------|--------|
| In-domain (WidowX) | ~72 |
| Fine-tuned new robot (Franka) | ~65 |
| Fine-tuned new robot (xArm) | ~58 |
| Zero-shot new task | ~35 |

### 9 Robotic Platforms 검증

다양한 로봇에서의 fine-tuning 결과 보고.

---

## 4. 한계 및 미해결 문제

1. **Zero-shot 성능 한계**: Fine-tuning 없이는 새 환경에서 성능 부족
2. **모델 크기 (93M)**: 최신 VLA (7B+) 대비 매우 작아 capacity 제한
3. **Language 이해**: 전용 VLM 없이 text encoder만 사용 → 복잡한 지시 이해 약함
4. **이후 OpenVLA에 의해 성능 추월**: VLM backbone 활용의 중요성을 역설적으로 보여줌

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — 최초의 실용적 open-source generalist policy |
| **Technical depth** | ★★★★☆ — 유연한 아키텍처 설계 |
| **Experimental rigor** | ★★★★★ — 9 platforms |
| **Practical impact** | ★★★★★ — 커뮤니티 기여 |
| **Writing quality** | ★★★★★ |

**강점**: Open-source, 유연한 architecture, 9 platforms 검증. VLA 연구의 baseline으로 광범위하게 활용됨. **약점**: 모델 크기 제한, language 이해 부족.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | VLM backbone을 추가하면? (→ OpenVLA) | 정확히 OpenVLA가 이를 수행. Language 이해 향상 + 성능 향상 확인 |
| 2 | 93M이 충분한가? Scaling하면? | HPT 등에서 larger trunk이 더 나음을 보임. Octo는 efficiency 초점 |
| 3 | Diffusion head의 step 수와 실시간 성능은? | ~50ms per step, 10 step → ~500ms. 2Hz 제어 |
