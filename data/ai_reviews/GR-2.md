# GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation

> **한 줄 요약**: 3800만 비디오 클립·500억+ 토큰의 대규모 인터넷 비디오 사전학습을 통해 범용 로봇 agent를 구축, 100+ 태스크에서 97.7% 성공률과 미지 환경에 대한 뛰어난 일반화 달성.

---

## 1. 배경 및 동기

- GR-1의 후속작으로 **규모를 극적으로 확대**: 비디오 데이터 38M clips, 50B+ tokens
- 핵심: **web-scale video pre-training이 로봇 조작의 일반화에 결정적**

---

## 2. 방법론 심층 분석

### 2.1 Web-Scale Pre-training → Robot Fine-tuning Pipeline

```
[38M web videos, 50B+ tokens] → Video generation pre-training
                                        ↓
[Robot trajectories] → Video + action generation fine-tuning
```

### 2.2 Model Scaling

| 모델 크기 | 100-task SR (%) |
|----------|---------------|
| Small | 89.2 |
| Medium | 94.5 |
| **Large** | **97.7** |

- **Scaling law**: 모델 크기에 따라 성능이 일관적으로 향상

> ❓ **예상 질문**: Scaling이 어디까지 유효한가? Saturate되는 지점은?
> **답변**: 97.7%에서 near-saturation이나, 더 어려운 태스크에서의 ceiling 미검증.

---

## 3. 실험 결과 심층 분석

| 설정 | SR (%) |
|------|--------|
| 100+ 학습 태스크 (in-domain) | 97.7 |
| 새로운 배경 | ~90% |
| 새로운 환경 | ~85% |
| 새로운 물체 | ~82% |
| 새로운 태스크 | ~75% |

- 각 일반화 축에서 점진적 하락하나 여전히 높은 수준

---

## 4. 한계 및 미해결 문제

1. **비공개**: 모델·데이터 비공개, 재현 불가
2. **100+ 태스크**: 대부분 tabletop manipulation. Diversity의 실체 불명확
3. **Cloud 의존**: 대형 모델의 on-device 실행 불가
4. **비디오 생성의 추론 시 역할**: 추론 시에도 비디오 생성하는지 불명확

### Attribution 문제
- **데이터 규모** vs **모델 아키텍처** vs **학습 레시피**의 독립적 기여 분리 불가

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★★☆ — GR-1의 규모 확장 |
| **Technical depth** | ★★★☆☆ — Scaling 위주, 아키텍처 혁신 제한적 |
| **Experimental rigor** | ★★★★☆ — 다양한 일반화 축 평가 |
| **Practical impact** | ★★★★☆ — Web-scale pre-training 유효성 입증 |
| **Writing quality** | ★★★★☆ |

**강점**: 97.7%의 강력한 결과, scaling law의 로봇 분야 검증. **약점**: 비공개, 재현 불가, 아키텍처적 혁신보다는 규모의 힘.

---

## 6. 🔥 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | 38M videos의 curation은 어떻게 했는가? | 미공개. 데이터 품질이 핵심이나 세부사항 없음 |
| 2 | 동일 아키텍처에서 데이터를 1/10로 줄이면? | Scaling 분석에서 데이터 규모의 영향 부분적 보고하나 상세하지 않음 |
| 3 | OpenVLA와 동일 벤치마크에서의 직접 비교는? | LIBERO/CALVIN 등 표준 벤치마크 결과 미제공 |
