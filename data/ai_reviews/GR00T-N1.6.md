# GR00T-N1.6: NVIDIA's Next-Gen Humanoid VLA with Cosmos-Reason and Larger DiT

> **한 줄 요약**: NVIDIA GEAR Lab이 CoRL 2025 키노트(2025-09-29)에서 공개한 GR00T-N1.5의 후속작. Cosmos-Reason-2B VLM(flexible resolution + native aspect ratio) + 32-layer 대형 diffusion transformer(N1.5의 16-layer 대비 2배) + state-relative action chunk 예측을 도입한 full-body humanoid control 모델. 300K step, batch 16384 학습. HF에 weights 공개(nvidia/GR00T-N1.6-3B). LangForce 논문 보고 RoboCasa GR1 Tabletop 24-task 47.6%.

> ⚠️ **arxiv 미발표** — 본 리뷰는 YAML 메타데이터 + NVIDIA 공식 프로젝트 페이지·모델 카드 정보에 기반하며, peer-reviewed 논문 부재로 abstract-only 검증.

---

## 1. 배경 및 동기

- N1.5의 한계: Eagle 2.5 VLM의 고정 해상도, 16-layer DiT의 표현력, action chunk가 absolute pose라 generalization 약함.
- 연구 질문:
  1) Cosmos-Reason 같은 더 강한 reasoning VLM을 backbone으로 가져오면 step-by-step task planning이 개선되는가?
  2) DiT를 2배로 키우면 action 다양성·정밀도가 향상되는가?
  3) state-relative action 표현이 humanoid full-body control에 더 적합한가?

---

## 2. 방법론 심층 분석

### 2.1 Cosmos-Reason-2B VLM (System 2)
- **Flexible resolution + native aspect ratio**: 다양한 카메라(헤드캠, 손목캠, 외부)의 raw 입력을 transcode 없이 수용.
- Cosmos-Reason은 NVIDIA의 video reasoning LLM(2025) — physical commonsense·causal reasoning에 특화 → step-by-step task planning에 활용.

### 2.2 Large Diffusion Transformer (System 1)
- **32 layers** (N1.5의 2배). action expert 표현력 확대.
- **State-relative action chunk** 예측: 절대 pose 대신 현재 state 대비 변화량 예측 → embodiment·initial pose에 강건.

### 2.3 Training Strategy
- 상위 4개 VLM 레이어만 unfreeze → catastrophic forgetting 완화하면서 임베딩 적응.
- 300K steps, batch 16384 (N1.5의 250K에서 +20% step).
- 데이터: real robot + synthetic. (구체적 수치 비공개)

### 2.4 Full-Body Humanoid Control
- 양팔 + 몸통 + 손가락 통합 액션 공간.
- N1.5는 manipulation 위주, N1.6에서 full-body로 확장 (YAML key_innovation).

---

## 3. 실험 결과 (YAML + 공개 정보)

### 3.1 RoboCasa GR1 Tabletop 24-task (LangForce 논문 보고)
- **47.6% avg** (fine-tuned, date_reported 2026-01-28).
- 평가 protocol이 N1.5의 RoboCasa Kitchen 100-demo와 다름 → 직접 비교 불가 (YAML에 명시).

### 3.2 표준 벤치마크 부재
- LIBERO/CALVIN/SimplerEnv 등 공식 보고 없음.
- 외부 논문에서 baseline으로 사용되는 형태로만 점수 등장.

### 3.3 NVIDIA 공식 시연
- CoRL 2025 키노트에서 양손 도구 사용·step-by-step 작업 수행 영상 공개 (정량 수치 미공개).

---

## 4. 한계

1. **arxiv·기술 보고서 부재**: 학술적 검증 불가능. 본 리뷰는 추정·일반 지식 기반.
2. **RoboCasa 47.6% < N1.5 64.1%**: 평가 protocol(Tabletop vs Kitchen) 차이로 단순 후퇴 해석 위험. N1.5와 동일 setting 비교 부재가 가장 큰 결손.
3. **Cosmos-Reason 종속성**: NVIDIA 자사 VLM이므로 외부 백본 교체 어려움.
4. **State-relative action의 trade-off**: 정밀 절대 좌표가 필요한 작업(예: 카메라 calibration, AR)에서 손실 가능.
5. **3B 동일 크기 유지**: DiT는 2배인데 총 파라미터는 3B 동일 — VLM 경량화? (Cosmos-Reason-2B + DiT 32-layer로 budget 재배분)
6. **Open weights — 데이터/학습 코드 부분 공개**: 재현성 NVIDIA 의존.
7. **Top 4 VLM layers unfreeze 의 ablation 부재**: 다른 layer 선택과의 비교 없음.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — Cosmos-Reason 통합 + state-relative action |
| Technical depth | ★★★☆☆ — 논문 부재로 깊이 평가 한계 |
| Experimental rigor | ★★☆☆☆ — 표준 벤치 미보고, 외부 baseline 인용만 |
| Practical impact | ★★★★★ — HF open weights, NVIDIA Isaac 생태계 통합 |
| Writing/Communication | N/A |

**강점**: full-body humanoid 확장, Cosmos-Reason 활용, open weights. **약점**: 학술 검증 부재, N1.5 대비 직접 비교 불가, 표준 벤치 미보고.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | DiT 16→32 layer 2배 확장의 실측 효과는? | NVIDIA 측 ablation 미공개. 일반적으로 DiT scaling은 action 다양성·CFG quality에 도움이지만 latency↑. |
| 2 | State-relative action이 정말 절대 좌표보다 좋은가? | π0.5나 GR-3 등 일부 후속 모델도 relative 채택 — 일반적 trend. 단 long-horizon에서 drift 누적 위험. |
| 3 | Cosmos-Reason-2B로 backbone 변경했을 때 N1.5의 FLARE objective는 유지되는가? | 공개 자료에서 명확치 않음. N1.6에서 FLARE 언급 약화 → 다른 학습 objective로 대체 가능성. |
| 4 | RoboCasa 47.6%가 baseline 수치로 인용되었다는 것은 GR00T-N1.6이 상대적으로 약하다는 의미? | LangForce가 비교군으로 사용했을 뿐, fine-tuned 47.6%는 24-task 평균으로는 합리적. Kitchen vs Tabletop의 protocol 차이 우선 고려. |
| 5 | flexible resolution + native aspect ratio의 실제 이득? | 다양한 카메라 셋업(GR-1 ego, 외부 RGBD)을 단일 모델에 입력 가능 → multi-camera humanoid의 핵심 enabler. 단, 정량 ablation 없음. |
| 6 | N1.5와 N1.6 중 무엇을 써야 하는가? | full-body humanoid task면 N1.6, manipulation 한정이고 RoboCasa Kitchen 등 검증된 setting이면 N1.5 점수가 더 높음. 평가 protocol을 우선 통일해야 합리적 선택 가능. |

<!-- VERIFIED: abstract-only -->
