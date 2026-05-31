# GR00T-N1.6: NVIDIA's Next-Gen Humanoid VLA with Cosmos-Reason and Larger DiT

> **한 줄 요약**: NVIDIA GEAR Lab이 CoRL 2025 keynote(2025-09-29)에서 공개한 GR00T-N1.5의 후속작. 백본을 **NVIDIA Cosmos-Reason-2B VLM**(flexible resolution + native aspect ratio)으로 교체하고, N1.5의 16-layer DiT를 **32-layer DiT**로 2배 확장, post-VLM 4-layer transformer adapter를 제거하고 대신 **VLM의 top 4 layer를 unfreeze**, 대부분 embodiment에서 **state-relative action chunk** 예측을 채택한 full-body humanoid VLA. 300K step · global batch 16384 학습. HuggingFace에 weights 공개(nvidia/GR00T-N1.6-3B). 학습 데이터는 YAM, AgiBot Genie-1, Galaxea R1 Pro, Unitree G1 등에서 수집한 **수천 시간의 teleop**.

> NVIDIA 공식 페이지가 1차 출처. 자체 발표 페이지에는 정량 벤치 수치가 거의 없으며, 외부 논문(LangForce 등)이 N1.6를 baseline으로 평가한 47.6%만 본 트래커가 기록.

---

## 1. 배경 및 동기

### N1.5의 한계
- **고정 해상도**: Eagle 2.5 VLM이 고정 해상도/aspect ratio에 묶여 있어 다양한 카메라(헤드캠/손목캠/외부) 입력을 일관 처리하기 어려움.
- **DiT 표현력**: 16-layer DiT가 full-body humanoid action 공간을 다루기에 부족할 가능성.
- **Absolute action**: action chunk가 절대 pose라 embodiment·initial pose에 민감.
- **Adapter 오버헤드**: 4-layer transformer adapter가 VLM과 DiT 사이에 별도 학습 부담.

### 연구 질문
1. Cosmos-Reason 같은 더 강한 video reasoning VLM이 step-by-step task planning에 도움이 되는가?
2. DiT를 2배(32-layer)로 키우면 action 다양성·정밀도가 향상되는가?
3. State-relative action 표현이 humanoid full-body control에 더 적합한가?
4. Adapter 제거 + VLM top layer unfreeze가 catastrophic forgetting 없이 작동하는가?

---

## 2. 방법론 심층 분석

### 2.1 Cosmos-Reason-2B VLM (System 2)
- NVIDIA 자체 video reasoning LLM, physical commonsense·causal reasoning에 특화.
- **Flexible resolution + native aspect ratio**: 카메라마다 다른 해상도/aspect ratio를 transcode 없이 직접 수용. 다중 카메라 humanoid 셋업의 핵심 enabler.
- **Top 4 layers unfreeze**: 나머지 layer는 frozen, 상위 4개만 미세 조정. catastrophic forgetting을 줄이면서 robot-domain 임베딩 적응.

### 2.2 Large Diffusion Transformer (System 1)
- **32 layers** (N1.5의 16-layer DiT 대비 정확히 2배).
- N1.5에 있던 **post-VLM 4-layer transformer adapter는 제거**. 대신 VLM 상위 layer unfreeze로 representation 적응을 직접 수행.
- **State-relative action chunk** 예측 (대부분 embodiment 기본값): 절대 joint angle/EEF 좌표가 아닌 현재 state 대비 변화량을 출력 → embodiment·initial pose에 강건, 모션 부드러움↑. 단 NVIDIA 자체 코멘트로 "소규모 데이터에서 error accumulation에 취약" 한계 명시.

### 2.3 Full-Body Humanoid Control
- 양팔 + 몸통 + 양손 + 다리 등 통합 action space.
- N1.5는 manipulation 위주였던 반면 N1.6는 full-body로 확장 (project page key innovation).

### 2.4 학습

| 항목 | 값 |
|---|---|
| 파라미터 | 3B |
| Pretraining steps | 300K |
| Global batch | 16384 |
| Post-training | 10K-30K steps, global batch ≤ 1K |
| Pretraining 데이터 | 수천 시간 teleop (YAM, AgiBot Genie-1, Galaxea R1 Pro, Unitree G1) + 합성 |

---

## 3. 실험 결과

### 3.1 NVIDIA 공식 페이지
- **정량 벤치마크 수치 부재**. CoRL 2025 keynote에서 양손 도구 사용, step-by-step 작업, full-body 모션 demo 영상만 공개.
- 도전 과제로 다음을 명시: (a) multi-task language following, (b) out-of-distribution generalization, (c) **state-relative action의 누적 오차**.

### 3.2 본 트래커가 기록한 외부 보고치

| Benchmark | GR00T-N1.6 | Source |
|---|---|---|
| RoboCasa GR1 Tabletop 24-task | **47.6%** | LangForce paper (2026-01-28), N1.6 baseline |

⚠️ 평가 protocol(Tabletop vs N1.5의 Kitchen 100-demo)이 달라 N1.5 64.1%와 직접 비교 불가 — YAML에도 명시되어 있음.

### 3.3 LIBERO/CALVIN/SimplerEnv
- NVIDIA 공식 보고 없음.
- 외부 논문에서 baseline으로 등장하는 형태로만 점수가 등장 가능 (현재 본 트래커에는 RoboCasa만 기록).

---

## 4. 어블레이션
- NVIDIA 공식 페이지에 **단일 컴포넌트 ablation 부재**.
- DiT 16→32 layer, adapter 제거, top 4 unfreeze, state-relative action 등 변경이 동시에 적용되어 개별 기여 분리 불가.

---

## 5. 한계

| # | 한계 | 코멘트 |
|---|---|---|
| 1 | arXiv·tech report 부재 | NVIDIA 페이지의 narrative 기술만 존재. 학술 검증 불가 |
| 2 | 표준 벤치 미직접 보고 | LIBERO/CALVIN/SimplerEnv 공식 수치 없음. 외부 paper 인용만 |
| 3 | RoboCasa 47.6 vs N1.5 64.1 | 평가 protocol 차이로 단순 후퇴 해석은 위험하지만 동일 setting 비교 부재는 큰 결손 |
| 4 | Cosmos-Reason 종속성 | NVIDIA 자사 VLM이라 외부 백본 교체 어려움. 재현성에도 영향 |
| 5 | State-relative action의 trade-off | 페이지 자체에서 "limited data에서 error accumulation" 명시 — 정밀 절대 좌표 작업에는 손실 가능 |
| 6 | 컴포넌트 ablation 부재 | DiT scaling / adapter 제거 / unfreeze top4 / relative action 의 marginal contribution 분리 불가 |
| 7 | 동일 3B parameter 유지 | DiT는 2배인데 총 파라미터 3B 유지 — VLM (Cosmos-Reason-2B)이 N1.5의 Eagle 2.5(2.1B)와 비슷한 budget. 즉 budget 재배분 |
| 8 | Open weights 외 코드/데이터 closed | 학습 데이터(teleop 수천 시간)와 GR00T-Dreams 데이터 비공개 |

---

## 6. 총평

| 항목 | 평가 |
|---|---|
| Novelty | ★★★★☆ — Cosmos-Reason 통합 + state-relative action + DiT scaling |
| Technical depth | ★★★☆☆ — paper 부재로 깊이 평가 한계 |
| Experimental rigor | ★★☆☆☆ — 자체 정량 벤치 부재, 외부 baseline 인용만 |
| Practical impact | ★★★★★ — HF open weights, NVIDIA Isaac 생태계 통합, full-body 확장 |
| Writing/Communication | N/A — paper 부재 |

**강점**: full-body humanoid 확장, Cosmos-Reason 활용, open weights, flexible resolution.  
**약점**: 학술 검증 부재, NVIDIA 자체 정량 벤치 미공개, N1.5와의 동일 setting 비교 부재.

---

## 7. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | DiT 16→32 layer 2배 확장의 실측 효과는? | NVIDIA 측 ablation 부재. 일반적으로 DiT scaling은 action 다양성·CFG quality에 도움이지만 latency↑. 본 paper는 그 trade-off 정량화 없음. |
| 2 | State-relative action이 정말 절대 좌표보다 좋은가? | π0.5 등 일부 후속 모델도 relative 채택. NVIDIA 페이지 자체에서 "smoother motion but error accumulation with small data" trade-off 인정. |
| 3 | Cosmos-Reason-2B 백본 변경으로 N1.5의 FLARE objective는 유지되는가? | 페이지에 FLARE 언급 없음. N1.6 학습 objective는 명시 안 되어 있어 다른 loss로 대체됐을 가능성. |
| 4 | RoboCasa 47.6%(N1.6) < 64.1%(N1.5) — 정말로 후퇴? | LangForce의 Tabletop 24-task vs HAMLET의 Kitchen 100-demo — protocol 차이가 더 큰 변수. 직접 비교 불가, YAML도 이를 명시. |
| 5 | Flexible resolution + native aspect ratio의 실제 이득? | 다양한 camera 셋업(GR-1 ego, 외부 RGBD, 헤드캠, 손목캠)을 단일 모델에 그대로 입력 가능 → multi-camera humanoid의 핵심 enabler. 단 정량 ablation 없음. |
| 6 | Top-4 VLM unfreeze는 어떻게 선택했나? | NVIDIA 페이지에 layer 수 sweep 결과 없음. 일반적인 partial fine-tuning heuristic으로 추정. |
| 7 | N1.5와 N1.6 중 무엇을 써야 하는가? | full-body humanoid · 다중 카메라·multi-embodiment 적응이면 N1.6; manipulation 한정·검증된 Kitchen setting이면 N1.5 점수가 더 높음. 평가 protocol 우선 통일이 우선. |
| 8 | "Cosmos-Reason"이 정말 step-by-step planning을 하는가? | NVIDIA가 CoRL keynote에서 영상으로 시연. 정량적 planning eval(예: long-horizon language following)은 미공개. |

<!-- VERIFIED: pdf -->
