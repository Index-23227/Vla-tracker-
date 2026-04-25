# Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs

> **한 줄 요약**: Demonstration tour 비디오와 multimodal instruction을 입력으로, **long-context VLM(Gemini 1.5 Pro)** 이 goal frame index를 찾아 전달하면 **offline 위상지도(COLMAP 기반)** 기반 저수준 정책이 waypoint를 생성하는 계층적 navigation 시스템. 836m² 사무실에서 Reasoning-Required 86%·Multimodal 90% end-to-end 성공률 달성.

---

## 1. 배경 및 동기

- 기존 ObjNav/VLN은 자연어 카테고리·문구 수준 goal에 한정되어 있고, multimodal(텍스트+사용자 시선의 이미지) 지시는 거의 다루지 않음 (§1).
- 저자들은 **MINT(Multimodal Instruction Navigation with Tours)** 라는 새로운 형식화를 제안: demonstration tour video F={f_i}와 narrative N, 텍스트 d, 옵션 이미지 I를 받아 robot-centric waypoint a∈ℝ³(Δx, Δy, Δθ)를 출력 (Section 3, Eq. 1–2).
- 단일 VLM zero-shot은 (i) context-length 한계, (ii) action 출력의 OOD 문제로 실패 — §1과 §5.3에서 수치로 입증.

---

## 2. 방법론

### 2.1 Offline: Topological Graph
- COLMAP으로 tour의 6-DoF 카메라 pose 복원 → vertex별 frame 저장; "in front of" + 2 m 이내 조건으로 directed edge 생성 (§4.1, Fig. 4).
- 전통 PRM 대비 단순하고 traversable mapping을 분리하지 않음.

### 2.2 High-level: Long-Context VLM Goal Finding (§4.2)
- Prompt: 모든 tour 프레임(narrative 포함) + 현재 관측 I + 사용자 텍스트 d를 interleave한 후 "find the closest frame"을 요구 → integer **goal frame index g** 출력.
- Gemini 1.5 Pro의 1M 토큰 context를 활용해 1 FPS 풀 tour(948 frames)를 그대로 입력.

### 2.3 Low-level: Topological Graph Goal Reaching (Algorithm 1)
- 매 step: (1) 현재 관측 O를 hierarchical visual localizer(global descriptor + PnP, Fig. 5에서 median ATE 0.056 m)로 G에 정합, (2) Dijkstra로 최단경로 S, (3) 다음 vertex와의 상대 pose를 waypoint로 출력.

### 2.4 Robot/Tour 설정 (§5)
- 휠 기반 mobile manipulator + iLQR-MPC waypoint tracker.
- Office 836 m² teleop tour 16분(948 frames @ 1 Hz, narrative 2개), home 환경은 Pixel 6 스마트폰 75 s 핸드헬드 tour(224 frames @ 3 Hz).

---

## 3. 실험 결과

### 3.1 RQ1 — Office 환경 종단 성능 (Table 2)

| 카테고리 | Goal Finding | Goal Reaching | E2E SR | SPL |
|----------|--------------|--------------|--------|-----|
| Reasoning-Free | 80% | 100% | 80% | 0.59 |
| **Reasoning-Required** | 80% | 100% | **80%** | 0.69 |
| Small Objects | 40% | 100% | 40% | 0.38 |
| **Multimodal** | 85% | 100% | **85%** | 0.64 |

- Personalization: "I'm Lewis…" → 7:14, "I'm visiting…" → 5:28로 narrative-conditioned 분기 (§5.1).
- NeRF 기반 시뮬레이션(20 instruction × 50 starts) 결과 RF/RR 모두 90% E2E SR, 0.83/0.84 SPL (**Table 8**).
- Home 환경 스마트폰 tour로도 5 instruction × 4 starts에서 100% SR, SPL 0.87.

### 3.2 RQ2 — Long-context VLM 비교 (Table 3, Table 4)

| 방법 | RF | RR | SO | MM |
|------|----|----|----|----|
| CLIP retrieval (NLMap 재현) | 35 | 33 | 25 | 20 |
| Text-Only Mobility VLA | 70 | 60 | 50 | 30 |
| **Mobility VLA (Gemini 1.5 Pro, 1 FPS)** | **95** | **86** | **42** | **90** |
| GPT-4V / GPT-4o (1 FPS) | token 한계 초과 | — | — | — |
| Gemini 1.5 Pro (0.2 FPS) | 95 | 67 | 36 | 60 |

→ MM에서 Mobility VLA가 60→90으로 급상승, **1 FPS 처리 가능한 long-context가 결정적**.

### 3.3 RQ3 — Topological Graph 필요성 (Table 5)
- VLM에게 직접 "left/forward/right" waypoint를 출력시키면 **0% E2E**, per-step inference **25.9 ± 8.36 s** — 사실상 사용 불가.
- 위상지도 결합 시 90% SR, SPL 0.84, per-step **0.19 s**.

---

## 4. 한계 및 미해결 문제

1. **Exploration 부재**: tour가 반드시 선행돼야 함 (§6 Limitation; frontier exploration 등으로 보완 가능하다고 언급).
2. **Inference latency**: Gemini 1.5 Pro 호출이 10–30 s 소요 → 사용자 대기 부자연스러움; tour 토큰 caching이 향후 해법으로 제시.
3. **Small Objects 40%**: tour 비디오 해상도/프레임률이 정밀 검색에는 부족 (§5.1, Table 2).
4. **정적 환경 가정**: COLMAP 기반 위상지도가 가구·사람의 동적 변화에 강건한지 정량 평가 미흡(다만 "months later"에도 100% goal reaching이라고 보고).
5. **Embodiment 제약**: wheeled mobile manipulator 1대만 테스트, locomotion·다관절 로봇 미검증(future work로 명시).

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — MINT 형식화 + long-context VLM의 navigation 적용 |
| Experimental rigor | ★★★★☆ — 실세계+NeRF sim+스마트폰 tour 3중 검증, baseline 다양 |
| Practical impact | ★★★★☆ — 스마트폰 tour로 사용자 즉시 배포 가능 |
| Reproducibility | ★★☆☆☆ — Gemini 1.5 Pro API 의존, 토큰 비용·지연 부담 |

**강점**: hierarchical 분해가 잘 동작하는 것을 정량적으로 입증(Table 5), low-level localizer 100% SR. **약점**: VLM 추론 지연·비용, 정적 tour 가정.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|-----------|
| 1 | Topological graph가 정말 필수인가? VLM에게 더 풍부한 prompt를 주면 직접 action 가능하지 않나? | Table 5에서 직접 출력 시 0% SR / 25.9 s/step — 현 세대 VLM으로는 불가 검증됨. |
| 2 | 0.2 FPS로 토큰을 줄이면 어떤가? | RR 67→86로 떨어지고 MM은 60→90 차이 (**Table 4**). 1 FPS, 1M 토큰 context가 결정적. |
| 3 | Multimodal instruction을 텍스트(이미지 caption)로 바꾸면? | Multimodal Tour + Text Instruction은 0.40 vs 이미지 직접 0.90 (**Table 7**). caption이 정보 손실. |
| 4 | localizer가 환경 변화(가구, 조명)에 강건한가? | 수개월 전 tour로도 low-level 100% SR (§5.1)이라 보고; blurry/feature-sparse일 때 last pose fallback (§7.1). |
| 5 | 다른 임베디먼트(다리형 로봇, drone)에서도 작동? | 현재 wheeled MPC로 waypoint 추적; cloud VLM 의존성은 낮으므로 deploy 자체는 용이하다고 §6에서 주장. |

<!-- VERIFIED: pdf -->
