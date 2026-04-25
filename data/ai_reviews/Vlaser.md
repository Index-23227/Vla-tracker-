# Vlaser: Vision-Language-Action Model with Synergistic Embodied Reasoning

> **한 줄 요약**: USTC + Shanghai AI Lab 연합의 ICLR 2026 논문. InternVL3(2B/8B) backbone + flow-matching action expert 구조에 6M 규모 Vlaser-6M(grounding 1.8M + QA 1.2M + spatial 0.6M + planning 0.4M + in-domain sim 2M) 데이터로 SFT. 12개 embodied reasoning 벤치마크에서 8B 모델이 +10% margin 우위, SimplerEnv WidowX 65.1 / Google Robot VM 76.2를 달성. 핵심 발견: 일반 reasoning 데이터보다 **in-domain 데이터가 VLA 미세조정에 훨씬 효과적**.

---

## 1. 배경 및 동기

- 기존 연구 두 흐름이 서로 단절: (a) embodied reasoning을 강화한 VLM(Cosmos-Reason1, RoboBrain, Embodied-R1), (b) end-to-end 제어 VLA(OpenVLA, π0).
- 핵심 질문: **어떤 종류의 VLM 데이터/능력이 downstream VLA 성능에 진짜 기여하는가?** (instruct → action transfer 미해명)
- 목표: 강한 reasoning을 가진 embodied VLM을 구축하고, 그 reasoning이 robot control로 잘 transfer되는지 체계 분석.

---

## 2. 방법론 심층 분석

### 2.1 모델 구조 (Figure 2)
- **Backbone**: InternVL3 (InternViT 비전 인코더 + Qwen2.5-1.5B / 7B LLM) → Vlaser-2B / Vlaser-8B.
- **Action Expert**: π0 스타일 flow-matching expert. LLM과 self-attention을 공유하는 MoE-like 구조 (vision/text 파라미터 + action/state 파라미터 분리). non-causal attention.
- 단일 프레임 관측에서 H=4 길이 action chunk 생성, δ⁻¹=10 integration steps.

### 2.2 Vlaser-6M 데이터 엔진 (Section 2.2)
- **Embodied Grounding 1.8M**: bbox + center point, [0,1000] 정규화. RoboPoint, ShareRobot, Pixmo-Points, Paco-LaVIS, RefSpatial + SA-1B에서 합성한 300k.
- **General + Spatial Reasoning**: RoboVQA 1.2M + SPAR/SpaceR-151k/VILASR 500k + ScanNet/ScanNet++/CA-1M/ARKitScenes로부터 100k 수동.
- **Planning 0.4M**: Alpaca-15k, MuEP, WAP, LLaRP+Habitat, EgoPlan-IT, EgoCOT.
- **In-Domain 2M**: SimplerEnv (Google + WidowX) + RoboTwin (Aloha-AgileX)로부터 grounding/spatial/planning/QA 형식 합성.

### 2.3 학습 레시피 (Section 2.3)
1) **VLM Pretraining**: InternVL3에 SFT, autoregressive LM loss (식 1).
2) **VLA Finetuning**: action expert에 flow-matching loss (식 2). Aᵗᵢ = τAₜ + (1−τ)ε, vθ가 ε−Aₜ를 예측.

### 2.4 Ablation 변종 (Section 3.2)
- Vlaser-OOD: out-of-domain reasoning data만 사용.
- Vlaser-QA / Vlaser-Spatial / Vlaser-Grounding: 단일 in-domain 카테고리.
- Vlaser-All: 전부 결합.

---

## 3. 실험 결과

### 3.1 12 Embodied Reasoning 벤치마크 (Table 1)
| Model | Avg |
|-------|-----|
| GPT-4o | 34.2 |
| Claude-3.7-Sonnet | 33.6 |
| Gemini-2.5-Pro | 44.4 |
| InternVL3-2B (base) | 15.2 |
| RoboBrain2.0-3B | 35.3 |
| **Vlaser-2B** | **45.3** |
| InternVL3-8B (base) | 22.3 |
| Embodied-R1-7B | 38.9 |
| RoboBrain2.0-7B | 37.0 |
| **Vlaser-8B** | **51.3** |

- 2B에서 베이스 InternVL3-2B 15.2 → 45.3 (3×). 8B에서 22.3 → 51.3 (+10% margin vs 차순위).
- 흥미로운 scaling: 단순 grounding(Where2place 74.0 vs 69.5)은 2B가 8B를 능가 / 복잡한 planning·closed-loop는 8B 우위 — CoT 효과.

### 3.2 SimplerEnv WidowX (Table 2, in YAML simpler_bridge_widowx 78.6)
| Model | Carrot | Eggplant | Spoon | Stack | Avg |
|-------|--------|----------|-------|-------|-----|
| π0 (3B) | 55.8 | 79.2 | 63.3 | 21.3 | 54.9 |
| SpatialVLA (4B) | 25.0 | 100.0 | 16.7 | 62.5 | 42.7 |
| Vlaser-OOD (2B) | 60.8 | 35.4 | 56.7 | 20.0 | 43.2 |
| Vlaser-QA (2B) | 55.8 | 83.3 | 77.9 | 33.3 | 62.6 |
| **Vlaser-All (2B)** | 52.5 | **87.9** | 76.6 | **43.3** | **65.1** |

> 주의: YAML에 기록된 simpler_bridge_widowx 78.6은 본 PDF Table 2의 65.1과 다름 — 별도 평가 프로토콜(예: 8B + 다른 setting)일 가능성. 추후 검증 필요.

### 3.3 Google Robot VM (Table 3) / RoboTwin (Table 4)
- Vlaser-All 2B Google Robot VM **76.2 평균** (Pick Coke 91.0 / Move Near 85.4 / Drawer 52.1), Variant Aggregation 59.0. π0 58.3, Magma 8B 68.4 대비 명확한 우위.
- RoboTwin Aloha-AgileX 12 task Vlaser-All 2B **67.5%** (RDT-1B 36.8, InternVL3-2B 55.8) — bimanual 일반화 검증.

### 3.4 Ablation 핵심 발견
- **Vlaser-OOD ≈ InternVL3-2B**: OOD reasoning만으로는 closed-loop 개선 미미. In-domain QA/Spatial/Grounding 각각 대폭 개선, 결합 시 추가 향상 → reasoning 벤치 ≠ closed-loop의 domain gap 입증.

---

## 4. 한계

1. **VLA-Tracker YAML과 PDF 수치 차이**: simpler_bridge_widowx 78.6 (YAML) vs Table 2 65.1. simpler_google_robot 68.4 (YAML) vs Table 3 76.2. — 어느 setting인지 명시 부재. (검증 필요)
2. **Embodied reasoning ≠ control 격차**: 저자도 인정. internet-scale reasoning이 robot policy에 직결되지 않는 근본 문제는 미해결, 단지 domain gap을 우회.
3. **In-domain 데이터가 SimplerEnv·RoboTwin sim에서 합성** → 실세계 generalization 미검증.
4. **Action expert의 단일 프레임 관측**: 시간 정보 부재 → 동적 환경 한계.
5. **2B/8B 외 scaling 부재**: 0.5B나 30B 같은 극단 미탐색.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★☆ — VLM↔VLA 분석 + 데이터 엔진 |
| Technical depth | ★★★★☆ — π0 expert 구조 차용, 데이터 큐레이션 정교 |
| Experimental rigor | ★★★★★ — 12 reasoning + 3 sim 플랫폼 + ablation 5종 |
| Practical impact | ★★★★☆ — open-source(github.com/OpenGVLab/Vlaser) |
| Writing | ★★★★☆ |

**강점**: in-domain 데이터의 중요성을 정량 입증, open-source, 다중 embodiment. **약점**: reasoning↔control gap의 근본적 원인은 미해결, sim-to-real 미검증.

---

## 6. 예상 날카로운 질문

| # | 질문 | 핵심 답변 |
|---|------|---------|
| 1 | "out-of-domain reasoning이 closed-loop에 도움 안 된다"는 강한 주장의 근거? | Table 2/3에서 Vlaser-OOD가 InternVL3-2B와 거의 동등. 단 OOD 정의가 reasoning benchmark용 데이터라 제한적. |
| 2 | flow-matching vs diffusion vs autoregressive action — 본 논문에서 비교? | π0 답습이라 비교 부재. action head 자체 영향은 ablation 안 됨. |
| 3 | 2B가 grounding에서 8B를 이기는 현상은? | 짧고 직접적인 답변(point/bbox)에서 작은 모델이 over-thinking 회피. CoT 길어지면 8B 우위. |
| 4 | Vlaser-6M에 Open-X 데이터가 빠져있는데? | Open-X는 robot trajectory 위주, VLM SFT용 QA/grounding 형태가 아님. In-domain으로는 SimplerEnv/RoboTwin 데이터에 의존. |
| 5 | YAML의 simpler 점수와 본문 점수의 차이는? | YAML 작성 시 별도 setting/checkpoint 사용 가능성. 검증 필요. |

<!-- VERIFIED: pdf -->
