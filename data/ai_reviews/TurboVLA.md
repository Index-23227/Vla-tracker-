# TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM

> **한 줄 요약**: VLA의 통념인 LLM-centric `V→L→A` 경로를 버리고 DINOv3(vision)와 BERT(text)를 독립 인코딩한 뒤 **양방향 cross-attention 6층**으로 직접 융합하는 `V+L→A` 패러다임. ACT-style 디코더가 연속 action chunk를 한 번의 forward pass로 뽑아, **0.2B 파라미터 / 31.2 ms(32 Hz) / 0.9 GB VRAM**으로 LIBERO 평균 **97.7%** 달성 — π0.5(3.4B, 93.6 ms, 96.9%)를 성능·속도·메모리 모두에서 상회.

- arXiv: 2607.27205 (v1, 2026-07-29)
- 저자: Hengyi Xie, Chenfei Yao 외 (HUST) / Xuanyang Xi 외 (Huawei)
- 코드: https://github.com/H-EmbodVis/TurboVLA (프로젝트 페이지: https://H-EmbodVis.github.io/TurboVLA)

---

## 1. 배경 및 동기

- 현행 VLA 다수(RT-2, OpenVLA, π0, π0.5)는 시각 관측을 LLM 토큰 공간으로 투영하고, 지시문 토큰과 함께 수십억 파라미터 LLM을 통과시킨 뒤 행동을 디코딩하는 **간접 `V→L→A`** 구조.
- 이 구조는 매 policy invocation마다 LLM full forward를 요구 → latency·VRAM 병목. Action expert(π0/π0.5)나 parallel decoding(OpenVLA-OFT)은 *행동 생성*만 가속할 뿐, **표현 생성 경로에는 여전히 LLM이 남아 있다**.
- 저자의 관찰: 실행 수준(execution-level) 제어에서 지시문은 이미 "무엇을 할지"를 명시하므로, 정책은 open-ended 생성이나 자율적 task decomposition을 할 필요가 없다. 필요한 것은 **"지시문이 현재 시각 증거를 어떻게 해석하도록 유도할 것인가"** 뿐.
- 따라서 BERT 급 경량 텍스트 인코더 + 컴팩트한 cross-modal 상호작용으로 충분하다는 가설을 세우고 이를 실증.

---

## 2. 핵심 질문

> "LLM을 중심에 두지 않고, 비전과 언어를 행동으로 **직접** 매핑하는 단순하고 효율적인 VLA를 설계할 수 있는가?"

이 질문은 두 하위 질문으로 분해된다: (a) 언어 조건화 자체가 필요한가(→ Sec. 9 ablation에서 필요함을 확인), (b) 그 언어가 반드시 *생성형 LLM*을 통과해야 하는가(→ 불필요함을 확인).

---

## 3. 방법론 개요 (Fig. 3a)

파이프라인은 4단계로 극도로 단순하다.

1. **Vision encoder** (DINOv3): K개 카메라 관측 각각에서 spatial feature 추출 → 공유 차원 `d=256`으로 투영 + positional embedding `E_pos` + camera-view embedding `e_view` → K 스트림 concat.
2. **Text encoder** (BERT): 지시문을 token-level feature `Z_l ∈ R^{N_l×d}`로. **pooled embedding이 아니라 전체 토큰 시퀀스를 유지** — 객체·속성·공간관계가 fine-grained visual conditioning에 살아 있어야 하기 때문.
3. **V-L Interaction Module**: N층 양방향 cross-attention (Sec. 4).
4. **Action Chunk Decoder**: ACT-style transformer가 융합 feature + robot state로부터 H-step 연속 행동을 병렬 예측 (Sec. 5).

Robot state `Z_s = f_state(s_n)`는 **cross-modal 상호작용에는 넣지 않고 디코더에만 주입**한다. 상호작용 모듈은 "task-conditioned scene understanding"에 집중시키고, embodiment configuration은 행동 변환 시점에만 필요하다는 설계 철학.

---

## 4. Vision-Language Interaction Module (Fig. 3b)

- 초기값 `V⁰ = Z_v`, `L⁰ = Z_l`에서 시작해 `(Vˡ, Lˡ) = FusionLayerˡ(Vˡ⁻¹, Lˡ⁻¹)`, ℓ=1..N.
- 각 층 구성: LayerNorm → **양방향 cross-attention** → modality별 FFN + residual.
  - *Visual-to-Instruction*: 장면 문맥을 지시문 스트림에 주입 → vision-aware instruction features.
  - *Instruction-to-Visual*: task semantics로 시각 feature를 변조 → instruction-conditioned visual features.
- 최종 `Z_vl = [V^N ; L^N]` concat.
- 설계 출처는 **Grounding DINO**의 feature enhancer. 실제로 가중치도 grounding-pretrained feature-enhancement weights로 초기화한다. Grounding DINO는 이 상호작용 결과를 object localization에 쓰지만, TurboVLA는 **연속 행동 예측용 control-oriented representation** 구성에 사용한다는 것이 차별점.
- 이 모듈이 사실상 LLM이 담당하던 "무엇이 지금 중요한 픽셀인가"의 역할을 대체한다.

---

## 5. Continuous Action Chunk Prediction

- `Â_n = D_θ(Q_a, [Z_vl ; Z_s]) ∈ R^{H×d_a}`, `Q_a = [q_1,...,q_H]`는 학습 가능한 action query.
- 모든 query가 **병렬 디코딩** → action tokenization도, autoregressive 생성도 없음.
- 학습은 expert action chunk에 대한 **behavior cloning + L1 loss** 뿐. auxiliary language-modeling objective 불필요.
- action head 분류상 **regression** (diffusion/flow-matching 계열이 아님). 이것이 latency 우위의 또 다른 축: denoising step 자체가 존재하지 않는다.

---

## 6. 실험 설정

| 항목 | LIBERO | RoboTwin 2.0 | Real-world |
|---|---|---|---|
| Backbone | DINOv3 ViT-B | DINOv3 ViT-L | ViT-B (LIBERO ckpt) |
| Params | 0.2B | 0.4B | 0.2B |
| Action | 12-step chunk, 7-DoF 연속 | 50-step chunk, 14-dim 절대 관절각 | 7-DoF |
| Data | OpenVLA `no_noops` RLDS, 4 suite 혼합 단일 모델 | official clean demo만 (randomized-scene 제외) | 4×65 teleop demo |
| Steps | 80k (warm-up 10k), batch 256 | 55k (warm-up 1k), batch 192 | 12.5k |
| Protocol | VLA-Adapter rollout, 50 rollout/task, 총 2,000 trial | StarVLA framework, 100 rollout/task ×50 task | 40 trial/task |

- 공통: LR 5e-5, **RTX 4090 4장**. Latency/VRAM은 모두 RTX 4090 batch=1, 공식 구현·체크포인트 기준으로 재측정.
- 중요: TurboVLA는 **Emb. PT.(추가 로봇 데이터 사전학습) ✗** — 벤치마크 데이터만으로 학습한 결과다.

---

## 7. 주요 결과 — LIBERO (Table 1)

| Method | Params (B) | VRAM (GB) | Latency (ms) | Spa. | Obj. | Goal | Long | **Avg** |
|---|---|---|---|---|---|---|---|---|
| Diffusion Policy (RSS'23) | 0.3 | 1.1 | 924.8 | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA (CoRL'24) | 7.5 | 14.9 | 202.9 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 (RSS'25) | 3.2 | 12.3 | 84.2 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 (CoRL'25) | 3.4 | 12.8 | 93.6 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| CogVLA (NeurIPS'25) | 8.3 | 16.1 | 115.5 | 98.6 | 98.8 | 96.6 | 95.4 | 97.4 |
| VLA-JEPA (ECCV'26) | 2.8 | 5.3 | 108.7 | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| OpenVLA-OFT (RSS'25) | 7.7 | 15.7 | 112.2 | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| DDVLA (ICML'26) | 7.5 | 14.5 | 60.8 | 97.2 | 99.4 | 96.8 | 92.2 | 96.4 |
| VLA-Adapter (AAAI'26) | 1.5 | 4.3 | 87.3 | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| Evo-1 (CVPR'26) | 0.8 | 1.7 | 137.2 | 92.7 | 97.7 | 96.3 | 92.3 | 94.8 |
| **TurboVLA (Ours)** | **0.2** | **0.9** | **31.2** | **99.2** | **99.8** | 97.4 | 94.2 | **97.7** |

- π0.5 대비 **파라미터 6%, latency 1/3(93.6→31.2 ms)** 이면서 avg +0.8%p.
- VLA-JEPA 대비 **3× 이상 빠르고 파라미터 7%** 수준이면서 avg 우위.
- Spatial(99.2)·Object(99.8)에서 표 전체 1위. 다만 **Goal(97.4)과 Long(94.2)은 최고가 아니다** — π0.5의 Goal 98.0, VLA-JEPA의 Long 95.8이 더 높다. 즉 우위는 "장기 계획"이 아니라 "지각-접지 정확도"와 효율에서 온다.

## 8. 주요 결과 — RoboTwin 2.0 및 실물 (Table 2, Fig. 4)

**RoboTwin 2.0 (50개 양팔 태스크, clean setting, multi-task)**

| Method | Params (B) | Lat. (ms) | Avg Success (%) |
|---|---|---|---|
| DP3 (per-task, RSS'24) | 0.3 | 78.4 | 55.2 |
| π0 (per-task) | 3.2 | 87.6 | 46.4 |
| UP-VLA (ICML'25) | 1.6 | 74.3 | 52.9 |
| π0.5 (CoRL'25) | 3.4 | 95.6 | 57.0 |
| StarVLA-α (ECCV'26) | 3.8 | 74.9 | 50.3 |
| **TurboVLA** | **0.4** | **43.4** | **60.2** |

단일 팔에서의 우위가 양팔 고차원(14-dim, 50-step chunk) 제어로도 확장됨을 보인 것이 중요하다. 아키텍처가 "LIBERO 특화 트릭"이 아니라는 반증.

**실물 (AgileX Piper, 6-DoF, wrist + third-view D435)**: grab roller 92.5%, move playing card 80%, press stapler 90%, stack three bowls 87.5% — 동일 데이터·프로토콜의 π0.5를 네 태스크 모두에서 상회.

---

## 9. Ablation (Table 3–6, Fig. 6)

**(a) 언어 조건화의 필요성 (Table 3)**

| Condition | Spa. | Obj. | Goal | Long | Avg |
|---|---|---|---|---|---|
| w/o Language | 87.0 | 99.4 | **11.6** | 85.0 | 70.8 |
| Task-ID Embedding | 95.6 | 98.6 | 95.8 | 91.6 | 95.4 |
| Semantic Instruction | 99.2 | 99.8 | 97.4 | 94.2 | **97.7** |

언어 제거 시 LIBERO-Goal이 **97.4 → 11.6**으로 붕괴. 같은 장면에서 여러 행동이 가능한 Goal suite가 언어 의존도의 리트머스 시험지다. Task-ID로도 95.4까지 회복되지만 여전히 2.3%p 부족 → 자연어는 closed-set task identity 이상의 정보를 준다.

**(b) 텍스트 인코더 (Table 4)**: BERT 97.7 > T5-Small 97.1 > SigLIP-Base 95.5. 흥미롭게도 T5-Small은 **141.9M 총 파라미터로 97.1%** — 특정 백본에 종속되지 않으며, 더 작게도 갈 수 있다.

**(c) 상호작용 설계 (Table 5)**: w/o Interaction(단순 concat) 95.2 → Language-queries-Visual 96.1 → Visual-queries-Language 96.5 → **Bidirectional 97.7**. 단방향으로는 부족하고 양방향이 상보적이라는 것이 핵심 주장의 실증.

**(d) 깊이 N (Table 6)**: N=2 → 93.5, N=4 → 95.7, **N=6 → 97.7**, N=8 → 96.6 (과적합/최적화 난이도). 파라미터는 206.6M→220.8M로 거의 변하지 않는데 성능 차가 4.2%p — 상호작용 층이 비용 대비 가장 효율적인 용량임을 시사.

**(e) Action horizon H (Fig. 6)**: H=8 96.4 → H=10 96.9 → **H=12 97.7** → H=15 95.6.

---

## 10. 한계 및 미해결 문제

1. **고수준 추론 부재 (저자 자인)**: 구체적 실행 수준 지시문 전용. task planning, 복잡한 semantic reasoning, 개방형 지시("아침 식사 준비해줘")는 불가. 저자도 결론에서 LLM planner + TurboVLA executor의 계층적 결합을 future work로 명시.
2. **일반화 증거 부족**: Emb. PT. 없이 벤치마크 데이터로만 학습했다는 점은 효율성 논증에는 유리하지만, LLM이 제공하던 **open-vocabulary·semantic generalization**(미학습 객체/표현)에 대한 정량 평가가 논문에 없다. 이것이 가장 큰 미검증 지점이다.
3. **RoboTwin 2.0 clean setting 한정**: compute budget 이유로 randomized-scene 데이터 제외. 시각 도메인 변동에 대한 강건성은 미측정.
4. **Long-horizon 열세**: LIBERO-Long 94.2로 표 내 최고(95.8)에 못 미침. LLM 제거의 비용이 장기 태스크에서 드러날 여지.
5. **Latency 비교의 공정성**: 자사 아키텍처는 4090에 최적화된 소형 모델이라 유리. 상대 모델들에 대한 컴파일/양자화 최적화는 적용되지 않은 "official implementation" 기준이므로, 실전 튜닝된 baseline과의 격차는 표보다 작을 수 있다.
6. **실물 평가 규모**: 4개 태스크 × 40 trial, 단일 플랫폼(AgileX Piper). 비교군도 π0.5 하나뿐.

---

## 11. 관련 연구와의 위치

- **경량 VLA**(TinyVLA, SmolVLA, Evo-1, VLA-Adapter): 백본을 줄이지만 *여전히 사전학습 멀티모달 표현을 실행 경로에 유지*. TurboVLA는 그 표현 자체를 제거.
- **가속 지향**(OpenVLA-OFT, DDVLA, Real-Time Chunking, speculative inference, Reflex/SnapFlow류): 행동 생성 또는 캐시/샘플링을 최적화하지만 **LLM 백본은 그대로**. 저자 주장 — "행동 생성 가속만으로도, 모델 축소만으로도 부족하다".
- **언어 인터페이스 계열**(CLIPort, BC-Z, HULC, PerAct, VIMA): 텍스트를 생성 프롬프트가 아니라 task specification으로 쓴 선례. TurboVLA는 이 전통을 현대 VLA 패러다임에 재도입한 것으로 볼 수 있다.
- **Grounding DINO**: 양방향 cross-modal fusion의 직접적 기술 조상. TurboVLA의 실질적 novelty는 "grounding용 feature enhancer를 제어용 표현 생성기로 전용(轉用)"한 데 있다.

---

## 12. 총평

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★★☆ — 구성요소(DINOv3, BERT, Grounding DINO fusion, ACT decoder)는 모두 기성품이며 조합 자체는 새롭지 않다. 그러나 "LLM은 실행 수준 제어에 불필요하다"는 반(反)직관적 주장을 SOTA 수치로 뒷받침한 것은 분야의 기본 가정을 흔드는 기여. |
| **Rigor** | ★★★★☆ — LIBERO 2,000 trial, RoboTwin 5,000 rollout, 실물 160 trial에 4종 ablation. latency/VRAM을 동일 하드웨어에서 재측정한 점도 성실. 다만 일반화 실험 부재가 감점. |
| **Practical impact** | ★★★★★ — 0.9 GB VRAM / 32 Hz는 소비자 GPU는 물론 엣지 보드 배포 가능성을 열며, 원격 서버 의존을 제거한다. 재현 비용도 4090 4장으로 학계 친화적. |

핵심 메시지: **"언어 조건화는 필수지만, 그 언어가 반드시 거대 생성 모델을 통과할 필요는 없다."** Table 3(언어 제거 시 70.8%로 붕괴)과 Table 1(LLM 제거 후에도 97.7%)의 병치가 이 명제를 정확히 절개한다. 커뮤니티에 던지는 진짜 질문은 "TurboVLA가 π0.5보다 좋은가"가 아니라, **"우리가 VLA에 얹은 수십억 파라미터 LLM이 실제로 실행 단계에서 무슨 일을 하고 있었는가"** — 그리고 그 답이 "open-vocabulary 일반화"라면, 그것을 측정하는 벤치마크가 LIBERO/RoboTwin이 아니라는 뜻이기도 하다.

---

### 예상 질문

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | LLM 없이도 되는데 왜 다들 LLM을 썼나? | 사전학습 semantic knowledge와 open-vocabulary 일반화 때문. 논문은 *실행 수준 지시문에 한해* 이 능력이 과잉임을 보였을 뿐, 미학습 객체/표현 일반화는 평가하지 않았다(Sec. 10-2). |
| 2 | 왜 pooled text embedding이 아니라 토큰 시퀀스 전체를 쓰나? | 객체·속성·공간관계가 개별 토큰에 분산되어 있어야 instruction-to-visual cross-attention이 fine-grained conditioning을 할 수 있기 때문(Sec. 4.1). Task-ID ablation(95.4%)이 이 정보 손실의 대가를 보여준다. |
| 3 | 31.2 ms의 출처는 어느 요소가 큰가? | (a) LLM forward 제거, (b) diffusion/flow denoising 부재(단일 forward L1 regression), (c) 0.2B 소형 백본. DDVLA(60.8 ms)가 행동 생성만 최적화하고도 2배 느린 것이 (a)의 기여를 방증. |
| 4 | Goal/Long에서 최고가 아닌 이유는? | 두 suite는 다단계 목표 해석과 장기 의존성을 요구하며, 여기서 LLM의 semantic/reasoning 용량이 아직 값을 한다. 저자가 결론에서 hierarchical LLM planner 결합을 제안한 이유. |

<!-- VERIFIED: pdf -->
