# ERVLA: Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation

> **한 줄 요약**: Embodied CoT는 inference-time 자기회귀 prefix가 아니라 **VLM 표현 공간을 reshape하는 training signal**이어야 한다는 가설을 226M 샘플 규모로 검증. ERVLA는 (i) p=0.5 **reasoning dropout**, (ii) **choice policy** auxiliary branch (N=5 후보 action chunk), (iii) **knowledge-truncated KV conditioning** (DiT가 semantic prefix만 attend)의 3종 세트로 LIBERO-Plus 86.9% (π0.5 대비 +1.4), VLABench 53.2% SR (π0.5 대비 +5.1)을 달성. CoT를 학습 신호로 내재화하고 추론 시에는 직접 action을 생성해 autoregressive compounding error를 우회.

---

## 1. 배경 및 동기

### 기존 Embodied CoT의 3대 한계
1. **무엇이 효과적인 CoT인가가 불명확** — scene understanding, subtask decomposition, spatial grounding, end-effector trajectory, future frame 등 다양한 형태가 특정 architecture와 강하게 결합되어 비교 불가
2. **CoT와 action policy의 결합 방식 문제** — 초기 ECoT는 reasoning을 **action prefix**로 autoregressive 생성 → latency + compounding error
3. **Scaling 불명확** — 공개 reasoning-annotated robot 데이터셋이 희소해 large-scale CoT가 실제로 action generation을 강화하는지 확증 불가

### 핵심 통찰
- ECoT는 "로봇이 더 많이 말하게" 만드는 것이 아니라, VLM이 학습한 의미적 추상을 **action generation에 유용한 중간 표현**으로 번역하는 것
- 따라서 CoT는 **출력 채널이 아니라 학습 신호**여야 한다

---

## 2. 방법론 심층 분석

### 2.1 Mixture-of-Transformers 아키텍처

```
[Multi-view obs + Instruction + (optional CoT)] 
        ↓ Qwen3-VL-4B (VLM backbone, 36-layer DiT 위)
        ↓ Hidden states + KV cache
        ↓ Control-query tokens (state-query + action-query, 학습 가능)
        ↓                                           ↓
 [Choice policy branch]                  [Knowledge truncation]
  N=5 candidate action                    DiT만 semantic-prefix
  chunks + scores                         KV cache만 attend
                                                    ↓
                                     [36-layer DiT, flow-matching]
                                                    ↓
                                       Continuous action chunk
```

### 2.2 3대 핵심 컴포넌트

| 컴포넌트 | 메커니즘 | 효과 |
|---|---|---|
| **Reasoning dropout (p_cot=0.5)** | 학습 중 50% 확률로 `<cot></cot>` 비워두고 동일 action 라벨 학습 | 추론 시 CoT 없이 action 생성 가능, AR compounding error 회피, CoT contamination 완화 |
| **Choice policy branch** | Auxiliary head가 action-query hidden state → N=5 후보 action chunk + 점수 (L1+score loss) | VLM backbone에 action-level discrimination 직접 주입 |
| **Knowledge truncation** | DiT가 attention할 때 control-query/state-query turn을 mask 제외, semantic prefix KV만 사용 | DiT가 synthetic control-query를 shortcut copy 못 함 |

### 2.3 학습 손실 종합

`L_total = L_flow (DiT) + L_choice (L1 + score) + L_CoT (next-token)`

세 손실 모두 end-to-end로 VLM backbone까지 역전파되지만, knowledge truncation으로 인해 DiT의 gradient는 **semantic prefix를 통해서만** VLM에 전달.

### 2.4 CoT Contamination 발견

Bounding box, end-effector coordinate처럼 **dense하지만 jitter가 심한** grounding label은 의미적으로 유사한 인접 프레임에 inconsistent supervision을 주어 학습을 망가뜨림. 해결: (a) 향상된 annotation pipeline, (b) 불안정 field는 sparse 감독, (c) reasoning dropout이 contamination-resistant scaffolding 역할.

---

## 3. 데이터셋

### 3.1 Pre-training Corpus (역대 최대 embodied-CoT)

| 데이터셋 | Trajectory | Sample | Sampling weight |
|---|---:|---:|---:|
| AgiBot | 765.6K | 204.9M | 0.518 |
| DROID | 74.7K | 15.1M | 0.180 |
| Fractal (RT-1) | 86.7K | 3.7M | 0.120 |
| BridgeData V2 | 43.8K | 1.4M | 0.100 |
| MolmoAct | 7.9K | 1.1M | 0.082 |
| **합계** | **978.7K** | **226.3M (2592.5 hours)** | 1.00 |

### 3.2 CoT 필드 구조 (Understanding/Grounding/Planning/Acting)

- Task understanding (글로벌 plan)
- Object grounding (멀티뷰: cam0/cam1 별도 bbox)
- Subtask decomposition + subtask reasoning
- Movement description (액션-언어)
- Gripper pose / future waypoint trajectory (2D image space)

### 3.3 Post-training

| Dataset | Trajectory | Sample |
|---|---:|---:|
| VLABench | 5,000 | 575,101 |
| LIBERO-10 | 500 | 138,090 |
| LIBERO-Goal | 500 | 63,728 |
| LIBERO-Object | 500 | 74,507 |
| LIBERO-Spatial | 500 | 62,250 |

---

## 4. 실험 결과 (Paper Table 3, 4, 14 직접 확인)

### 4.1 LIBERO-Plus (Table 3, zero-shot from LIBERO)

| Method | Spatial | Object | Goal | Long | Camera | Robot | Lang | Light | BG | Noise | Layout | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECoT | 31.8 | 27.9 | 30.6 | 8.6 | 0.3 | 26.8 | 40.2 | 42.6 | 16.4 | 10.2 | 36.9 | 24.3 |
| OpenVLA-OFT | 84.0 | 66.5 | 63.0 | 66.4 | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | 60.7 | 61.4 | 44.9 | 48.4 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.8 | 53.6 |
| π0-FAST | 74.4 | 72.7 | 57.6 | 43.4 | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| PokeVLA | 85.4 | 81.8 | 77.6 | 72.7 | 84.7 | 46.1 | 84.8 | 94.6 | 82.6 | 89.8 | 77.2 | 79.3 |
| π0.5 | 90.4 | 89.9 | 81.0 | 80.8 | 71.7 | 75.5 | 85.9 | 96.1 | 95.7 | 86.4 | 87.5 | 85.5 |
| **ERVLA** | **96.2** | **89.6** | **79.6** | **82.1** | **77.2** | **75.3** | **87.1** | **95.1** | **94.7** | **92.3** | **86.4** | **86.9** |

→ ERVLA가 π0.5 대비 **+1.4** 총점; Spatial track에서 background/lighting 변동에 대해 **100%** 성공.

### 4.2 VLABench (Table 4, full post-training)

| Method | In-dist SR | Cross Category SR | Commonsense SR | Instruction SR | Texture SR | **Avg SR** | Avg PS | Avg IS |
|---|---|---|---|---|---|---|---|---|
| π0 | 47.0 | 21.2 | 29.1 | 17.3 | 32.2 | 29.4 | 44.1 | 55.0 |
| π0-FAST | 56.2 | 31.0 | 38.0 | 35.0 | 39.0 | 39.8 | 49.5 | 58.6 |
| X-VLA | — | — | — | — | — | — | 51.1 | — |
| ACoT-VLA | — | — | — | — | — | — | 47.4 | 63.5 |
| π0.5 | 65.4 | 38.2 | 43.9 | 48.2 | 44.9 | 48.1 | 62.3 | 64.9 |
| **ERVLA** | **69.7** | **47.0** | **44.0** | **58.0** | **47.4** | **53.2** | **65.9** | **70.4** |

→ π0.5 대비 **+5.1 SR**, Instruction track 특히 **+9.8** 개선.

### 4.3 Real-Robot (Table 14, 20 tasks × 5 trials)

| Method | Basic SR | Distractor SR | Semantic SR | Long-horizon SR | **Avg SR** | Avg PS |
|---|---|---|---|---|---|---|
| ECoT | 60 | 18 | 10 | 6 | 24 | 35 |
| WorldVLA | 78 | 28 | 18 | 12 | 34 | 47 |
| UniVLA | 76 | 31 | 22 | 18 | 37 | 50 |
| π0.5 | 97 | 45 | 31 | 35 | 53 | 60 |
| **ERVLA** | **96** | **44** | **42** | **38** | **55** | **67** |

→ Basic/Distractor는 π0.5와 동급, **Semantic (+11)** 과 **Long-horizon (+3 SR, +17 PS)** 에서 우위 — CoT 내재화의 효용이 분포 밖에서 두드러진다는 가설을 강하게 지지.

---

## 5. Ablation 핵심

### 5.1 컴포넌트별 기여 (LIBERO-Plus / VLABench Avg SR)

| Variant | LIBERO-Plus | VLABench |
|---|---|---|
| No Choice (E2E, DiT gradient → VLM 전체) | 61.9 | 39.2 |
| No CoT (CoT supervision 제거) | 70.8 | 40.9 |
| No Choice + Knowledge Insulation (ThinkAct-style) | 76.5 | 42.6 |
| Choice + No Knowledge Truncation | 84.7 | 47.2 |
| **Full ERVLA** | **86.9** | **53.2** |

→ **Choice branch 단독 효과 ≈ +25 pt** (E2E → full), **knowledge truncation 효과 ≈ +2.2 pt** (KT 없음 → full). Choice가 dominant 기여.

### 5.2 CoT field-level (Appendix Table 11 / Sec 3.1 요약)

- **Low-level action-centric field** (movement description, 2D image-space end-effector trajectory) → 대부분의 이득
- **High-level reasoning만** (subtask decomposition + planning) → marginal gain
- **불안정한 dense grounding** (jittery bbox) → reasoning dropout 없으면 contamination

### 5.3 Scaling (Sec 3.2 Figure 4)

- ERVLA: CoT 데이터 증가에 따라 LIBERO-Plus / VLABench 모두 **꾸준히** 성능 향상
- AR CoT + FAST (explicit prefix): 일정 규모 후 **saturate** 또는 **퇴화**
- VLM + DiT (no CoT): saturate

---

## 6. Related Work 비교

| 접근 | Reasoning ↔ Action 결합 |
|---|---|
| ECoT (Zawalski et al.) | Reasoning = AR action prefix → slow, brittle |
| Emma-X | AR multi-modal reasoning → 동일 한계 |
| ThinkAct / Latent reasoning (UniVLA) | Latent plan으로 distillation → 표현 흐려짐 |
| WorldVLA | Visual prediction = reasoning → semantic ambiguity에 약함 |
| π0.5 | VLM-conditioned DiT + 사전학습 reasoning → 강력하지만 CoT를 활용한 표현 reshaping은 없음 |
| **ERVLA** | **Training-only CoT + choice branch + KT** → inference 시 CoT 없이도 reshape된 representation 활용 |

---

## 7. Limitations

1. **Single-arm 위주 평가**: pre-train에 AgiBot (bimanual) 포함되지만 main evaluation은 single-arm. Bimanual real-world 검증 부족.
2. **Choice branch 하이퍼파라미터** (N=5): 다른 N에 대한 ablation 없음.
3. **Reasoning dropout p=0.5**: 다른 p에 대한 sweep은 제한적.
4. **CoT 데이터 annotation pipeline**: 자동 라벨링 정확도가 contamination에 영향 — 라벨 노이즈 정량화 부족.
5. **VLM backbone 의존**: Qwen3-VL-4B 외 다른 backbone (Table 8 transfer study는 있으나 ERVLA full system 변경 X).

---

## 8. 종합 평가

| 항목 | 평점 (5점) |
|---|---|
| 혁신성 | 4.5 (CoT를 학습 신호로 재해석 + 3종 컴포넌트 결합) |
| 재현성 | 4.0 (code/checkpoint 공개 예정, 풀 하이퍼파라미터 명시) |
| 실험 폭 | 5.0 (sim 2개 + real-robot 4 tier + scaling 분석 + 4가지 ablation variant) |
| 이론적 깊이 | 4.0 (CoT contamination 개념 + knowledge truncation 정당화) |
| 실용성 | 4.5 (inference 시 CoT 미사용 → latency 우위, 86.9% LIBERO-Plus) |

**총평**: "Reasoning은 출력이 아니라 학습 신호"라는 단순하면서도 강력한 주장과 그를 뒷받침할 3가지 구체적 메커니즘(reasoning dropout + choice policy + knowledge truncation), 그리고 226M 샘플 corpus + 4가지 ablation variant + 4-tier real-world 검증으로 입증. **embodied CoT 분야의 새 기본선**이 될 가능성이 매우 높다.

---

## 9. 예상 세미나 질문

> ❓ **Reasoning dropout p=0.5는 일반 dropout과 어떻게 다른가?**
> 일반 dropout은 hidden state를 랜덤하게 0으로 마스킹. Reasoning dropout은 **input의 CoT 토큰 전체 span을 통째로 비우면서 같은 action 라벨을 학습** — explicit reasoning을 mandatory prefix로 의존하지 못하게 만들고 backbone hidden state로 정보를 내재화하도록 강제.

> ❓ **Knowledge truncation이 단순히 control-query token을 attention 마스크하는 건데 왜 그렇게 중요한가?**
> Choice branch가 action-query token으로 forward되면서 hidden state에 action-related representation을 만드는데, 만약 DiT가 그 토큰의 KV에 직접 접근하면 **"choice가 이미 정답을 알려주니까 그걸 copy하자"** 식의 shortcut learning이 발생. KT는 DiT를 semantic prefix(원본 obs + 명령어 + CoT)에만 attend시켜 의미적 정보를 통해서만 action을 풀게 강제. LIBERO-Plus 84.7 → 86.9, VLABench 47.2 → 53.2.

> ❓ **No CoT가 No Choice보다 좋은 게 좀 이상하다 — CoT가 정말 효과가 있나?**
> Yes. No Choice는 DiT의 flow loss가 VLM 전체에 역전파되어 representation을 망가뜨림. No CoT는 choice + KT는 유지하므로 action-relevant supervision은 살아있음. 그러나 Full ERVLA vs No CoT를 보면 LIBERO-Plus +16.1, VLABench +12.3 → CoT는 명백히 기여. 단, **CoT의 형태**가 (high-level 만은 marginal, low-level 포함이 critical) 중요.

> ❓ **AgiBot이 weight 0.518로 dominant인데 single-arm 평가에서도 효과가 있을까?**
> 60-dim padded action vector로 single-arm/bimanual을 통합. AgiBot의 dual-arm trajectory는 한쪽 arm만 활용해도 풍부한 contact/semantic 신호 제공. LIBERO/VLABench single-arm 평가에서 ablation 없이 그대로 강한 결과 → 효과적인 transfer.

> ❓ **Choice policy의 N=5는 어떻게 정했고 더 키우면?**
> Table 9에 N=5 명시되지만 다른 N에 대한 ablation은 없음. 향후 연구 과제. 직관적으로는 모드 수가 적은 task에서는 N=2-3로 충분, 다중 모드 dexterous task에서는 N 확장 필요할 수 있음.

---

## 10. 코드 & 재현

- **Project page**: https://taoshuaiz.github.io/ERVLA/
- **Code / Data / Checkpoint**: 공개 예정 ("will be released")
- **VLM backbone**: Qwen3-VL-4B-Instruct (공개)
- **Pre-train**: 120K steps, batch 64, LR 5e-5, bfloat16, DeepSpeed, 17.6K-token packing
- **Post-train**: LIBERO 10K steps, VLABench 20K steps, LR 1e-4 → 5e-7 cosine, action horizon 10

---

## 11. 데이터셋 / 후속 연구

- **Embodied-CoT corpus (226M 샘플)**: 향후 모든 reasoning-VLA 연구의 표준 pre-train data가 될 가능성
- **후속 방향**:
  - Bimanual + dexterous hand에서의 검증
  - Choice branch의 N 적응 (task-dependent N)
  - CoT contamination에 robust한 unsupervised label cleaning
  - 다른 VLM backbone (Gemma, Llama, InternVL)에서 동일 recipe 검증

---

## 12. 결론

ERVLA의 세 기여:
1. **개념적**: Embodied CoT를 "test-time 발화"가 아닌 **"학습 시 표현 공간을 reshape하는 신호"** 로 재정의.
2. **방법론적**: Reasoning dropout + choice policy + knowledge truncation 세트로 explicit CoT 의존 없이 reasoning의 이득을 흡수.
3. **실증적**: 226M 샘플 corpus + LIBERO-Plus 86.9 / VLABench 53.2 SR / Real-robot 55 SR (semantic·long-horizon에서 π0.5 우위)로 입증.

ECoT 연구의 분기점이 될 만한 논문. 특히 "**CoT는 출력이 아니라 학습 신호**" 라는 framing은 향후 다른 모달리티 (visual reasoning, world model reasoning)에도 그대로 이식될 수 있다.

---

<!-- VERIFIED: pdf -->
