# CAC-VLA: Context-Gated Action Conditioning for Vision-Language-Action Models

> **한 줄 요약**: π0.5 아키텍처에 VLM 쿼리 토큰 8개가 OAT(Ordered Action Tokenizer)로 인코딩된 미래 행동 세그먼트의 raw latent를 예측하도록 학습시키고, 이를 context gate(cross-attention retrieval + channel-wise sigmoid gate)로 action expert 각 레이어에 적응적으로 residual 주입하여 LIBERO 평균 98.3%, LIBERO-Plus SFT 89.5%를 달성한 VLM-native latent-action conditioning VLA. USTC 발표, 2026-07-06 arXiv 공개.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 대부분의 VLA(π0, OpenVLA, GR00T-N1 등)는 **VLM의 visual-language representation을 그대로 action expert의 conditioning 인터페이스**로 사용 → 이 표현은 행동 조건화(action conditioning)를 위해 명시적으로 최적화된 것이 아님
- 결과적으로 action expert가 task-level 행동 구조와 fine-grained motor command를 **암묵적으로 동시에 추론**해야 하는 부담
- 최근 action-reasoning 계열(ACoT-VLA 등)은 explicit action plan / reference trajectory / action-space prior를 생성해 효과를 보였으나, **별도의 action generator/reasoner 모듈**이 필요하고, 생성된 guidance를 **고정 강도의 conditioning**으로 취급

### 핵심 질문
- **VLM 자체에 명시적인 latent-action 인터페이스를 심을 수 있는가?** (별도 action reasoner 없이)
- **action expert가 이 action-structured guidance의 영향력을 장면·태스크 단계·생성 상태에 따라 적응적으로 조절할 수 있는가?**

📌 [Figure 1 삽입] — CAC-VLA 파이프라인: VLM(query token 포함) → OAT latent 정렬(학습 시) → Context-Gated Action Conditioning → flow-matching action expert

---

## 2. 방법론 심층 분석

### 2.1 문제 정식화
표준 VLA: a_{t:t+He-1} = π_θ(o_t, l). CAC-VLA는 중간에 latent action z_t를 도입:
- z_t = f_θ(o_t, l) (VLM-side latent-action predictor)
- a_{t:t+He-1} = π_θ(o_t, l, z_t)

### 2.2 VLM-native Latent Action Prediction
| 요소 | 세부사항 |
|------|---------|
| Supervision | 미래 행동 세그먼트 a_{t:t+Hl-1}을 frozen **OAT(Ordered Action Tokenizer)**로 인코딩한 raw latent z^oat |
| Query 토큰 | 학습 가능한 latent query 토큰 q (Nq=8)를 VLM 입력에 append |
| 예측 헤드 | ẑ_t = W_z LN(h^q_t) — LayerNorm + linear projection (raw OAT latent dim Dz=4) |
| Loss | 토큰별 마스크된 **Smooth-L1** alignment loss (식 6) |
| Attention mask | **expert-aware mask**: action expert 토큰이 query 토큰에 직접 attend 하지 못하게 차단 — query의 영향은 오직 예측된 latent action과 gating 모듈을 경유 |

- latent horizon Hl은 expert horizon He(=10 고정)와 독립적으로 설정 가능 → LIBERO는 Hl=20, LIBERO-Plus는 Hl=10
- coarse-to-fine 구조를 가진 OAT latent가 다양한 시간 범위의 행동 구조를 요약

### 2.3 Context-Gated Action Conditioning (핵심 기여)
각 action expert 레이어에서 self-attention 이후, FFN 이전에 삽입:
1. **Projection**: latent action → expert 호환 conditioning 토큰 c_t = φ_c(z^c_t)
2. **Retrieval**: u^l = CrossAttn(Q = Norm(x^l), K = V = c_t)
3. **Gating**: s_x = AvgPool(x̄^l), s_u = AvgPool(u^l); g^l = σ(W_g tanh(W_x s_x + W_u s_u)) — **channel-wise gate** (모든 action 토큰 공유)
4. **주입**: x^{l+} = x^l + g^l ⊙ u^l (gated residual)

핵심 설계 철학: **retrieval과 injection의 분리**. cross-attention이 현재 action state 조건으로 정보를 검색하고, gate가 주입 강도를 결정 → 고정 fusion 대비 expert의 연속 행동 모델링 능력을 보존.

### 2.4 학습·추론 비대칭
- **학습**: conditioning source = 인코딩된 OAT latent 타깃(안정적 supervision), L = L_act(flow-matching) + λ_align·L_align (λ=0.1)
- **추론**: OAT tokenizer 제거, **VLM이 예측한 latent action만** 사용 → 배포 시 완전히 VLM-native

> ❓ **예상 질문**: LAPA/UniVLA의 latent action과 무엇이 다른가?
> **답변**: LAPA·UniVLA는 unlabeled/heterogeneous 비디오에서 latent action을 학습해 **pretraining·transfer**에 사용하고 latent를 중간 action space로 디코딩한다. CAC-VLA는 **로봇 행동 라벨에서 OAT로 인코딩한 ordered latent를 supervision**으로 쓰고, 디코딩 대상이 아닌 **연속 expert의 conditioning 신호**로만 사용하며, context gate로 residual 기여도를 조절한다.

---

## 3. 데이터 전략

- **LIBERO**: 표준 4-suite (Spatial/Object/Goal/Long), 40K step 학습
- **LIBERO-Plus**: 공식 training set으로 30K step SFT; zero-shot transfer 세팅도 별도 평가
- **실로봇**: UR7e + RealSense D435i 2대(wrist + third-person), 태스크당 50 demonstrations, π0.5 baseline과 완전히 동일한 데이터·모달리티로 fine-tune
- 별도의 대규모 사전학습 데이터 추가 없음 — π0.5 기반 SFT 중심 검증

---

## 4. 시스템/학습 세부사항

| 하이퍼파라미터 | 값 |
|---|---|
| Expert action horizon He | 10 |
| Latent-action horizon Hl | 20 (LIBERO) / 10 (LIBERO-Plus) |
| Latent query 수 Nq | 8 |
| Raw OAT latent dim Dz | 4 |
| Global batch size | 64 |
| Optimizer | AdamW, grad clip 1.0, EMA 0.999 |
| LR | peak 1.25e-5, cosine, warmup 10K, decay 1M, final 1.25e-5 |
| λ_align / conditioning dropout | 0.1 / 0.1 |
| 하드웨어 | NVIDIA A100 4장 |

- OAT tokenizer는 전 과정 frozen
- 학습 step: LIBERO 40K / LIBERO-Plus 30K

---

## 5. 실험 설계 및 평가 프로토콜

- **벤치마크**: LIBERO(4 suites) + LIBERO-Plus(7가지 분포 변화: camera, robot, language, light, background, noise, layout)
- **비교 프로토콜**: ACoT-VLA의 benchmark protocol·baseline collection을 따름 (27개 방법과 비교, guidance 유형을 Visual/Linguistics/Action으로 분류)
- **지표**: per-suite 및 평균 success rate
- **실로봇**: 태스크당 25 trials, Task Score(부분 성공 0.5점) + Full SR / Stacking SR

---

## 6. 실험 결과 심층 분석

### LIBERO (Table 1, SFT)
| Suite | CAC-VLA | ACoT-VLA | π0.5 | OpenVLA-OFT |
|---|---|---|---|---|
| Spatial | 98.4 | 98.6 | 98.8 | 97.6 |
| Object | **99.8** (1위) | 99.0 | 98.2 | 98.4 |
| Goal | **99.6** (1위) | 99.4 | 98.0 | 97.9 |
| Long | 95.4 | **97.0** | 92.4 | 94.5 |
| **Avg** | **98.3** (전체 2위) | **98.5** (1위) | 96.9 | 97.1 |

- Object·Goal에서 최고 성능, 평균은 ACoT-VLA(98.5)에 이어 2위
- 베이스인 π0.5(96.9) 대비 +1.4%p — latent-action conditioning의 직접 효과

### LIBERO-Plus (Table 2)
- **Zero-shot transfer**: 83.8 (π0.5* 81.5 대비 +2.3, 전 카테고리에서 우위)
- **SFT**: **89.5** — ACoT-VLA 88.0, π0.5* 85.7 대비 우위. light 97.5 / background 97.1 / noise 95.4에서 강함. 단 camera(91.2)는 ACoT-VLA(96.6)에 뒤짐

### 실로봇 (UR7e, 25 trials)
- Pick-and-place: Task Score 72% / Full SR **64%** (16/25) vs π0.5 48% / 16% (4/25) — Full SR +48%p
- Block stacking: Stacking SR 52% (13/25) vs π0.5 36% (9/25)

---

## 7. Ablation 분석

### Latent-action horizon (Table 3, LIBERO)
| Hl | Avg |
|---|---|
| 10 | 98.1 |
| **20** | **98.3** |
| 30 | 98.1 |

- 중간 길이 horizon이 최적: 더 긴 미래 문맥 vs 현재 action chunk와의 관련성 trade-off. Hl=30은 Long에서 97.2로 최고인 점이 흥미로움(장기 태스크엔 긴 horizon 유리)

### Conditioning 설계 (Table 4)
| Variant | Avg |
|---|---|
| w/o latent-action conditioning | 98.0 |
| w/o context gate (고정 residual fusion) | 97.9 |
| **Full CAC-VLA** | **98.3** |

- gate 제거가 conditioning 자체 제거보다도 낮음 → **잘못된 강도의 고정 주입은 오히려 해로울 수 있음**, 적응적 gating의 필요성 입증
- App. B 시각화: gate 값이 flow step·suite·manipulation phase(approach/grasp/move/drop)에 따라 변동 — 고정 강도가 아님을 확인. 단, 저자 스스로 "인과적 증거가 아닌 행동 분석"이라고 명시

---

## 8. 관련 연구 비교

| 방법 | Guidance 유형 | guidance 획득 방식 | 사용 방식 |
|---|---|---|---|
| CoT-VLA, DreamVLA, F1 | Visual | 미래 이미지/관측 생성 | 고정 conditioning |
| π0-FAST, ThinkAct 등 | Linguistics | 텍스트 plan/subgoal | 고정 conditioning |
| ACoT-VLA | Action | 별도 explicit/implicit action reasoner | 고정 conditioning |
| LAPA / UniVLA | Latent action | 비디오에서 비지도 학습 | pretraining용 중간 action space |
| **CAC-VLA** | **Action** | **VLM 자체가 OAT latent 예측** | **context gate로 적응적 residual 주입** |

- ACoT-VLA와의 차별점: (1) standalone reasoner 불필요 — VLM-native, (2) 고정이 아닌 gated conditioning
- 위치상 "action-level guidance의 경량·적응형 버전"에 해당

---

## 9. 한계 및 미해결 문제

1. **실로봇 검증 협소**: 단일 embodiment(UR7e), tabletop pick-and-place/stacking 2개 태스크, 태스크당 50 demos — 저자도 명시적으로 인정
2. **LIBERO 평균에서 ACoT-VLA에 미달** (98.3 vs 98.5, 특히 Long 95.4 vs 97.0) — 긴 horizon 태스크에서 explicit reference trajectory의 이점 잔존 가능
3. **고정 latent horizon**: Hl이 태스크 단계와 맞지 않으면 실패 가능 — task-adaptive/hierarchical horizon은 future work
4. **예측 latent 부정확 시 실패 모드**: gate가 완화하지만 근본 해결은 아님
5. OAT tokenizer 의존: latent 품질이 tokenizer 성능에 종속, tokenizer 자체 ablation 부재
6. 파라미터 수·추론 속도 등 효율성 지표 미보고

---

## 10. 총평

CAC-VLA는 "action-level guidance는 유용하지만 별도 모듈과 고정 주입은 부담"이라는 문제의식을 **VLM 쿼리 토큰 + frozen tokenizer supervision + gated residual**이라는 가벼운 조합으로 해결한 깔끔한 엔지니어링 기여다. 특히 학습 시 ground-truth latent로 conditioning하고 추론 시 예측 latent로 교체하는 비대칭 설계와, "retrieval(cross-attn)과 injection(gate)의 분리"는 재사용 가치가 높은 패턴이다. 성능은 LIBERO 최상위권(98.3%)이고 LIBERO-Plus SFT에서는 SOTA(89.5%)이나, LIBERO 절대 개선폭(π0.5 대비 +1.4%p, ablation 폭 0.3~0.4%p)은 포화 구간이라 통계적 여유가 크지 않다. 실질 설득력은 오히려 LIBERO-Plus robustness와 실로봇 Full SR +48%p에서 나온다.

---

## 11. 🔥 예상 날카로운 질문 모음

1. **Q**: LIBERO ablation 개선폭이 0.3~0.4%p인데 seed 분산 대비 유의한가?
   **A**: 논문에 seed 수·신뢰구간 미보고. 포화 벤치마크 특성상 LIBERO-Plus(+1.5~3.8)와 실로봇 결과가 더 신뢰할 만한 증거.
2. **Q**: 학습 시 GT OAT latent, 추론 시 예측 latent — train/inference mismatch가 문제되지 않는가?
   **A**: L_align이 예측을 GT latent 공간에 정렬시키고, conditioning dropout 0.1이 expert의 latent 과의존을 방지. 다만 mismatch 크기의 정량 분석은 부재.
3. **Q**: expert-aware attention mask 없이 query 토큰에 직접 attend하게 하면?
   **A**: 논문 주장으로는 query의 영향 경로를 gated conditioning으로 단일화해 gate의 calibration 의미를 보존하기 위함. 해당 ablation은 미제공.
4. **Q**: camera perturbation에서 ACoT-VLA(96.6)에 크게 뒤지는(91.2) 이유는?
   **A**: explicit reference trajectory가 시점 변화에 더 강건할 가능성. latent 공간 guidance는 시각 표현 변화에 민감할 수 있음 — 논문은 원인 분석 없음.
5. **Q**: Dz=4라는 극단적으로 작은 latent 차원으로 충분한가?
   **A**: OAT의 coarse-to-fine ordered 구조 덕분에 압축 효율이 높다는 것이 전제. Nq=8×Dz=4=32차원 요약이지만, 차원 ablation은 미보고.

---

## 12. 결론

CAC-VLA는 VLA의 action conditioning 인터페이스를 "암묵적 VL representation"에서 "VLM-native 예측 latent action + 적응적 gating"으로 옮긴 연구로, 별도 action reasoner 없이 LIBERO 98.3% / LIBERO-Plus SFT 89.5%를 달성했다. context gate의 phase-adaptive 거동 분석까지 포함해 설계 근거가 잘 정리되어 있으며, latent-action 기반 conditioning 계열의 실용적 기준점이 될 만하다. 다음 단계는 다양한 embodiment로의 확장과 task-adaptive latent horizon이다.

<!-- VERIFIED: pdf -->
