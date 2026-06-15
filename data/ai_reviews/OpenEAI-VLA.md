# OpenEAI-VLA: Fully Open-Source Hardware-Software Unified Embodied AI Platform

> **한 줄 요약**: Qwen3-VL-4B 백본 + 18-layer DiT flow-matching action head를 **learnable query token + dataset adapter + multimodal co-training** 세 가지 설계 선택으로 묶어, **Open-X-Embodiment만으로 사전학습**하면서도 4개 실제 manipulation task 평균 0.82를 달성한 풀스택 오픈소스 VLA. 동일 데이터·동일 아암에서 ACT(0.36), Octo(0.05), OpenVLA-oft(0.24), π0(0.78)과 비교해 π0를 평균적으로 앞서고 π0.5(0.87)에 근접.

---

## 1. 배경 및 동기

- Physical Intelligence의 π0 / π0.5는 SOTA지만 **사전학습 데이터와 인프라가 비공개**여서 재현이 불가능하다.
- 한편 공개 로봇 데이터셋들은 state/action convention이 제각각인 "data islands" — 단순히 합쳐 학습하면 collapse한다 (Table 5: shared adapter → 0.00).
- 하드웨어도 ARX R5/Piper 같은 상용 6+1-DoF arm은 7,000~40,000 USD에 black-box로 제공돼 low-level 제어 연구를 막는다.
- 저자들의 입장: **VLA 진보는 모델 디자인이 아니라 end-to-end 재현성**이 결정한다. 따라서 mechanical design, low-level controller, dataset 파이프라인, 학습 레시피, checkpoint까지 전부 공개하는 **OpenEAI-Platform**을 제안. 본 리뷰는 그중 정책 모델 **OpenEAI-VLA**에 집중한다.

---

## 2. 방법론

### Architecture (Sec. 3, Fig. 2)

- **Backbone**: Qwen3-VL-Instruct 4B (frozen by default during pretraining).
- **Learnable feature query** (핵심 디자인): 길이 M의 학습가능 토큰 sequence `q ∈ R^{M×d}`를 이미지/텍스트 토큰 뒤에 append하고 VLM을 통과시킨 뒤, **마지막 layer의 query 위치 hidden state h_q만** action head로 전달.
  - 효과: action head 계산량이 이미지 패치/텍스트 토큰 수와 **독립적**이 되는 fixed-bandwidth bottleneck. Query token들이 "action에 필요한 정보가 무엇인지" 압축하도록 학습.
- **Action expert**: 18-layer DiT, layer당 32 attention head. 입력은 (i) noisy action chunk A_t^τ, (ii) 현재 robot state s_t, (iii) h_q. 출력은 conditional flow matching velocity field v_θ.
- **Dataset adapters**: OXE의 이질적 state/action convention을 공유 공간으로 매핑하는 dataset-specific projector. Ablation에서 이걸 단일 shared adapter로 바꾸면 **성공률 0.00**으로 붕괴 (Table 5) — heterogeneous data 정렬이 사전학습 핵심.

### Two-stage training recipe (Sec. 3)

**Stage 1 — Pretraining on OXE only**
- VLM 완전 frozen, learnable query + dataset adapter + DiT action head만 학습.
- π0-style conditional flow matching:
  ```
  L_EM(θ) = E[‖v_θ(A_t^τ, o_t, h_q, τ) − (ε − A_t)‖^2]
  ```
  where τ ~ Beta(·) (noisier timestep emphasis), ε ~ N(0, I).
- 의도: VLM의 multimodal prior를 보존하면서 embodied control만 입힌다.

**Stage 2 — Fine-tuning with multimodal co-training**
- 로봇 demo Br + multimodal sample Bm (COCO + VQA-v2 + PixMo-Points) 혼합 배치.
- Joint loss: `L = λ_EM · L_EM(Br) + λ_VLM · L_NTP(Bm)`.
- 효과: 로봇 데이터만으로 fine-tune하면 spatial grounding이 무너진다 (Table 5: w/o multimodal data → 0.30).
- VLM unfreezing은 **늦게** 풀어야 함 (early unfreezing 0.52 vs late 0.72).

### Hardware/control은 별도 컨트리뷰션

OpenEAI-Arm은 NSGA-III로 manipulation operability F1과 endurance efficiency F2를 동시 최적화한 MDH 파라미터 + FF-PID + Bézier action-chunking interpolation으로 구성된 790 USD짜리 ARX R5 형태 6+1-DoF arm. 본 리뷰의 초점은 아니지만, "policy comparison을 동일 하드웨어에서 수행 가능" 자체가 reproducibility 주장의 토대.

---

## 3. 실험 결과

### Hardware suitability (Table 2, π0 policy 고정)

| Task | ARX R5 | Piper | OpenEAI-Arm |
|---|---|---|---|
| Clean Table | 0.88 | 0.86 | **0.92** |
| Make Tea | 0.40 | 0.40 | **0.60** |
| Fold Towel | 0.73 | **0.80** | 0.73 |
| Fold T-shirt | **0.83** | 0.50 | 0.75 |
| **Average** | 0.71 | 0.64 | **0.75** |

→ 790 USD arm이 8,600 USD ARX R5보다 평균 성공률 우위. OpenEAI-Arm을 이후 모델 비교의 기준 플랫폼으로 사용 정당화.

### Model comparison on OpenEAI-Arm (Table 3, 동일 fine-tune 데이터)

| Model | Clean Table | Make Tea (final) | Fold Towel (final) | Fold T-shirt (final) |
|---|---|---|---|---|
| ACT | 0.72 | 0.6 | 0.33 | 0.0 |
| Octo | 0.2 | 0.0 | 0.0 | — |
| OpenVLA-oft | 0.68 | 0.0 | 0.27 | 0.0 |
| π0 | 0.92 | 0.6 | 0.73 | 0.75 |
| π0.5 | **0.96** | **0.8** | 0.8 | **0.83** |
| **OpenEAI-VLA (Ours)** | 0.94 | 0.7 | **0.8** | **0.83** |

→ OXE만으로 사전학습한 OpenEAI-VLA가 **대규모 비공개 데이터로 학습된 π0를 평균적으로 추월**하고 π0.5와 Fold T-shirt에서 동률. Octo·OpenVLA-oft 같은 초기 generalist는 precision·contact-rich task에서 거의 작동하지 않음.

### Ablation (Table 5)

| Variant | Avg Success |
|---|---|
| OpenEAI-VLA (full) | **0.82** |
| w/o dataset adapter | 0.00 |
| w/o multimodal data (FT) | 0.30 |
| w/ Qwen2.5-VL backbone | 0.51 |
| w/ UMI data added | 0.61 |
| w/ relative action space | 0.67 |
| w/ early VLM unfreezing | 0.52 |
| w/ late VLM unfreezing | 0.72 |

→ **dataset adapter > multimodal co-training > backbone 선택** 순으로 영향. UMI 데이터 추가가 오히려 평균을 떨어뜨린 것은 "diversity만으로는 부족하고 careful mixture alignment가 필요"하다는 저자 진술과 일치.

### Control ablation (Table 4)

Chunk-level Bézier smoothing + FF-PID dynamics compensation 둘 다 필수. Dynamics feedforward를 제거하면 acceleration error가 약 70배, torque error가 3.4배 폭증.

---

## 4. 한계 및 미해결 문제

1. **시뮬레이션 벤치마크 부재**: LIBERO/CALVIN/SimplerEnv 등 표준 시뮬레이션 결과가 없다. 모든 비교가 자체 4-task 실제 평가로만 이뤄져 cross-paper 비교가 어렵다.
2. **π0.5에 약간 미치지 못함**: 평균 0.82 vs π0.5 0.87. 저자들은 π0.5의 "substantially larger and higher-quality pretraining corpus" 때문이라 인정 — 즉 OXE만으로는 절대 성능 천장이 있다.
3. **OOD 일반화 미검증**: 평가는 in-domain fine-tune 분포에서만 진행. Cross-embodiment, unseen object, scene-level shift에 대한 정량 실험은 future work로 미룸.
4. **Multimodal co-training 비율 가이드 부재**: λ_EM / λ_VLM 선택과 COCO/VQA/PixMo 혼합 비율이 task에 따라 어떻게 바뀌어야 하는지 systematic study가 없다.
5. **코드 공개 시점**: 논문 abstract에 "released after the paper is accepted"라고 명시 — 리뷰 작성 시점(2026-02-25 draft) 기준 일부 자산은 아직 비공개.

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Learnable query token, dataset adapter, multimodal co-training 모두 개별적으로는 기존 기법. 기여는 "이걸 정확히 어떻게 묶어야 OXE만으로 π0 급이 나오는지"의 systematic recipe. |
| **Practical impact** | ★★★★★ — 790 USD arm + Qwen3-VL-4B + 공개 데이터만으로 π0를 따라잡는 stack 전체를 공개. 학계·소규모 lab의 실험 진입장벽을 크게 낮춤. |
| **Reproducibility** | ★★★★★ — Hardware BOM, low-level driver, dataset pipeline, training script, checkpoint까지 모두 공개 약속. "data island" 문제를 dataset adapter로 명시적으로 해결. |

OpenEAI-VLA는 알고리즘 혁신보다 **"무엇을 공개해야 VLA가 재현 가능한가"라는 질문에 가장 정직하게 답하는 시스템 논문**이다. Ablation에서 dataset adapter 제거가 성공률을 0으로 떨어뜨린 결과는 OXE-style 멀티-임바디먼트 사전학습에서 명시적 alignment가 얼마나 critical한지를 보여주는 가장 깨끗한 증거 중 하나다. 모델 자체의 디자인 선택은 보수적이지만, "공개 데이터만으로 π0를 따라잡는다"는 결과는 그 자체로 강한 진술이며, 후속 연구가 이 stack을 baseline으로 삼기 좋다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | Learnable query token이 단순히 attention pooling과 무엇이 다른가? | (a) 길이 M이 고정돼 action head 계산이 입력 길이에 독립, (b) **학습가능 파라미터**라서 task-relevant 정보를 능동적으로 압축, (c) VLM 본체와 함께 backprop돼 robot-conditional 표현을 자체적으로 형성. 단순 average pooling은 이런 적응성이 없다. |
| 2 | Dataset adapter가 없으면 왜 0.00으로 collapse하는가? | OXE는 동일 task라도 데이터셋마다 action 단위·좌표계·gripper convention이 다르다. Shared adapter는 모든 sample을 동일 헤드로 쏘므로 평균적으로 mode-collapse하는 trajectory를 학습 — VLM이 어떤 dataset인지 distinguish할 신호가 없어진다. Dataset-specific projector는 각 데이터셋의 idiosyncrasy를 흡수해 공유 latent action space만 남긴다. |
| 3 | OXE만 써서 π0를 잡았다면 π0의 대규모 비공개 데이터는 가치가 없는가? | π0.5(0.87) vs Ours(0.82) 차이가 정확히 그 가치의 정량적 estimate. 즉 ~5%p의 성공률이 비공개 corpus의 마진. Precision·contact-rich task일수록 격차가 커지므로 일반적인 desktop manipulation에는 OXE가 충분하지만 long-horizon 일반화에는 부족. |
| 4 | Qwen3-VL-4B → Qwen2.5-VL 다운그레이드 시 0.82 → 0.51로 폭락하는 이유? | Action head와 dataset adapter는 동일한데 VLM의 multimodal alignment 품질만 바뀐 실험. 즉 VLA 성능은 **action head 디자인보다 backbone VLM의 vision-language 일관성**에 크게 의존한다는 강력한 증거. 신규 VLA 설계 시 backbone 선택이 가장 큰 hyperparameter임을 시사. |
| 5 | Multimodal co-training이 fine-tuning에서만 적용되는데 사전학습에서는 왜 빼는가? | 사전학습은 VLM frozen이라 next-token loss를 줄 필요가 없다. Fine-tuning에서는 robot data가 narrow distribution이라 VLM unfreezing 시 spatial grounding이 무너지는 catastrophic forgetting을 막기 위해 COCO/VQA/PixMo의 NTP loss를 보조 anchor로 쓴다. 빼면 0.82 → 0.30. |

<!-- VERIFIED: pdf -->
