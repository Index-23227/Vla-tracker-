# VLA-GSE: Boosting Parameter-Efficient Fine-Tuning in VLA with Generalized and Specialized Experts 세미나 리뷰

## 1. 연구 배경 및 핵심 아이디어

VLA-GSE(Jiang et al., 2026, arXiv:2605.06175)는 Vision-Language-Action(VLA) 모델을 로봇 제어에 적응시킬 때 발생하는 두 가지 고질적 문제 — (a) full fine-tuning(FFT)의 downstream overfitting 및 catastrophic forgetting, (b) 기존 LoRA 계열 PEFT의 robot task 적응 부족 — 를 동시에 해결하려는 PEFT 프레임워크다.

핵심 통찰은 frozen VLM 가중치의 **spectral 구조**를 활용하는 데 있다. SVD로 backbone weight를 분해한 뒤,
- **선행(leading) singular components**는 사전학습된 도메인-일반 지식을 담고 있다고 보아 항상 활성화되는 단일 *generalized expert* 로 할당하고,
- **잔여(residual) singular components**는 disjoint하게 분할하여 top-k 라우터로 token-level 선택되는 *specialized experts* 로 할당한다.

이 spectral routing 설계는 일반 지식 보존과 task-specific 적응 사이의 trade-off를 명시적으로 분리해 처리한다.

## 2. 아키텍처

**Backbone**: Qwen3-VL-4B-Instruct(약 4.55B parameters)를 frozen VLM으로 사용한다. README에서 확인되는 framework 명칭은 `QwenFM` / `QwenOFT`로, OpenVLA-OFT 계열의 parallel decoding action head를 Qwen 백본 위에 얹은 형태다.

**Action head**: 65.62M parameters의 회귀형 head(OFT-style parallel decoding). Diffusion/flow-matching이 아닌 single-pass regression이므로 action head category는 `regression`으로 분류된다.

**GSE PEFT 모듈** (각 target linear layer마다):
- Generalized expert 1개 (always-active, leading singular components로 초기화)
- Specialized experts 7개 (residual singular components로 disjoint init, top-k=2 라우팅)
- Rank r = 16, num_experts = 8, generalized_experts = 1, top_k = 2
- s_g = 2 (generalized expert spectral scaling), aux_loss_weight = 0.01 (load balancing)
- GSE module 총 48.41M parameters

**전체 trainable**: 114.04M / 4.55B = **2.51%** (GSE 48.41M + Action head 65.62M).

## 3. 학습 데이터 및 최적화

- **Backbone**: Qwen3-VL-4B-Instruct 사전학습 가중치 사용 (Hugging Face).
- **Datasets**: 표준 LIBERO 4개 suite(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)를 LeRobot format으로 로드하여 co-train.
- **Optimization**:
  - Per-GPU batch 16, 8×A100 → effective batch 128
  - Total steps 80,000 (~48시간)
  - lr_vlm = 1e-5, lr_action_head = 1e-4
  - DeepSpeed 기반 multi-GPU 학습
- **Auxiliary losses**: Top-k MoE의 expert utilization 균형을 위한 load-balancing loss (가중치 0.01).
- **Optional**: `gradient_scale_balancing` 모드에서 trace-inverse 기반 specialized scale 초기화 가능.

## 4. 핵심 실험 결과

논문은 두 평가 트랙을 보고한다:

- **LIBERO-Plus** (zero-shot generalization, Table 1/2): Camera/Robot/Language/Light/Background/Noise/Layout 7가지 perturbation 카테고리 평균 **81.2%** zero-shot success — abstract에서 확인된 단일 핵심 수치. 동일 budget의 LoRA·FFT를 모두 상회한다.
- **표준 LIBERO** (Table 6/7): Spatial/Object/Goal/Long(10) suite별 success rate를 보고하나 README 텍스트에는 수치가 명시되지 않아 본 리뷰에서는 omit. YAML에서도 per-suite 수치는 기재하지 않았다.
- **Ablation** (Table 3): `num_generalized_experts`, `init_type`(SVD vs random), `aux_loss_weight`를 변경하며 각 설계 요소의 기여도를 분리 검증.
- **Real-world manipulation**: 실로봇 평가에서도 FFT 대비 우위를 주장.

## 5. 비교 분석

본 논문이 비교한 baseline은 다음과 같다 (모두 동일 Qwen3-VL-4B 백본 위에서 재현):

- **Full Fine-Tuning (FFT)**: 전체 4.55B parameter 학습 — overfitting/forgetting에 취약.
- **LoRA**: native PEFT 라이브러리 LoraConfig 사용.
- **GOAT**: gated MoE-LoRA — 본 논문의 직접적 비교 대상 (MoE-LoRA 계열 SOTA).
- **LoRA family**: rsLoRA, DoRA, PiSSA, MoLoRA, AdaMoLE, HydraLoRA, MiLoRA — 8종 PEFT 변형과 일괄 비교.
- **FFT → GSE**: FFT checkpoint에서 GSE를 이어 학습하는 2-stage 변형도 제공.

VLA-GSE의 차별점은 (a) SVD 기반 spectral 초기화로 expert 간 정보 분리가 explicit, (b) generalized expert가 항상 활성화되어 catastrophic forgetting 완화, (c) 동일 rank·동일 trainable budget 하에서 LoRA/MoLoRA/HydraLoRA 등을 능가한다는 점이다.

## 6. 평가 및 한계

**강점**:
- Spectral decomposition을 PEFT의 expert 초기화에 직접 결합한 첫 시도 중 하나로, MoE-LoRA의 무작위 초기화 한계를 정공법으로 해결.
- 2.51% trainable로 FFT를 능가 — 학습 비용·메모리 효율과 generalization을 동시에 확보.
- 8 PEFT baseline + GOAT를 단일 코드베이스에서 재현 가능 — 공정 비교 측면에서 가치 큼.
- MIT 라이선스 오픈소스 (https://github.com/YuhuaJiang2002/VLA-GSE).

**약점**:
- LIBERO 표준 suite별(Spatial/Object/Goal/Long) 수치가 README/abstract에서 직접 확인되지 않아 leaderboard `libero_avg`를 abstract의 **LIBERO-Plus** 수치(81.2)로 채웠다. 표준 LIBERO 수치는 PDF의 Table 6/7을 직접 검증한 뒤 보강 필요.
- Backbone이 Qwen3-VL-4B로 단일 — 다른 VLM(예: PaliGemma, Llama-VL)에서의 일반성 미검증.
- LIBERO-Plus 7-perturbation 평균만 강조되어, 어떤 perturbation에서 약한지 카테고리별 breakdown 검증 필요.
- Real-world 실험은 abstract/README 수준에서만 언급, 정확한 task-by-task 결과는 PDF 확인 필요.

**YAML 점검**:
- `parameters: 4.55B`는 Qwen3-VL-4B-Instruct 풀모델 크기 기준 (trainable 114.04M = 2.51%).
- `action_head_category: regression`은 QwenOFT framework가 OpenVLA-OFT 계열 parallel decoding head임에 근거 — diffusion/flow matching 사용 시사 없음.
- `eval_conditions.libero`에 fine-tuned(표준 LIBERO) + zero-shot(LIBERO-Plus) 양쪽을 명시.
- `tags`에 `peft`, `mixture-of-experts`, `lora`, `svd`, `qwen3-vl` 추가하여 검색성 강화.
- 표준 LIBERO 4-suite 점수가 PDF에서 확인되면 `libero_spatial/object/goal/long`을 보강하고 `libero_avg`를 표준 평균으로 교체할 것 (현재는 LIBERO-Plus 평균이 들어가 있음을 명시).

<!-- VERIFIED: abstract-only -->
