# LIBERO-Occ & VIM: Viewpoint Imagination for VLA under Scene-Induced Occlusion

**Paper**: LIBERO-Occ: Evaluating and Improving Vision-Language-Action Models under Scene-Induced Occlusion via Viewpoint Imagination
**arXiv**: 2606.10862 (2026-06-09)
**Authors**: Taishan Li, Jiwen Zhang, Siyuan Wang, Xuanjing Huang, Zhongyu Wei (Fudan University, Shanghai Innovation Institute, CUHK)
**Code**: https://github.com/litsh/Libero-Occ

---

## 1. 한 줄 요약

표준 LIBERO 벤치마크가 대상 물체가 항상 보인다고 가정하는 비현실성을 비판하고, 물리적으로 그럴듯한 가림(scene-induced occlusion)을 도입한 **LIBERO-Occ** 벤치마크(2,000 task)와, 가려진 주 시점에서 보완 시점을 *상상*(generative)해 행동을 예측하는 **VIM(Viewpoint Imagination)** 정책을 함께 제안한다.

## 2. 문제 정의

기존 VLA 모델들은 LIBERO/CALVIN 같은 벤치마크에서 task-relevant 객체가 카메라에서 완전히 보인다고 암묵적으로 가정한다. 그러나 실제 manipulation에서는 (a) 인접 물체에 가려짐, (b) 열린 서랍·문에 가려짐, (c) 동작 중인 로봇 팔에 의한 self-occlusion 등으로 인해 핵심 단서가 부분 관측만 가능하다. 저자들은 이를 *scene-induced occlusion*으로 정의하고, 외형을 인위적으로 바꾸는 visual perturbation(LIBERO-Plus, LIBERO-PRO)과 구분한다 — 가림은 텍스처를 보존하면서 *증거 자체*를 제거한다.

## 3. LIBERO-Occ 벤치마크

### 3.1 자동 생성 파이프라인 (3 step)
1. **Occlusion Target Identification**: BDDL 파싱으로 task-relevant entity(manipulated object, receptacle, goal region)를 식별.
2. **View-aware Occluder Placement**: 카메라-타깃 ray를 따라 3D 공간에서 occluder 후보 위치를 샘플링; LIBERO object library에서 의미상 충돌 없는 occluder 선택.
3. **Occlusion Validity Verification**: (1) Visibility check(충분히 가리지만 완전 불가시는 아님), (2) Physical validity(충돌 없음), (3) Task executability(원본 demonstration 재생이 성공해야 retain).

### 3.2 두 축의 카테고리화
- **Occlusion Type** (3): Manipulated object (900 tasks), Receptacle (750), Dual (350).
- **Severity** (3): per-suite quartile로 분할 — Light 500, Medium 1000, Heavy 500. 시비어리티 점수 S_occ = (A_full − A_visible) / A_full.
- 총 4개 suite × 500 occluded instance = **2,000 task**.

## 4. VIM 정책 (Viewpoint Imagination)

### 4.1 공식화
기존 VLA: p_θ(a_t | o_t, l). VIM은 다음으로 확장:

`p_θ(ô_c_t, a_t | o_t, l) = p_θ(ô_c_t | o_t, l) · p_θ(a_t | o_t, ô_c_t, l)`

가려진 주 시점 o_t와 instruction l이 주어지면, 모델이 먼저 보완 시점 ô_c_t의 visual token을 생성한 뒤, 두 시점을 모두 조건으로 action token을 예측한다.

### 4.2 아키텍처
- 백본: **Emu3-MoE** (autoregressive transformer, unified image+text+action token generation).
- 기반: UniVLA의 world model 위에 구축.
- Action tokenizer: **FAST** (Pertsch et al., 2025).
- 주 시점: third-person, 보완 시점: wrist/gripper view (학습 시에만 ground-truth로 제공; 추론 시 모델이 생성).

### 4.3 2-Stage 학습
- **Stage 1 (View Pretraining)**: 가려진 front view + instruction → 보완 시점 토큰 생성만 학습. cross-view spatial correspondence 습득.
- **Stage 2 (Joint Optimization)**: L = L_action + λ L_view (λ = 0.5). view loss는 단순한 보조 reconstruction이 아니라 imagination-to-action 인터페이스를 *형식상 유지*시키는 structural regularizer 역할.

### 4.4 학습 설정
8× H100 GPU, batch size 192, cosine LR, peak LR = 8e-5 (Stage 1) / 4e-5 (Stage 2). 모든 baseline은 동일 demonstration data로 fine-tuning되어 공정 비교.

## 5. 주요 결과

### 5.1 원본 LIBERO vs LIBERO-Occ (success rate %, comp-view unavailable, Table 2)

| Method | LIBERO Avg | LIBERO-Occ Avg | Drop |
|---|---|---|---|
| UniVLA | 88.25 | 57.10 | 31.15 |
| OpenVLA | 92.65 | 40.65 | 52.00 |
| OpenVLA-OFT | **95.75** | 47.95 | 47.80 |
| π-0 | 89.25 | 49.30 | 39.95 |
| π-0.5 | 90.00 | 40.55 | 49.45 |
| **VIM (Ours)** | 90.75 | **65.05** | **25.70** |
| VIM w/ GT comp. view | 93.00 | 74.00 | 19.00 |

VIM은 원본 LIBERO에서는 OpenVLA-OFT에 약간 뒤지지만 (90.75 vs 95.75), LIBERO-Occ에서는 **+7.95pp**로 SOTA를 달성하고 *drop*이 가장 작다 (25.70 vs 47.80–52.00). Ground-truth 보완 시점을 주면 74.00%까지 상승해, 학습된 visual prior에 의한 상상이 실제 추가 카메라에 근접한 유용한 단서를 만들어냄을 입증.

### 5.2 Hidden Dependence on Complementary Views (Fig. 3)
GT 보완 시점이 있고 없고의 성능 차이가 원본 LIBERO에서는 2.2–8.3pp에 불과하지만, LIBERO-Occ에서는 22.1–45.5pp로 확대된다 → 기존 VLA가 보이지 않을 때 *외부 시점에 강하게 의존*한다는 잠재적 약점을 드러냄.

### 5.3 Occlusion Type 분석 (Table 3)
모든 모델이 Receptacle(67.47–91.33) > Manipulated(28.89–54.67) > Dual(9.71–35.43) 순. 받침대 위치는 공간적 불확실성을 허용하지만, 가려진 물체를 정확히 잡는 grasping은 어렵고 Dual은 가장 어려움. VIM은 세 케이스 모두에서 최고.

### 5.4 Severity 분석 (Fig. 4)
Light → Heavy로 갈수록 baseline들은 급락 (Object/Goal suite에서 특히), 반면 VIM은 모든 severity에서 강건성 유지.

## 6. Ablation

### 6.1 Two-Stage Training (Table 4)
| Variant | Avg |
|---|---|
| w/o Stage-2 view loss | 0.00 (완전 실패) |
| w/o Stage-1 view training | 36.25 |
| Full | **65.00** |

- Stage-2 view loss 제거 시 출력 토큰이 valid visual-token grid를 형성하지 못하고 텍스트/special token과 섞여 *format collapse* → action 생성 불가. view loss가 단순 보조가 아닌 **구조적 정규화**임을 시사.
- Stage-1 제거 시도 큰 폭으로 하락 → cross-view spatial correspondence 사전학습이 필수.

### 6.2 Unified vs Separated Pipeline (Table 5)
"Ours → UniVLA"(VIM이 이미지 생성만 하고 UniVLA가 action 예측)는 62.00%, Full Unified는 65.00%. 동일 autoregressive 시퀀스 안에서 view+action을 함께 최적화하는 것이 더 유리.

## 7. 강점

1. **물리적으로 검증된 occlusion 생성**: BDDL parsing + 3D ray sampling + 3-step verification으로 단순 image masking이 아닌 실행 가능한 가림 시나리오 보장.
2. **Hidden dependence 노출**: GT 보완 시점 gap이 33% 이상 증가하는 현상을 정량화 — 기존 VLA의 잘 알려지지 않은 약점.
3. **Hardware-free 배포**: VIM은 추론 시 추가 카메라/active control 불필요 → 캘리브레이션·배치 비용 없이 robustness 확보.
4. **Two-stage가 필수적임을 ablation으로 명확히 증명** (Stage-2 제거 시 0%).
5. **2,000 task의 체계적 분포** (3 type × 3 severity × 4 suite).

## 8. 약점 / 한계

1. **시뮬레이션 한정**: LIBERO 기반이므로 실 센서 노이즈, 물체 다양성, dynamics 미포함. 실로봇 검증 부재.
2. **단일 보완 시점**: wrist/gripper만 평가. side, bird-eye, task-adaptive 등 다른 view의 일반화는 미검증.
3. **Paired complementary-view 데이터 의존**: Stage-1에서 GT 보완 시점 필요 → 실세계로 확장 시 데이터 수집 부담.
4. **Severe dual occlusion 시 imagination의 정확도가 본질적으로 학습된 visual prior에 종속**, 단서가 극히 적으면 hallucination 우려 (LIBERO-10에서 25.00%로 여전히 매우 낮음).
5. **불확실성 모델링 부재**: 생성된 view의 신뢰도를 추정하지 않으므로 잘못된 imagination을 그대로 action에 반영할 수 있음.
6. **원본 LIBERO에서는 SOTA가 아님** (90.75 vs OpenVLA-OFT 95.75): occlusion 특화 학습이 일반 manipulation 정확도에 약간의 trade-off.

## 9. 비교 컨텍스트

- **LIBERO-Plus / LIBERO-PRO** (Fei et al. 2025, Zhou et al. 2025): visual perturbation(배경·조명·노이즈) 중심 → 외형 변화. LIBERO-Occ는 *구조적 증거 결손*에 집중.
- **Multi-camera 정책** (VLA-LPAF, StereoVLA): 추가 카메라 사용. VIM은 가상의 보완 시점을 생성한다는 점에서 hardware-free.
- **Active perception** (ActiveVLA, SapaVe, ObserveThenAct): 카메라를 적극적으로 움직임. VIM은 카메라 고정.
- **World models / Generative VLA** (UniVLA, WorldVLA, CoT-VLA, RoboDreamer): VIM은 이 계열의 직접 후속으로, *generation = understanding* 통찰을 perception completion에 적용한 사례.

## 10. 재현성 / 실용 정보

- 코드 공개: https://github.com/litsh/Libero-Occ
- 8× H100, batch 192 — 학계 lab 수준 자원으로 재현 가능.
- UniVLA + Emu3-MoE + FAST tokenizer stack을 사용하므로 dependency가 무겁다.
- 평가는 main result는 500 rollouts/suite, ablation은 100 rollouts/suite로 비용 절감.

## 11. 주목할 인사이트

- **"Generation as understanding"**: 보완 시점을 *생성*할 수 있다는 것은 모델이 가려진 공간 구조를 *이해*했다는 신호로 사용 가능 — 단순 데이터 증강이 아닌 perception completion 메커니즘.
- **Receptacle vs Manipulated의 비대칭성**: 놓기는 공간 오차에 관대하고, 잡기는 정밀한 localization이 필요하다는 manipulation의 본질적 특성을 데이터로 확인.
- **Format collapse 발견**: view loss 제거 시 0%는 단순 성능 저하가 아니라 token sequence 구조 자체가 무너지는 현상으로, autoregressive multi-modal 학습의 *interface fragility*를 보여주는 흥미로운 관측.

## 12. 결론

LIBERO-Occ는 VLA robustness 연구에서 visual perturbation 일변도의 패러다임을 *partial observability*로 확장한 중요한 벤치마크이며, VIM은 추가 하드웨어 없이 generative world model의 능력을 perception completion으로 전환한 강력한 baseline이다. 시뮬레이션 한정·단일 보완 시점 등 한계가 있으나, "가린 것을 상상하라"는 단순하고 일반적인 원리를 통해 SOTA model들의 가려진 환경 성능 격차 25–47pp를 8pp 이내로 좁힌다는 점에서 추후 occlusion-aware VLA 연구의 출발점이 될 가능성이 크다.

<!-- VERIFIED: pdf -->
