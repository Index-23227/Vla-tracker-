# MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation

> **한 줄 요약**: PaliGemma-2 기반 VLA가 task instruction + RGB만 보고 **sensor token (`<None>/<Thermal>/<Acoustic>/<mmWave>`) + target description**을 "tool call" 형태로 먼저 출력 → SAM3 마스킹 위에 비전 모달리티의 heatmap을 덮어쓴 **grounded sensor image**를 동일한 SigLIP 인코더로 재입력 → DiT action expert로 dexterous hand 액션 생성. Real-world dexterous manipulation에서 평균 **80.6% (사전학습 포함)**, **unseen zero-shot 66.7%** 달성. RGB-only π₀ baseline (20.8%) 대비 **+59.8%p**.

---

## 1. 배경 및 동기 (§1, §2)

- 기존 VLA(OpenVLA, π₀, π₀.₅, VITRA 등)는 RGB만 입력으로 사용 → 온도·소리·레이더 반사처럼 **RGB로 추론 불가능한 물리적 속성**을 다루지 못함 (§1).
- 선행 multisensory VLA(PointVLA, Tactile-VLA, MLA, OmniVLA 등)는 세 가지 한계: (i) 모달리티별 **specialized encoder**가 필요해 새 센서 추가에 비용이 큼, (ii) 대규모 sensor-paired robot dataset에 의존, (iii) **고정 센서 set + static fusion** — task에 따라 켜고 끄는 적응적 라우팅 부재 (§2).
- 본 논문의 핵심 가설: "**센서는 항상 켜져 있는 입력이 아니라, 인간처럼 task-conditioned로 호출되는 on-demand 'tool'이다.**" 이 관점 위에 (a) sensor token, (b) grounded sensor image, (c) RGB-only 데이터로부터의 sensor data synthesis라는 세 가지 디자인을 세움.

---

## 2. 방법론 — Adaptive Multimodal Sensing (§3, Fig. 2)

### 2.1 Task formulation (Eq. 1-2)
- 표준 multi-sensor 정책 π : (l, o_t, s_{1,t}, …, s_{N,t}) → A. 대신 본 논문은 이를 **세 단계로 분해**:
  1. π : (l, o_t) → (l_s, l_d)  — sensor token + target description 생성 (Eq. 2 좌)
  2. G : (o_t, s_{i,t}, l_d) → m_{i,t}  — grounded sensor image 구성 (Eq. 2 중)
  3. π : (l, o_t, l_s, l_d, m_{i,t}) → A  — 액션 청크 생성 (Eq. 2 우)
- **같은 VLA π를 두 번 호출**하는 통일 구조 — sensor selection과 action generation을 모두 한 모델이 담당.

### 2.2 Sensor tokens (§3, §4.1)
- `<None>` (sensing 불필요 시 RGB-only fallback) / `<Thermal>` / `<Acoustic>` / `<mmWave>` — VLM은 이 special token을 일반 언어 토큰과 동일하게 cross-entropy로 학습.
- 토큰 디자인이 enum 형태이기 때문에 **새 센서 추가 = vocabulary 확장**으로 환원 — backbone retraining 불필요.

### 2.3 Grounded sensor image (Eq. 3, Fig. 3)
- VLM이 출력한 target description l_d를 SAM3에 넣어 binary mask M = f_seg(o_RGB, l_d)를 얻고, **m = M ⊙ s + (1-M) ⊙ o_RGB** 로 구성.
- 즉 **타겟 물체 영역만 sensor heatmap으로 덮어쓰고 나머지는 RGB 그대로** 유지 → 동일한 SigLIP 인코더가 RGB와 sensor image를 모두 처리 가능 (modality-specific encoder 불필요).
- SAM3를 VLA 외부에 둠으로써 **mask 갱신을 비동기 처리** → action generation 지연 최소화.

### 2.4 모델 백본 (§4.1)
- **VLM**: PaliGemma-2 (Gemma-2 LLM + SigLIP vision + linear projection). VITRA 가중치로 초기화.
- **Action expert**: Diffusion Transformer (DiT). Visual-language feature + robot state + noisy action → denoised action sequence 예측.
- VITRA의 learnable **cognition token**을 그대로 가져와 추가 prefix로 사용.

---

## 3. 데이터 합성 파이프라인 (§4.2, Fig. 4)

- 핵심 통찰: grounded sensor image는 **target object region에 색상 코드를 입힌 RGB**일 뿐이므로 **실제 센서 없이도 합성 가능**.
- 절차: (1) Sensor dictionary 정의 (e.g. thermal → {hot, cold, warm}, acoustic → {ringing, quiet}, radar → {occupied, empty}). (2) RGB 비디오마다 모달리티·키워드 무작위 샘플. (3) **GPT-5.2**로 task instruction에 키워드 주입 (예: "Put the mug…" → "Put the **hot** mug…") 및 target description 생성. (4) SAM3로 segmentation 후 키워드에 매핑된 색을 덮어씌움. (5) 원본 RGB와 색상 마스크 비디오를 concat → synthesized data.
- **데이터 증강 트릭**: 마스크된 region을 다른 위치로 **clone하되 잘못된 sensor color**를 덮어 만든 distractor를 함께 입력 → 모델이 spatial prior가 아니라 sensor 정보 자체에 의존하도록 강제.
- 소스: **MolmoAct + AgiBotWorld-Alpha + VITRA**. 합성 규모: **9.6K episodes / 1.05M frames / >1000 objects**.

---

## 4. 학습 (§4.3)

- 두 단계 손실의 합:
  - L_VLM = L_sensor + L_target (cross-entropy, Eq. 4)
  - L = L_VLM + λ · E[‖ε - ε_θ(a_τ, τ, c)‖²], **λ=1e-2** (Eq. 5)
- 저자 관찰: **VLM loss 없이 VLA만 학습 시 VLM 출력이 degrade** — 공동 학습이 필수.
- 학습: 64× A100 40GB, batch 512, 20K step (~20h), lr 1e-5, AdamW.
- VITRA 가중치 초기화로 large-scale human-hand pretraining 활용.

---

## 5. 실험 셋업 (§5.1, Fig. 5)

- **로봇**: 12-DoF Robotera XHand dexterous hand × 1, 마지막 단의 sensor module 부착.
- **센서 suite**: Intel RealSense RGB-D, infiRay T2S thermal × 2, Calterah 4T4R 60GHz mmWave radar, Sipeed 6+1 mic array. 모두 RGB FOV에 spatial alignment된 2D heatmap을 제공.
- **데이터 수집**: MANUS gloves teleoperation으로 손/팔/센서 동시 기록 → 720개 demonstration, 10개 sub-task instruction, 3개 sensing modality.
- **세 가지 task family**:
  1. Thermal-guided pick-and-place: "Pick up the hot drink and place it into the basket." (hot/cold/room-temp × 2 drink type).
  2. Acoustic-grounded search: "Pick up the clothes/towels covering the ringing phone…" (덮개에 가린 음원 위치 탐지).
  3. mmWave radar-guided search: "Open the occupied box." (닫힌 박스 내부 가시화).
- 평가 metric: (i) task success rate (24 trials/cell), (ii) task score = 0.5·sensing + 0.5·manipulation. Fisher exact two-sided p-value 보고.

---

## 6. 주요 결과 — Multisensory Manipulation (§5.2, Tab. 1)

| Method | Thermal | Acoustic | mmWave | Avg | Sensing | Manip. | Score |
|---|---|---|---|---|---|---|---|
| π₀-RGB | 33.3 | 25.0 | 4.17 | 20.8 | 48.6 | 43.1 | 0.458 |
| π₀.₅-RGB | 16.7 | 33.3 | 8.33 | 19.4 | 41.7 | 33.3 | 0.375 |
| π₀-Raw | 16.7 | 41.7 | 25.0 | 27.8 | 86.1 | 27.8 | 0.569 |
| π₀.₅-Raw | 16.7 | 33.3 | 20.8 | 23.6 | 83.3 | 29.2 | 0.563 |
| MuseVLA-RGB | 12.5 | 33.3 | 20.8 | 22.2 | 41.7 | 43.1 | 0.424 |
| MuseVLA-Raw | 41.7 | 25.0 | 33.3 | 33.3 | 91.7 | 33.3 | 0.625 |
| MuseVLA-RawAdapt | 70.8 | 41.7 | 66.7 | 59.7 | 93.1 | 59.7 | 0.764 |
| **MuseVLA** | **83.3** | **58.3** | **87.5** | **76.4** | **95.8** | **77.8** | **0.868** |

- π₀-RGB 대비 **+55.6%p** (76.4 vs 20.8), π₀-Raw 대비 **+48.6%p**. Acoustic이 상대적으로 낮은 이유는 **부드러운 옷·수건을 잡는 manipulation 자체의 난이도** (사운드 localization은 정확).
- mmWave가 가장 큰 격차 (4.17% → 87.5%) — RGB로는 닫힌 박스 내부 추론 불가, 본 모델의 동기를 가장 잘 드러냄.

---

## 7. Ablation — 무엇이 효과적인가? (§5.2, Tab. 1)

- **MuseVLA-RawAdapt** (grounded sensor image 제거, raw heatmap + sensor token 유지): 59.7% — 풀 모델 대비 -16.7%p. → **grounded sensor image의 unification이 노이즈가 큰 raw heatmap을 vision encoder가 다루기 어려움을 해소**.
- **MuseVLA-Raw** (adaptive selection까지 제거, 모든 센서 항상 concat): 33.3% — -43.1%p. → 모든 센서 input은 **불필요한 input noise + 더 큰 메모리 부담**.
- 흥미로운 관찰: MuseVLA-Raw는 sensing accuracy 91.7%로 높으나 manipulation 33.3%로 낮음 — **raw heatmap이 sensing 단계는 보조하지만 manipulation 단계엔 distractor**.
- π₀-Raw 역시 sensing 86.1% / manipulation 27.8%로 동일한 패턴 → grounded 표현의 가치가 manipulation 단계에서 가장 크게 드러남.

---

## 8. Adaptive Sensor Selection 성능 (§5.3, Tab. 2)

| | Training Sensor | Training Target | Unseen Sensor | Unseen Target |
|---|---|---|---|---|
| PaliGemma-2 | 0% | 13.0% | 0% | 9.5% |
| MuseVLA w/o pretrain | 100% | 100% | 85% | 40.5% |
| **MuseVLA (pretrained)** | 100% | 93.5% | **100%** | **82.0%** |

- 200 sample/세팅으로 평가.
- Synthesized data pretrain은 **unseen task의 target description 정확도를 40.5% → 82.0%**로 두 배 끌어올림 — zero-shot 일반화의 핵심.
- **Inference GPU memory**: MuseVLA-Raw 대비 13.23 GB → 6.61 GB (모델 가중치 11.9GB 제외, 100 sample 평균). 새 센서를 추가해도 메모리가 더 늘지 않음 — "필요할 때 한 모달리티만" 처리하기 때문.

---

## 9. Pretraining의 효과 + Zero-shot 일반화 (§5.4, Tab. 3)

| | Thermal | Acoustic | mmWave | Seen Avg | Thermal | Acoustic | mmWave | Unseen Avg |
|---|---|---|---|---|---|---|---|---|
| MuseVLA-Raw | 41.7 | 25.0 | 33.3 | 33.3 | 31.3 | 25.0 | 18.8 | 25.0 |
| MuseVLA w/o pretrain | 83.3 | 58.3 | 87.5 | 76.4 | 25.0 | 31.3 | 25.0 | 27.1 |
| **MuseVLA (pretrained)** | **87.5** | **70.8** | **83.3** | **80.6** | **75.0** | **56.3** | **68.8** | **66.7** |

- Seen task에서는 pretrain이 +4.2%p 정도의 marginal 개선.
- **Unseen task에서는 +39.6%p (27.1 → 66.7)**, 전 모달리티 일관 개선. 합성 dataset 없으면 일반화는 거의 우연 수준 (27.1%).
- Seen acoustic이 +12.5%p로 크게 좋아진 것은 **다양한 합성 manipulation episode가 soft fabric 조작 정책을 강화**했기 때문.

---

## 10. Multi-stage Multi-sensor 태스크 (§5.2 말미)

- **Radar → RGB**: 닫힌 박스 중 점유된 것을 mmWave로 식별 → 열고 → RGB-only pick-and-place. **12 trial 평균 66.7%** (localization 100% / opening 83.3% / pick-and-place 66.7%).
- **Radar → Thermal**: radar로 박스 열기 → 안의 여러 음료 중 hot/cold/room-temp 한 잔을 thermal로 골라 잡기. **75.0%** (localization 100% / thermal 선택 100% / opening 100% / pick-and-place 75.0%).
- → 단일 모델이 **stage마다 sensor token을 다르게 출력**하며 long-horizon에서 dynamic dispatch 가능함을 확인.

---

## 11. 비교/포지셔닝

| | Specialized encoder | Fixed sensor set | Adaptive selection | Data synthesis |
|---|---|---|---|---|
| PointVLA / 3D-CAVLA | 필요 (depth) | 있음 | × | × |
| Tactile-VLA / VLA-Touch | 필요 (tactile MLP/transformer) | 있음 | × | × |
| MLA / OmniVLA | 일부 공유 | 대체로 고정 | △ (제한) | × |
| **MuseVLA** | **불필요 (SigLIP 공유)** | **확장 가능** | **○ (sensor token)** | **○ (RGB→합성)** |

- 본 논문의 핵심 차별점: **(a) modality-specific encoder를 제거**, **(b) sensor를 enum tool token으로 환원**, **(c) RGB 데이터셋만으로 multisensory training 데이터를 합성**. 세 요소가 결합되어 새 센서 추가가 거의 zero-cost가 됨.

---

## 12. 한계 / 미해결 문제 / 향후 방향 (§6)

1. **실세계 데이터 규모 한계**: 720 episode만으로는 더 다양한 환경·조명·물체 일반화 미검증.
2. **Segmentation 모듈 의존**: SAM3가 target description을 잘못 마스킹하면 grounded sensor image 전체가 오염. 저자도 명시.
3. **2D-aligned heatmap 가정**: 각 센서가 RGB FOV에 spatial alignment된 heatmap을 제공한다고 가정 — 1D microphone 같은 비공간 센서나, 3D point-cloud 형태 lidar로 일반화하려면 G 함수 재설계 필요.
4. **Sensor dictionary 휴리스틱**: thermal→hot/cold 같은 매핑이 hand-crafted. 더 fine-grained continuum (구체적 온도값)이나 학습된 token vocabulary로 확장 여지.
5. **Action head는 unimodal DiT**: action 자체는 표준 diffusion expert. action chunk horizon, smoothing, recovery 등 dexterous control 측면의 별도 기여는 없음.
6. **Open-source 미공개 (논문 작성 시점)**: §1 말미에서 공개 의사를 밝히지만 reproducibility는 추후 확인 필요.
7. **VLA backbone 단일 (PaliGemma-2 + VITRA init)**: π₀.₅, GR00T 등 다른 backbone과의 호환성/상한은 미검증.
8. **Multi-stage 12 trial / task family당 24 trial**: 통계적 표본 수가 제한적 — Fisher exact p-value 보고는 하지만 confidence interval 협소.
9. **인간 의도 모호성**: "hot drink" 같은 description이 절대 온도 미지정 → 환경 분포에 의존. Open-world 시나리오에서 prompt sensitivity 미평가.

---

## 결론
MuseVLA는 **"센서 = on-demand tool"** 패러다임을 (a) sensor token, (b) grounded sensor image, (c) RGB-only 합성 데이터 세 요소로 일관성 있게 구체화한 작업이다. 동일한 SigLIP 인코더로 RGB와 sensor를 모두 다룬다는 단일 선택이 architecture 단순성과 데이터 합성 가능성을 동시에 잠금 해제했고, 결과로 dexterous hand + multi-sensor 도메인에서 **80.6% seen / 66.7% zero-shot**이라는 명확한 마진을 얻었다. RGB-only VLA의 한계가 명확히 노출되는 thermal/acoustic/mmWave 환경에서 "표준 VLA 위에 어떻게 더 일반적으로 sensor를 얹을지"의 강한 baseline을 제시한 paper.

<!-- VERIFIED: pdf -->
