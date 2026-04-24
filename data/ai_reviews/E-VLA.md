# E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

> **한 줄 요약**: DAVIS346 event camera stream을 SmolVLA 기반 VLA에 파라미터 프리 overlay 또는 13M hierarchical event adapter로 주입하여, 20 lux 저조도에서 Pick-Place 성공률 0% → 90%, 1000ms motion blur에서 0% → 25%까지 robustness를 회복한 event-augmented VLA.

---

## 1. 배경 및 동기

- 기존 VLA(RT-2, π family, OpenVLA, RDT-1B, Gemini Robotics)는 모두 frame-based RGB에 의존 — sensing-stage degradation(극저조도, motion blur, black clipping)에서 perception이 붕괴(Sec. 1).
- 저조도에서 brightness를 복구하려 exposure를 늘리면 motion blur + latency 증가, 심하면 black clipping으로 frame이 무의미.
- Image enhancement(RetinexNet, Retinexformer, EvLight) / event-to-image reconstruction(E2VID)은 이미 열화된 측정을 사후 처리할 뿐 capture-time 정보 손실을 회복 못 함.
- Event camera는 μs 단위 brightness change + 넓은 dynamic range로 이상적이지만, (i) sparse하고 RGB와 통계 분포 다름, (ii) manipulation에서 arm motion에 따라 event rate가 non-stationary.
- 저자들의 목표: "pretrained VLA를 그대로 두고, event를 plug-in으로 통합해 degradation robustness를 얻자."

---

## 2. 방법론

### Event Windowing: Recent-Count (Sec. 3.3)
Fixed-duration window(5/20/40ms)는 manipulation의 정지/가속 구간에서 불안정. 대신 **최근 N event 유지**(Eq. 2): W_I = {e_k}_{k=|E_t_e|-N+1}^{|E_t_e|}. 누적 frame은 polarity-agnostic counting(Eq. 3): Ẽ(x,y) = Σ δ(x-x_i, y-y_i), min-max normalize 후 Bayer demosaic으로 3채널 RGB 포맷.

### Fusion Strategy A: Parameter-Free Overlay (Sec. 3.4)
Event window 내 어떤 pixel에 event가 있으면 RGB image에 polarity-to-color c(p_j)로 덧씌움(Eq. 5): I^o(x,y) = I(x,y) if no events else c(p_j). **추가 파라미터 0**, ViT encoding 전에 입력 단계에서 fusion.

### Fusion Strategy B: Hierarchical Event Adapter (Sec. 3.4)
Event frame E를 16×16 **weight-sharing patch embedding**(RGB branch와 공유) + 4개 transformer block으로 처리. 각 block 출력을 SigLIP의 layer 3, 6, 9, 12에 `F^(l+1) = F_{l+1}(Fuse(F^(l), E^(l)))`로 fusion, Fuse = MLP(Concat(·))(Eq. 6). 추가 파라미터는 **13M으로 전체의 3% 미만**.

### Architecture & Training
- Backbone: SmolVLA (~0.5B, SigLIP + SmolVLM + action expert)
- Action head: interleaved self/cross-attention transformer, chunk size 40 at 30 Hz (~1.33s/forward)
- VLM 동결, action expert + projection MLP(+ event adapter)만 학습
- Progressive training: Action→Event→Joint (ablation에서 필수성 확인)
- Loss: imitation learning, RGB branch에 random dropout을 적용해 event utilization 강제
- 20k iter on single NVIDIA A800

### Dataset (Sec. 4)
SO100 robot + DAVIS346(wide-angle lens, 100° FoV). 3 tasks(Pick-Place, Sorting, Stacking), 4 조도(200/100/75/40 lux), **총 724 episodes / 339,310 frames**. RGB와 event는 μs timestamp로 동기화, LeRobot 포맷 저장.

---

## 3. 실험 결과

*Note: E-VLA YAML의 `benchmarks: {}`로 simulation benchmark 점수는 없고, 모든 실험은 self-collected real-world dataset.*

### Low-illumination Pick-Place (Table 1, %)

| Method | 75 lux | 40 lux | 35 lux | 30 lux | 25 lux | 20 lux | Avg |
|---|---|---|---|---|---|---|---|
| Image (10ms) | 100 | 80 | 70 | 35 | 0 | 0 | 47.5 |
| Image+RetinexNet | 100 | 100 | 85 | 80 | 25 | 10 | 66.7 |
| Image+Retinexformer | 100 | 80 | 80 | 75 | 20 | 10 | 60.8 |
| Image+EvLight | 100 | 95 | 95 | 75 | 45 | 10 | 70.0 |
| Image+E2VID | 80 | 60 | 55 | 10 | 5 | 5 | 35.8 |
| **Overlay (Ours)** | 100 | 100 | 85 | 75 | 65 | 60 | 80.8 |
| **Adapter (Ours)** | 100 | 100 | 95 | 90 | **90** | **90** | **94.2** |

초저조도(20 lux)에서 Image-only 0% → Overlay 60% → Adapter 90%. E2VID(event-to-image reconstruction)는 오히려 35.8%로 baseline보다 나쁨 — reconstruction이 정보를 손실한다는 증거.

### OOD Illumination Generalization (Table 2)
200 lux만으로 학습, 저조도에서 평가. Adapter는 75 lux 85%, 40 lux 75%, 20 lux 45%로 baseline(0-39% avg) 대비 안정. OOD 평균 **75.0% vs image-only 39.0%**.

### Motion Blur (Table 3, Pick-Place/Sorting)

| Exposure | Image P&P/Sort | Overlay | Adapter |
|---|---|---|---|
| 100ms | 100/77.5 | 100/95.0 | 100/95.0 |
| 500ms | 15/20 | 60/55 | 50/45 |
| 1000ms | 0/5 | 20/32.5 | 25/32.5 |

극심한 blur(1000ms)에서도 adapter가 25/32.5를 유지.

### Ablations (Sec. 5.6)
- **Event window size** (Table 4, 30 lux): 2000 events가 최적(94.2% avg) — 500(67.5), 4000(85.8). 시간 기반 window(5/20/40ms)는 20ms가 최선이나 82.5로 count-based에 미달.
- **Training recipe** (Table 5, 30 lux): Action→Event→Joint + weight-sharing patch embed → 90%. Action→Joint 75%, Event→Joint 50%, 공유 없이 80%. **Progressive + weight sharing 모두 필수**.

---

## 4. 한계 및 미해결 문제

1. **Color sorting에서 실패 (저자들이 Sec. 5.7에서 명시)**: Event는 intensity change를 인코딩하므로 grasp 이후 static color가 사라짐. RGB event는 Bayer demosaicing의 밀도 가정을 충족하지 못해 color discrimination이 약함. Sorting task의 placement 단계에서 주 실패 원인.
2. **Stacking의 occlusion (저자들이 Sec. 5.7에서 명시)**: Wrist-mounted camera가 잡은 물체에 가려지는 viewpoint occlusion. Event 정보가 있어도 perceptual bottleneck은 해결 안 됨.
3. **Monocular wrist-view 한정**: Multi-view나 table-top camera 없음. 넓은 작업 공간으로의 확장 미검증.
4. **Single SmolVLA backbone**: 다른 VLA(π0, OpenVLA 등)로의 generality는 평가되지 않음.
5. **Real dataset 규모 724 episodes**: 일반화를 위해서는 확장 필요. Simulation benchmark 평가 없음(YAML benchmarks 비어있는 이유).

---

## 5. 총평

| 항목 | 평가 |
|------|------|
| **Novelty** | ★★★☆☆ — Event camera를 VLA에 통합한 최초 시도라는 응용적 가치가 크지만, overlay/adapter 자체는 기존 sensor fusion 아이디어의 adaptation |
| **Practical impact** | ★★★★☆ — 실제 배포 시 가장 흔한 failure mode(조명/모션블러)를 정면으로 해결, +13M params로 pretrained 자산 그대로 활용 가능 |

E-VLA의 실용적 가치는 "VLA가 실험실 조명을 떠날 때 가장 먼저 깨지는 게 frame-based perception"이라는 현실을 정면으로 다룬 데 있다. Overlay 방식은 거의 한 줄짜리 변경인데도 20 lux에서 0% → 60%로 끌어올리며, adapter는 추가 13M param으로 90%까지. E2VID가 오히려 성능을 떨어뜨린 점은 "event를 image로 변환하지 말고 그대로 주입하라"는 교훈을 준다. Failure analysis(color/occlusion)도 정직하고, recent-count windowing은 manipulation-specific 기여로 재사용 가치가 있다.

---

## 6. 예상 질문

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | E2VID(event-to-image reconstruction)가 왜 오히려 성능을 떨어뜨렸나? | Reconstruction은 (i) event → image 변환 시 고유 temporal 정보 손실, (ii) reconstructed image가 RGB pretraining 분포와 다름, (iii) 저조도 event sparsity로 reconstruction quality 자체가 나쁨. Overlay는 변환 없이 raw event signal을 주입해 우위. |
| 2 | 왜 parameter-free overlay가 이미 그렇게 강한가? Adapter는 왜 필요한가? | DAVIS의 pixel-aligned event가 RGB와 공간적으로 정확히 겹치므로 overlay만으로 edge/motion cue를 시각 pipeline에 직접 주입. 그러나 극심한 degradation에서는 overlay만으로 부족 — adapter는 intermediate SigLIP layer에서 fine-grained cross-modal 상호작용을 제공해 20 lux에서 60 → 90%로 추가 이득. |
| 3 | Recent-count windowing이 fixed-duration 대비 왜 robust한가? | Wrist-camera ego-motion에 따라 event rate가 task phase(정지/transport/contact)마다 급변. Fixed-duration은 정지 시 event 부족, 빠른 움직임 시 과포화로 "motion blur" in event domain을 유발. 최근 N event로 제한하면 항상 비슷한 density가 확보되어 perception-action coupling이 안정화. Table 4에서 2000-count가 optimal. |

<!-- VERIFIED: pdf -->
