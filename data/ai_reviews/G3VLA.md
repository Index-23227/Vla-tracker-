## G3VLA: Geometric inductive bias for Vision-Language-Action Models

> **한 줄 요약**: 사전학습된 VLA의 비주얼 토큰 스트림에 (1) intrinsic-conditioned ray embedding, (2) PRoPE projective positional encoding, (3) bidirectional cross-view fusion을 삽입하고 π³X dense point-map을 confidence-gated distillation으로 지도학습하여, π0 백본에서 LIBERO 평균을 84.6→88.1(+3.5%p)로 끌어올린 camera-aware geometric module. 백본·action space·imitation objective는 그대로 유지.

---

### 1. 배경 및 동기

- RT-2, OpenVLA, π0, GR00T 같은 VLA는 의미(semantics)는 잘 다루지만 **시각 토큰이 2D 이미지 좌표에 머물러 있어** 카메라 캘리브레이션(intrinsics/extrinsics)이 행위 감독에서 암시적으로만 학습된다.
- PerAct/RVT/Act3D처럼 명시적 3D 표현을 쓰는 방법은 정밀하지만 **VLM semantics 재활용이 어렵다**. SpatialVLA/3D-VLA 같은 가교들은 추가 depth 센서 입력이나 action space 변경을 요구.
- 핵심 질문: **"백본·행위 공간·imitation objective를 건드리지 않고 캘리브레이션된 카메라 기하만 visual-token 통로로 주입하면 generalist VLA의 공간 정밀도가 좋아질까?"** → 답: 예.

---

### 2. 방법론 심층 분석

#### 2.1 Camera-Aware Geometric Module Fψ
사전학습된 ViT가 뽑은 patch token z_p^v 위에 세 모듈을 끼워넣어 h = Fψ(z, {K^v, T^v})를 만든다.

#### 2.2 Intrinsic-conditioned Ray Embedding
픽셀 u=(x,y,1)^T에 대해 정규화 ray r̃^v(u) = (K^v)^-1 u의 첫 두 성분 R^v(x,y)를 학습 임베딩 G_φ로 patch grid에 투영, encoder output에 더함:
  z₀,p^v = z_p^v + G_φ(R^v)_p
→ 같은 픽셀이라도 intrinsics가 다르면 다른 viewing direction이라는 사실을 토큰에 직접 새김.

#### 2.3 PRoPE (Projective Positional Encoding)
[Cameras as Relative PE, NeurIPS 2025]에서 가져온 attention bias. K^v, T^v, patch 위치로부터 query/key/value의 fixed projective transform을 유도해 **외형 유사도가 아닌 카메라 모델 기반 cross-view projective 관계**로 attention을 안내.

#### 2.4 Bidirectional Cross-View Fusion
- **Frame Attention**: view별로 토큰 self-attention (view-local 구조 보존).
- **Cross-View Attention**: view×patch flatten 후 모든 valid token이 양방향 attend, positional signal로 PRoPE 사용.
- 출력 H가 사전학습 VLA가 기대하는 동일 token interface로 action 모듈에 전달.

#### 2.5 Geometry Distillation
Fused token에 auxiliary point head를 붙여 per-pixel ray 좌표 q̂_u^v ∈ R² + log-z depth d̂_u^v를 예측. Target은 (a) 시뮬레이션에선 GT depth, (b) 실세계에선 **π³X teacher의 dense point map**, confidence c_u^v를 σ(c) > τ(τ=0.1)로 hard gate:
  L_distill = Σ m·(½||q̂-q||² + (d̂-d)²) / (Σm + ε)
추론 시 point head 폐기.

#### 2.6 Two-Stage Training
- **Stage 1**: backbone frozen, geometric module + point head만 학습, distillation 지배.
- **Stage 2**: 전체 finetune, action loss 지배, distillation은 regularizer.
- 통합 loss: L = λ_act L_act + λ_distill L_distill (π0 인스턴스화에서 L_act는 원래 flow matching).

---

### 3. 실험 결과 심층 분석

#### 3.1 LIBERO on π0 (Table 1, %)

| Suite | π0 baseline | G³VLA (π³X) | G³VLA (GT) | Gain (GT) |
|-------|-------------|-------------|------------|-----------|
| Goal | 87.4 | 88.4 | 88.4 | +1.0 |
| Spatial | 85.2 | 88.6 | 89.2 | +4.0 |
| Object | 89.4 | 93.4 | **94.4** | **+5.0** |
| L-10 | 76.5 | 77.6 | 80.4 | +3.9 |
| **Average** | **84.6** | **87.0** | **88.1** | **+3.5** |

- Object와 Spatial에서 가장 큰 폭(+5.0, +4.0). 저자의 동기(객체 위치/공간 관계 추론 task가 캘리브레이션 구조의 수혜)와 정합.
- π³X teacher 만으로도 +2.4%p → depth 센서 없이도 실용적.

#### 3.2 RoboCasa24 & RoboTwin2.0 on π0 (Table 2)
- RoboCasa24 평균: 34.2 → 36.5 (π³X) / 37.1 (GT).
- RoboTwin2.0 handover-block: 44.0 → 41.0 (π³X) / **49.0 (GT)**. 합성 깨끗한 장면에선 π³X teacher가 오히려 noise를 주입하는 failure case를 솔직히 보고.

#### 3.3 π0.5 (Table 3) & GR00T 1.5 (Table 4)
- π0.5: 95.9 → 97.0 (+1.1%p). 이미 saturation 근처라 작지만 호환성 확인.
- GR00T 1.5: 94.90 → 95.25 (π³X) / 94.50 (GT, **개선 없음**). two-tower 구조에서 diffusion 정책이 frozen VLM에 cross-attention으로만 접근 → "geometry-aware token이 action 경로에 직접 닿아야 효과가 크다"는 **흥미로운 architectural caveat**.

#### 3.4 Real-World UR5 bimanual (Table 5)
- Pouring Nut(GT supervision): π0 OOD 70.8–75.0 → **83.3–87.5**, overall 82.5–85.0 → 90.0–92.5.
- Test Tube: 후기 checkpoint에서 π0.5 OOD 25→50(25K), 41.7→58.3(30K).
- OOD 카메라 viewpoint(11/12/13)에서 효과가 두드러져, "view shift generalization"에 캘리브레이션 inductive bias가 유효함을 검증.

---

### 4. Ablation 분석 (Figure 3, LIBERO π0)

| Variant | Avg | Δ |
|---------|-----|---|
| Baseline | 84.6 | — |
| w/o Ray | 85.0 | −2.0 vs G³VLA(π³X) |
| w/o PRoPE | 85.9 | −1.1 |
| 1-Stage (no curriculum) | 86.3 | −0.7 |
| G³VLA (π³X) | 87.0 | — |
| G³VLA (GT) | **88.1** | +3.5 vs baseline |

- Ray embedding 제거가 가장 큰 손실 → **per-patch viewing direction이 핵심 신호**.
- PRoPE는 cross-view 관계 보강(상보적).
- Two-stage curriculum이 명확히 기여(+0.7%p).

---

### 5. 한계 및 미해결 문제

1. **캘리브레이션 의존**: intrinsics/extrinsics가 정확해야 하며 calibration drift, sync error, train-test mismatch에 민감.
2. **Teacher bias**: π³X가 occlusion/specularity/blur/희박한 viewpoint에서 약함. Confidence gating은 완화일 뿐.
3. **Architecture-dependent**: GR00T 1.5 two-tower처럼 geometry-aware token이 action 경로와 cross-attention 한 단 떨어진 경우 효과 감쇠.
4. **추가 오프라인 비용**: π³X teacher cache + auxiliary point head 학습 비용. 배포 시점엔 불필요하나 학습 파이프라인 부담.
5. **Action space는 그대로**: imitation objective의 근본적 한계(demonstration scarcity, language-action grounding)는 해결하지 못함.

---

### 6. 총평

| 항목 | 평가 |
|------|------|
| Novelty | ★★★★ — Ray + PRoPE + cross-view fusion + π³X distillation을 VLA token 통로에 통합한 점이 깔끔 |
| Technical depth | ★★★★ — 3개 백본(π0/π0.5/GR00T1.5) × GT/π³X 두 감독 × 3축 ablation |
| Experimental rigor | ★★★★ — LIBERO+RoboCasa24+RoboTwin2.0+실세계, GR00T failure case 정직 보고 |
| Practical impact | ★★★★ — depth 센서 없이 동작, 백본 unchanged → 기존 VLA에 plug-in 가능 |
| Writing quality | ★★★★ |

**강점**: action space/objective 보존이라는 제약을 지키면서 LIBERO +3.5%p, OOD viewpoint 실세계 +12%p 이상. GR00T 1.5 failure를 통해 **"geometry-aware token이 action 경로와 얼마나 직접 닿느냐"**라는 유용한 일반 원리 시사.
**약점**: calibration accuracy 가정이 강하고, teacher noise가 합성 장면에서 역효과를 내는 사례가 존재.

---

### 7. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 요점 |
|---|------|---------------|
| 1 | PRoPE 단독 효과는? | w/o Ray 85.0, w/o PRoPE 85.9 → Ray가 더 큰 기여이지만 둘은 상보적(논문 명시) |
| 2 | π³X teacher가 노이즈인 RoboTwin2.0 경우는 어떻게? | GT depth로 49.0 회복. 실세계에선 confidence gating(τ=0.1)으로 완화하되 도메인 mismatch 시 약함 |
| 3 | GR00T 1.5에서 안 되는 이유는? | two-tower 구조 — diffusion policy가 frozen VLM에 cross-attention으로만 접근, geometry-aware token이 action 모듈에 직접 들어가지 못함 |
| 4 | SpatialVLA/3D-VLA 대비 장점? | depth 센서·action space 변경 불필요, pretrained VLA에 plug-in. 단 명시적 3D 표현이 주는 강한 spatial precision은 trade-off |
| 5 | Inference cost? | point head는 추론 시 폐기. ray embedding + PRoPE + cross-view fusion만 남음 → 추가 비용 제한적 |
| 6 | Calibration drift에 robust한가? | 명시적 한계로 언급. online calibration robustness는 future work |

---

### 8. 핵심 수식 요약
- Ray map: R^v(x,y) = [(K^v)^-1 u]_{1:2}, u=(x,y,1)^T
- Ray-augmented token: z₀,p^v = z_p^v + G_φ(R^v)_p
- Fusion: H = Fusion_ψ(Z; {K^v, T^v})
- Distillation gate: m_u^v = 1[σ(c_u^v) > 0.1]
- 통합 loss: L = λ_act L_act(flow matching) + λ_distill L_distill

---

### 9. 데이터/벤치마크 메모
- **LIBERO**: 4 suites(Goal/Spatial/Object/L-10), G³VLA(GT) 평균 **88.1%**.
- **RoboCasa24**: 가정 환경 24 task family, 37.1%.
- **RoboTwin2.0 handover-block**: bimanual 진단 task, 49.0%.
- **Real UR5 bimanual**: Pick-and-Place Test Tube + Pouring Nut, 120 episodes×task, 10×10cm grid, view 1/3 ID, 11/12/13 OOD.

---

### 10. 재현/엔지니어링 노트
- Backbone은 π0/π0.5/GR00T 1.5 그대로, action 디코더 미변경 → 기존 체크포인트에서 fine-tune.
- π³X teacher는 오프라인 캐싱 가능, deployment 시 호출 없음.
- Two-stage curriculum의 stage 1 hyperparameter는 Appendix D 참조(본문엔 미정량).
- 추론 입력: RGB views + proprio + language + camera calibration(K, T). depth 불필요.

---

### 11. 후속 연구 방향
1. **Online calibration estimation**: extrinsics drift에 robust한 self-calibration loop.
2. **Tighter geometry-action coupling**: GR00T 1.5 failure를 토대로 geometry-aware token이 diffusion/flow expert에 직접 들어가는 architecture 탐색.
3. **3D action representation**: 현 method는 토큰 수준만. 행동 공간 자체를 3D-aware로 확장.
4. **Cross-embodiment 일반화**: UR5 외 다양한 실세계 platform에서 검증.
5. **Teacher 다양화**: π³X 외 VGGT/DUSt3R 앙상블, occlusion-robust target.

---

### 12. 한 줄 결론
**"백본을 건드리지 않고 카메라 캘리브레이션을 token-level inductive bias로 주입하는 것만으로 pretrained VLA의 spatial generalization이 의미 있게 개선된다"**는 것을 LIBERO 88.1%(+3.5%p), RoboCasa 37.1%, real OOD +12%p로 입증한 lightweight plug-in module.

<!-- VERIFIED: pdf -->
