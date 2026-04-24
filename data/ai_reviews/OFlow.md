# OFlow: Injecting Object-Aware Temporal Flow Matching for Robust Robotic Manipulation

## 1. 배경 및 동기

기존 VLA 모델은 대부분 현재 관측 프레임에 반응적으로(action-as-reaction) 행동을 생성하기 때문에, 움직이는 객체나 상호작용이 수반되는 장면에서 취약하다는 한계가 있다(Sec. 1). OFlow는 (i) 자동으로 task-relevant 객체를 식별하는 object-aware perception, (ii) 장면의 시간적 진화를 예측하는 foresight — 두 능력을 하나의 semantic latent space 안에서 통합한다. 픽셀을 예측하는 기존 CoT-VLA/DreamVLA/F1/VPP 계열과 달리, DINOv2 semantic latent 위에서 직접 미래를 예측하여 VAE 디코딩–검출기 재인코딩 같은 중복 처리 파이프라인을 제거한 것이 핵심 출발점이다(Sec. 2).

## 2. 방법론

저자는 GR00T-N1.5(Eagle-2.5 VLM + Qwen2.5-VL)를 VLA 백본으로 유지한 채, 외부 foresight 모듈을 주입한다(Sec. 3.1–3.3, Fig. 2).

- **Causally-Constrained Autoregressive Flow Matching (Sec. 3.2, Fig. 3)**: DINOv2-Base(with registers)로 프레임별 latent $z_i \in \mathbb{R}^{L \times d}$를 추출하고, 12-layer DiT에 "시간축은 causal / 프레임 내 공간은 dense" 방식의 attention mask $M = C \otimes 1_{L \times L}$를 적용해 병렬 훈련 가능한 autoregressive flow matching을 구성한다(Eq. 2–7).
- **Object-Aware Scene Factorization (Sec. 3.2, Fig. 4)**: 학습 없이 K-Means로 patch feature를 군집화해 $K$개의 prototype token으로 요약하고, $K \in \{2,4,6,8,12\}$의 계층적 multi-scale 집합 $\mathcal{S} = C_{K_1} \cup \dots \cup C_{K_N}$을 구성한다(Eq. 8).
- **ControlNet-Style Injection (Sec. 3.3, Eq. 10)**: 행동 expert의 cross-attention에 $\mathcal{S}$를 key/value로 결합하되, 선형 projection을 zero-init해 사전학습된 VLA 백본의 표현을 보존한다. Stage 1은 foresight만 훈련, Stage 2는 VLM 고정 후 VLA를 파인튜닝한다(8×A100).

## 3. 실험 결과

**Table 1 (LIBERO / LIBERO-Plus)**: OFlow는 LIBERO 네 suite 평균 96.6%(Spatial 98.0 / Object 99.0 / Goal 95.0 / Long 94.5)로 GR00T-N1.5(94.4), π0(94.2), InternVLA-M1(95.9)을 모두 상회. LIBERO-Plus에서는 72.3%로 GR00T-N1.5(69.5) 대비 +2.8%p.

**Table 2 (SimplerEnv WidowX)**: Spoon 76.8 / Carrot 62.7 / Eggplant 56.6 / Stack 72.4, 평균 67.1로 GR00T-N1.5(59.8), π0-FAST(48.3) 대비 큰 폭 우위.

**Table 3 (MetaWorld MT50)**: 85.6% 평균(Easy 93.6 / Middle 87.3 / Hard 63.6), GR00T-N1.5 76.8% 대비 +8.8%p.

**Table 4 (Real-world, 7 tasks)**: OFlow 69% vs GR00T-N1.5 51% vs π0 41%. Panda-Car(동적 타깃)에서 70 vs 20, Handover 75 vs 45로 동적 상호작용 태스크에서 특히 큰 격차.

**Table 5 (Ablation)**: DINO + Object-Aware + Foresight 조합이 76.1%로 최고. Foresight 없이 DINO만 쓰면 70.0%에 그쳐, 성능 이득의 핵심은 "미래 예측"임을 시사. **Fig. 7**: prediction horizon $M=4$가 최적(69.2%), multi-scale $K=\{1,2,4,8\}$이 single-scale을 상회.

**Module Efficiency**: VLM 1.6B/49ms, action head 1.2B/207ms, foresight 0.2B/120ms. action chunk(16 step) 당 1회만 호출되어 amortization 효과 우수.

## 4. 한계 및 미해결 문제

- Foresight 모듈이 별도 DINOv2 인코더와 12-layer DiT를 요구해 파라미터/지연이 증가하며, 저지연 실시간 제어(<10ms)에는 여전히 부담.
- Object-aware factorization이 학습 없는 K-Means에 의존하므로, DINOv2 feature 품질이 떨어지는 도메인(예: 저해상도/적외선)에서는 프로토타입이 무너질 위험.
- EBasket(56.6%)처럼 그리퍼가 바스켓 안으로 깊이 들어가는 가려짐 상황에서는 여전히 성능이 낮음(Table 2).
- Real-world 평가가 20회/태스크에 그치고, 사람 개입 없는 완전 자율 동적 상호작용 스케일은 아직 제한적.

## 5. 총평

- **Novelty**: ★★★★☆ — "pixel foresight → DINO latent foresight + unsupervised object prototypes"라는 설계는 명확히 새로운 trade-off 지점을 찾음. ControlNet-style zero-init은 영리하나 개별 요소는 조합.
- **Practical impact**: ★★★★☆ — GR00T-N1.5 플러그인 형태라 채택 장벽이 낮고, 동적 태스크(+18%p)·perturbation robustness(+3.6~+5.9%p)에서 설득력 있는 gain.

## 6. 예상 질문

**Q1. 왜 픽셀이 아닌 DINOv2 latent를 예측하는가?**
A. Sec. 4.2의 perturbation breakdown이 답이다. 픽셀 foresight는 appearance noise에 취약하지만 DINOv2 semantic은 viewpoint/noise/layout shift에서 안정적이다(Fig. 5).

**Q2. Object-aware factorization에 학습이 필요 없는 이유는?**
A. DINOv2 feature 자체가 object-level semantic을 이미 내포(Sec. 3.2). 단순 K-Means로도 Fig. 4처럼 의미 있는 prototype이 나오며, 추가 supervision 없이 robust한 것이 Table 5의 ablation(OA w/o supervision)에서 증명됨.

**Q3. Foresight horizon $M$을 키우면 왜 오히려 성능이 떨어지는가?**
A. Fig. 7에서 $M=4$(69.2)가 peak이고 $M=8$은 68.3으로 하락. 멀리 예측할수록 compounding error가 커지고, manipulation은 단기 reactive 의사결정이 더 중요하기 때문으로 해석된다.

**Q4. ControlNet-style zero-init이 필수인가?**
A. GR00T-N1.5 사전학습 표현을 보존하면서 새 conditioning을 점진적으로 학습하기 위한 선택. Zero-init이 없으면 Stage 2에서 VLA 백본 표현이 foresight 노이즈에 의해 손상될 수 있음(Sec. 3.3, Eq. 10).

<!-- VERIFIED: pdf -->
