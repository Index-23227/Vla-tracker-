# GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation 세미나 리뷰

> **한 줄 요약**: 1024개의 학습 가능한 GaussianDream Query를 VLM 인코더에 삽입하고, **TGE(Temporal Gaussian Evolution)** 모듈로 현재/단기 미래의 3D Gaussian 장면을 압축한 prefix를 만들어 flow-matching action expert에 conditioning. 학습 시에는 두 보조 head(현재 reconstruction, 미래 prediction)를 RGB+depth+pseudo 3D scene-flow로 감독하지만, **추론 시 보조 head는 모두 버리고 prefix만 남긴다** — LIBERO 98.4%, RoboCasa Human-50 54.8%, real-robot 50.0% (π₀.₅ 34.4% 대비 +15.6pp).

---

## 1. 배경 및 동기

기존 VLA 정책은 사전학습 vision-language 모델의 의미론적 prior를 잘 가져오지만 **명시적 3D 공간 구조, dense geometric supervision, 미래 환경 진화**가 빠져있다. 정밀 조작에서는 이 셋 모두 중요하다. 한쪽 방향으로 video-based world model이 시도되어 왔으나 추론 시 비싼 rollout/렌더링이 필요해 실배포에 부담이 크다. GaussianDream의 핵심 베팅은 "3D Gaussian world model을 **보조 신호로만 학습**시키고, 추론 시 그 무거운 부분을 전부 잘라낸다"는 비대칭 설계다.

---

## 2. 방법론 심층 분석

### 2.1 전체 구조

- **VLM 인코더**: 입력 이미지/언어/state → 토큰 시퀀스 (2048-d)
- **GaussianDream Queries**: 1024개의 학습 가능한 query 토큰을 인코더에 삽입 — 이 query들이 prefix가 됨
- **TGE(Temporal Gaussian Evolution) 모듈**: 다중 스케일 VGGT 피처를 **t-10, t-5, t** 세 시점에 걸쳐 처리. **12 attention block × 8 head**, 프레임-별 spatial interaction과 time-slot temporal attention을 교차. 2048-d ↔ 512-d 프로젝션.
- **두 보조 head (학습 전용)**:
  - **Current Reconstruction head**: 1024 query → 32×32 grid → upsample → **256×256×128 feature map**. Geometry head로 depth/회전(쿼터니언)/scale/opacity 예측, appearance head로 degree-1 SH 계수 예측. depth unprojection으로 256×256 Gaussian centers 산출.
  - **Future Prediction head**: 공유 256×256 feature map + learnable horizon embedding $e_\Delta$로 horizon-conditioned **center displacement**를 velocity head로 예측. 비-위치 속성(rotation/scale/SH)은 현재 head의 값을 재사용. 감독 horizon: **t+1 ~ t+5**.
- **Action policy**: flow-matching 형식. observation/언어/robot state + GaussianDream prefix에 conditioning.

### 2.2 학습 손실

**Stage I — 사전학습(보조 head 포함)**:

$$
\mathcal L_\text{GD} = \lambda_\text{cur}^\text{depth} \mathcal L_\text{cur}^\text{depth} + \lambda_\text{cur}^\text{render} \mathcal L_\text{cur}^\text{render} + \sum_\Delta w_\Delta \left( \lambda_\text{depth} \mathcal L_\text{depth}^{(\Delta)} + \lambda_\text{render} \mathcal L_\text{render}^{(\Delta)} + \lambda_\text{flow} \mathcal L_\text{flow}^{(\Delta)} \right)
$$

- **depth loss**: pseudo depth(Depth Anything V2)와 L2
- **rendering loss**: Gaussian splatting의 RGB consistency
- **flow loss**: 예측 center displacement vs **pseudo 3D scene flow**(RAFT optical flow를 depth로 back-project)와 L1

**Stage II — 정책 학습**:

$$
\mathcal L = \mathcal L_\text{act} + \lambda_\text{GD} \mathcal L_\text{GD}, \quad
\mathcal L_\text{act} = \mathbb E \| v_\theta(\tau\epsilon + (1-\tau) a^*_t, c_t, \tau) - (\epsilon - a^*_t) \|^2_2
$$

flow-matching 표준 형식의 action loss를 prefix conditioning $c_t$ 위에서 사용.

### 2.3 추론 시 디자인 — 비대칭의 핵심

> "During online inference, all auxiliary decoding heads are discarded. The policy retains only a compact, information-rich GaussianDream prefix to condition action generation, bypassing test-time Gaussian decoding, geometric rendering, video rollout, and additional planning."

즉, 학습은 dense 3D 감독(Gaussian splat + depth + scene-flow)이지만 **추론에는 prefix conditioning만 남는다**. 이것이 video-based world model(WAM 등) 대비 latency를 줄이는 핵심.

---

## 3. 데이터셋 및 평가 프로토콜

- **LIBERO**: 50 demo / 태스크, 평가는 태스크당 50 trial
- **RoboCasa**: Human-50 split, 24 long-horizon 태스크, 5 scene, 태스크당 50 trial
- **Real-robot**: dual-arm 플랫폼 — leader arm은 teleoperation 데이터 수집용, follower arm이 자율 실행. 두 RGB 카메라(agent + wrist), 4개 scene(A/B/C/D)

---

## 4. 실험 결과

### 4.1 LIBERO (Table 1)

| 방법 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | 97.8 | 98.8 | 97.6 | 92.4 | 96.7 |
| GeoPredict | 98.0 | 98.2 | 95.7 | 94.0 | 96.5 |
| QDepth-VLA | 97.6 | 96.6 | 95.2 | 90.0 | 94.9 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| VLA-4D | 97.9 | 98.6 | 97.8 | 94.8 | 97.4 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| Spatial Forcing | 98.6 | 98.4 | 98.2 | 95.4 | 97.6 |
| **GaussianDream** | **99.0** | **99.6** | **99.0** | **96.0** | **98.4** |

LIBERO-Long에서 LingBot-VA 98.5 / 3D-CAVLA 96.1 와 비교해 GaussianDream 96.0 — 평균 1위지만 Long suite에서는 LingBot-VA에 약간 뒤지는 모습.

### 4.2 RoboCasa Human-50 (Table 2)

| 방법 | Pick&Place | Doors/Drawers | Others | Avg |
|---|---|---|---|---|
| π₀ | 14.0 | 53.1 | 58.5 | 42.4 |
| π₀.₅ | 36.0 | 46.5 | 39.5 | 40.1 |
| BC-Transformer | 3.8 | 46.7 | 38.0 | 28.8 |
| GWM | 14.8 | 54.3 | 49.8 | 39.3 |
| GeoPredict | 22.7 | 75.1 | 62.4 | 52.4 |
| Being-H0.5 | 36.0 | 71.7 | 57.6 | 53.9 |
| **GaussianDream** | **43.8** | 66.3 | 54.4 | **54.8** |

특히 **Pick&Place에서 43.8%**로 차상위 36.0(π₀.₅/Being-H0.5)을 크게 상회 — 3D 위치 grounding이 가장 필요한 카테고리에서 강점.

### 4.3 Real-Robot (Table 3)

| 방법 | Scene-A | Scene-B | Scene-C | Scene-D | Avg |
|---|---|---|---|---|---|
| π₀.₅ | 42.5 | 50.0 | 25.0 | 20.0 | 34.4 |
| **GaussianDream** | **55.0** | **70.0** | **35.0** | **40.0** | **50.0** |

전 scene에서 baseline 대비 +10pp 이상 — **+15.6pp 평균** 향상. 특히 어려운 Scene-D에서 20.0 → 40.0으로 2배.

### 4.4 추론 latency

| 방법 | per-chunk latency |
|---|---|
| GaussianDream (standard) | **531 ms** |
| GaussianDream (with diagnostic decoder) | 569 ms |
| WAM (video world-model baseline) | **>700 ms** |

학습 단계의 무거운 보조 head 비용을 추론 시 떼어내면서 video world model 대비 ~24% 빠른 추론을 보임.

---

## 5. Ablation 분석 (Table 4)

| Variant | Current Recon | Future Pred | Render | Depth | LIBERO Avg |
|---|---|---|---|---|---|
| V1 | ✓ | ✗ | ✗ | ✗ | 97.0 |
| V2 | ✓ | ✗ | ✓ | ✓ | 97.3 |
| V3 | ✓ | ✓ | ✗ | ✓ | 97.5 |
| V4 | ✓ | ✓ | ✓ | ✗ | 97.2 |
| **Full** | ✓ | ✓ | ✓ | ✓ | **98.4** |

- **Future Prediction 추가만으로** V2→V3: 97.3→97.5 (+0.2)
- **Rendering branch 제거** V3→V5: 97.5→? (full 98.4와 비교하면 -0.9는 Render+Depth 둘 다 필요)
- **Depth branch 제거** V4: 97.2 → Full 98.4 (depth 감독이 +1.2pp 기여, 가장 큰 단일 기여)
- 네 신호를 모두 결합한 full configuration이 단일 신호 V1(97.0) 대비 +1.4pp

---

## 6. 한계 및 의의

**한계**:
- LIBERO Long suite에서 LingBot-VA(98.5)에 0.5pp 뒤지는 점 — 4-suite 평균 1위지만 절대적 SOTA는 아님.
- Real-robot은 4 scene만 — 더 다양한 환경/long-horizon real-world 검증 필요.
- Pseudo 감독(Depth Anything V2 + RAFT)에 의존 — depth quality가 떨어지는 outdoor / 투명 물체 시나리오에서 어떻게 동작하는지 미검증.
- 학습 단계의 보조 head 비용(GPU memory, training time)이 명시되지 않음. 60K step / A100이라는 정보만 있음.
- Action head 자체의 파라미터 수, 전체 VLA size 미보고.

**의의**:
- **"training에는 dense 3D supervision, inference에는 prefix만"** 이라는 비대칭 설계가 video world model 대비 24% 빠른 추론 + Real-robot +15.6pp라는 실측 결과로 정당화됨.
- RoboCasa Pick&Place에서 43.8% — 3D 위치 grounding이 핵심인 카테고리에서 최대 차이.
- Plug-in 형태라 다른 VLA backbone으로 이식 가능성 시사 (다만 본 논문에서는 단일 backbone만 검증).
- 코드 공개(https://github.com/TuojingAI/GaussianDream) — 후속 3D-VLA 연구의 baseline으로 사용 가능.

---

## 7. 예상 날카로운 질문 모음

| # | 질문 | 핵심 답변 |
|---|---|---|
| 1 | 추론 시 prefix만 남기는 게 정말 3D 정보를 보존하는가? | Ablation Table 4: 보조 head 제거 후에도 prefix는 학습된 latent 상태로 3D 신호를 압축 보유. depth branch 제거 시 -1.2pp로 가장 큰 영향 |
| 2 | Pseudo depth(DAV2)의 metric scale 오류는? | scene flow 감독은 displacement에 작용해 절대 scale에 덜 민감. 다만 RoboCasa Others 54.4%(GeoPredict 62.4)에서 약점이 보임 |
| 3 | TGE 12 layer는 과한가 / 적은가? | Ablation 미보고 — 향후 layer 수 / horizon 수 ablation 필요 |
| 4 | LIBERO-Long 96.0이 SOTA가 아닌 이유? | LingBot-VA 98.5가 더 높음. Long suite에서는 단순 3D 정보가 아니라 long-term planning이 더 결정적일 수 있음 |
| 5 | Real-robot 50.0% — sample 수가 적지 않은가? | scene당 trial 수 미명시(아마 20개 내외). 통계적 유의성은 추가 시드 필요 |
| 6 | π₀.₅를 baseline으로만 비교하는 게 공정한가? | RoboCasa에서는 Being-H0.5(53.9), GeoPredict(52.4) 등 더 강한 baseline 포함. LIBERO도 9개 비교 |
| 7 | 학습 비용은? | 60K step / A100 / global batch 24만 명시. 보조 head 포함이라 실제 메모리/시간은 상당할 것으로 추정 |
| 8 | 1024 query × 2048-d = 2M latent. 정보 bottleneck인가? | 32×32 grid + 256×256 Gaussian centers로 upsample되는 구조 — 학습 시 supervision 신호를 다 통과시키므로 정보 손실은 작을 것 |

<!-- VERIFIED: pdf -->
