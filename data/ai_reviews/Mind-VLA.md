# Mind-VLA: Instruction-Aware Spatial Representation Alignment for Vision-Language-Action Models

> **한 줄 요약**: 기존 3D-aware VLA의 정렬 타깃이 "장면 전체"라서 instruction-agnostic하다는 점을 지적하고, 지시문이 지목한 **타깃 물체의 tri-view(top/front/side)** 를 VAE latent 예측 + 4-layer VGGT feature 정렬로 학습 시에만 감독하는 방법. 345M 백본으로 LIBERO 93.9%, CALVIN ABC-D 4.47을 달성하고, 실로봇 25% 가림 조건에서 성능 하락이 13pp에 그친다(scene-image 정렬 ablation은 29pp 붕괴).

- arXiv: 2608.04633 (2026-08-05)
- 소속: Nanjing University / Institute of Automation, CAS / UCAS
- 코드: "Code will be publicly available" (논문 시점 미공개)

---

## 1. 개요 (Overview)

Mind-VLA는 VLA 모델의 3D 이해를 강화하는 **학습 시 보조 감독(training-time auxiliary supervision)** 계열 방법이다. 추론 시 입력은 RGB + 언어 + 상태로 바닐라 VLA와 동일하며, VAE·VGGT·보조 디코더는 모두 제거된다. 핵심 주장은 "3D 감독의 대상을 장면이 아니라 **지시문이 지정한 물체**로 바꾸면, 동일 아키텍처·동일 추론 비용으로 미세 조작과 가림(occlusion) 강건성이 올라간다"는 것이다.

## 2. 문제 정의 (Problem Statement)

논문은 3D-aware VLA를 두 갈래로 정리한다.

| 패러다임 | 대표 연구 | 방식 | 문제 |
|---|---|---|---|
| (a) 3D를 **입력**으로 | PointVLA, GeoVLA, OG-VLA, StereoVLA | point cloud / RGB-D / orthographic view를 추론 시 투입 | depth 센서·3D 전처리 필요 → 배포 비용 증가 |
| (b) 3D를 **학습 감독**으로 | Spatial Forcing, GLaD, QDepth-VLA | frozen 3D foundation model feature에 VLA 표현을 정렬 | 추론 비용은 없으나 **감독 신호가 장면 전체** |

(b)의 구조적 결함이 이 논문의 출발점이다. 타깃 물체·distractor·배경이 한 덩어리로 인코딩되므로 (1) 타깃이 주변 물체와 시각적으로 유사할 때 판별이 어려워지고, (2) 타깃이 부분 가림될 때 모델이 해당 물체의 3D 구조를 보존하도록 명시적으로 학습된 적이 없어 신뢰도가 떨어진다.

## 3. 핵심 기여 (Key Contributions)

1. **문제 규정**: 기존 3D-aware VLA의 instruction-agnostic 3D 모델링을 한계로 정식화.
2. **Mind-VLA 제안**: 타깃 물체 tri-view VAE latent 예측 + 타깃 물체 VGGT feature 다층 정렬로 instruction-aware 3D 이해를 부여.
3. **결과**: 345M 백본으로 LIBERO 93.9%, CALVIN ABC-D 4.47, 실로봇 가림 조건 평균 54% (Seer 대비 +32pp).

## 4. 방법론 (Methodology)

### 4.1 베이스라인 정식화

입력 토큰화 → causal transformer → action query → diffusion action head(DiT). 액션 손실은 표준 epsilon-prediction MSE이고 추론은 DDIM 샘플링이다.

베이스 VLA의 보조 손실은 `L_VLA = L_act + λ_aux · L_aux`이며, 본 논문에서 `L_aux`는 **per-patch RGB 재구성(L_obs)** 과 **dense 2D motion 예측(L_traj)** 으로 인스턴스화된다(DreamVLA/Seer 계열 관행).

기존 3D-aware VLA는 여기에 `L_3D = D[φ(h_img_t), g_t]`를 추가하는데, 문제는 `g_t`가 **전체 장면에서 계산되고 지시문 ℓ과 무관**하다는 점이다. Mind-VLA는 바로 이 `L_3D`를 instruction-aware 감독으로 교체한다.

### 4.2 타깃 물체 tri-view 준비

- 시뮬레이션: 물체 mesh로부터 top/front/side **orthographic 사전 렌더링**
- 실로봇: 물체당 핸드헬드 카메라로 근사 정준 시점 3장 촬영 (캘리브레이션 리그 없음, 물체당 1분 미만)
- 각 지시문은 태스크 정의를 통해 고정된 tri-view와 페어링됨 → **오프라인 전처리이므로 추론 시 시점 변화·가림의 영향을 받지 않는다**

### 4.3 Tri-view latent 예측 (L_tri)

frozen Stable Diffusion VAE로 tri-view를 인코딩해 raw RGB 대비 **48배 작은** stacked latent `Z_m(ℓ)`를 만들고, object query 토큰 `q_tri_t`를 작은 MLP로 디코딩한 `Ẑ_t`와의 latent-space MSE를 K 스텝 평균한다. `Z_m(ℓ)`는 에피소드 내에서 고정이지만 **지시문이 다른 물체를 가리키면 바뀌므로**, 손실이 언어와 물체 기하를 직접 묶는다.

### 4.4 VGGT 다층 정렬 (L_geo)

백본의 **중간 4개 층**을 VGGT의 대응 층과 페어링한다. 페어 j에 대해 뷰·공간 위치로 mean-pool한 VGGT feature `g_m(ℓ),j`와, 같은 방식으로 pool한 image-token feature `h̄_t,j`를 투영 헤드 `φ_j`로 사상한 뒤 **코사인 거리(1 - cos)** 를 최소화한다. Spatial Forcing/GLaD와의 유일한 차이는 VGGT의 입력이 full-scene image가 아니라 **타깃 물체 tri-view**라는 점이다.

### 4.5 최종 목적함수

```
L_MindVLA = L_VLA + λ_tri · L_tri + λ_geo · L_geo
```

추론 시 VAE·VGGT·모든 보조 디코더 제거. 남는 비용은 백본 시퀀스의 보조 query 토큰뿐이라 바닐라 VLA 대비 무시할 수준.

## 5. 아키텍처 (Architecture)

| 구성요소 | 내용 |
|---|---|
| 시각 입력 | 224×224×3 RGB 2장 (primary + wrist), frozen vision encoder |
| 언어 | frozen CLIP text encoder |
| 상태 | 8-D proprioception, linear projection |
| 백본 | causal transformer, context window K, **345M** |
| Query 그룹 | scene-level / object-level / action-level (query-group attention mask로 분리 + 시간 인과성 강제) |
| Scene query 디코딩 | per-patch RGB, dense 2D motion |
| Object query 디코딩 | tri-view VAE latent (MLP head) |
| Geometry 정렬 | frozen VGGT 4개 중간 층 |
| Action head | Diffusion Transformer, 다음 3개 액션(7-D) 예측, DDIM |

## 6. 데이터 및 학습 (Data & Training)

- **LIBERO**: 4개 스위트(Spatial/Object/Goal/Long), 각 10 태스크 × 50 데모. 선행 연구(DreamVLA)와의 비교 가능성을 위해 **LIBERO-90 사전학습 후 스위트별 fine-tune**, 태스크당 20 rollout으로 성공률 보고.
- **CALVIN ABC-D**: 1000 rollout, 표준 프로토콜, 평균 완료 길이(Avg. Len.).
- **실로봇**: UFactory xArm 6 (듀얼암 워크스페이스에서 한 팔 사용), static RealSense D455 + wrist RealSense D435. **DROID 사전학습 후** 태스크별 50 데모 fine-tune. 추론은 RTX 5090 1장.
- 학습: AdamW, **8× NVIDIA H20**.

## 7. 실험 결과 (Experiments)

### LIBERO (Table 1a, 7B급 대형 VLA와 비교)

| Method | Params | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|---|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| TraceVLA | 7B | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| SpatialVLA | 4B | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 7B | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| CogACT | 7B | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 |
| GLaD | 7B | 95.0 | 97.4 | 94.4 | 89.4 | 94.1 |
| π0 | 3B | 96.8 | 98.8 | 95.8 | 85.2 | **94.2** |
| **Mind-VLA** | **345M** | **98.0** | **98.0** | 92.0 | 87.5 | **93.9** |

π0(94.2)·GLaD(94.1)와 사실상 동률이면서 백본은 **9~20배 작다**. compact 백본 비교(Figure 3)에서는 DreamVLA 대비 평균 +1.3pp, LIBERO-Object에서 +4.0pp.

### CALVIN ABC-D (Table 1b, 1000 rollout)

| Method | T5 | Avg. Len. |
|---|---|---|
| 3D Diffuser Actor | 41.2 | 3.27 |
| OpenVLA† | 43.5 | 3.27 |
| CLOVER | 45.4 | 3.53 |
| RoboDual | 54.4 | 3.66 |
| π0† | 59.9 | 3.92 |
| Seer | 74.0 | 4.28 |
| VPP | 75.0 | 4.29 |
| **Mind-VLA** | **79.4** | **4.47** |

장기 horizon에서 격차가 더 크다 (VPP +0.18, π0 +0.55).

### 실로봇 (Table 3, 셀당 30 trial)

| 조건 | OpenVLA | Seer | Mind-VLA (scene-image VGGT) | **Mind-VLA** |
|---|---|---|---|---|
| Normal 평균 (5-task) | — | 50 | 58 | **69** |
| Occluded(~25%) 평균 | 7 | 22 | 28 | **54** |
| 하락폭 | −21pp | −23pp | −29pp | **−13pp** |

Pick(Banana/Potato), Place, Drawer 3종. 가림 조건은 Pick·Drawer에만 적용되며 타깃의 약 25%를 물리 차폐물로 가린다.

## 8. 비교 분석 (Comparison)

| 축 | Spatial Forcing / GLaD | QDepth-VLA | 3D 입력형(PointVLA 등) | **Mind-VLA** |
|---|---|---|---|---|
| 감독 신호 | scene-level VGGT | 양자화 depth 토큰 | point cloud/RGB-D 입력 | **타깃 물체 tri-view VGGT + VAE latent** |
| 지시문 조건화 | ✗ | ✗ | ✗ | **✓** |
| 추론 시 3D 모듈 | 없음 | 없음 | 필요 | 없음 |
| 정준 시점 prior | ✗ | ✗ | △ | **✓ (현재 관측 밖 시점 포함)** |

DreamVLA(모션·시맨틱 예측), PALM(진행도 인지 affordance), Geometry Forcing(비디오 생성 통한 VGGT 전이) 같은 예측형 보조 목적 계열과는 **상보적**이다. Mind-VLA는 추론 입력을 그대로 두고 **감독의 대상만 instruction-aware하게** 바꾼다.

## 9. 절제 연구 (Ablation)

### 점진적 구성요소 추가 (Table 2 상단)

| ID | 구성 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| A0 | 바닐라 VLA + image recon. | 95.5 | 92.0 | 88.0 | 82.5 | 89.5 |
| A1 | + scene-level trajectory pred. | 97.0 | 93.5 | 89.5 | 85.0 | 91.3 (+1.8) |
| A2 | + 타깃 물체 tri-view latent pred. | 97.5 | 95.0 | 91.0 | 86.5 | 92.5 (+1.2) |

### Instruction-aware vs instruction-agnostic 정렬 (Table 2 하단, A2 위에 추가)

| ID | 구성 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| A3a | + scene-image VGGT (instr.-agnostic) | 97.5 | 94.0 | 92.0 | **90.0** | 92.8 |
| A3b | + 타깃 물체 tri-view VGGT (full) | **98.0** | **98.0** | **92.0** | 87.5 | **93.9** |

4개 중 3개 스위트와 평균에서 앞서고, **LIBERO-Object에서 +4.0pp**로 격차가 가장 크다. 반대로 **LIBERO-Long에서는 scene-image 변형이 90.0 vs 87.5로 앞선다** — 논문은 장기 horizon이 전역 공간 맥락에서 이득을 본다고 해석한다. (저자가 불리한 셀을 숨기지 않고 명시한 점은 신뢰도를 높인다.)

### Object focus vs canonical-view prior 분해

tri-view 타깃을 **현재 이미지에서 잘라낸 instruction-aware SAM crop**으로 대체:

- scene-image VGGT: LIBERO-Object 94.0
- SAM crop (focus만): **96.5** → object focus 기여 **+2.5pp**
- full tri-view: **98.0** → canonical view 기여 **추가 +1.5pp**

즉 깨끗한 시뮬레이션에서는 focus가 이득의 대부분을 설명하지만, **현재 관측이 불완전할 때(가림) canonical view prior가 결정적**이 된다. 실로봇 가림 실험의 −13pp vs −29pp 대비가 이 주장의 증거다. 이 ablation이 논문에서 가장 정보량이 큰 부분으로, scene-image VGGT 변형은 아키텍처·scene-level 손실·VGGT 정렬을 모두 공유하고 **정렬 타깃만 다르다**.

## 10. 강점 (Strengths)

1. **문제 규정이 날카롭다**: "3D 감독이 instruction-agnostic하다"는 한 문장으로 Spatial Forcing/GLaD/QDepth-VLA를 관통하는 공통 결함을 짚었고, 그에 정확히 대응하는 최소 개입을 설계했다.
2. **비용 구조가 좋다**: 추론 인터페이스 무변경, 보조 모듈 전부 제거. 345M로 3~7B급과 동률.
3. **Ablation 설계가 정직하다**: scene-image VGGT 변형이 정렬 타깃만 다른 통제군이라 인과 귀속이 명확하고, focus(+2.5)와 canonical view(+1.5)를 SAM crop으로 분리한 것은 특히 잘 설계됐다.
4. **시뮬레이션의 작은 이득과 실로봇의 큰 이득을 연결**: LIBERO-Object에서 +1.5pp에 불과한 canonical-view 기여가 25% 가림에서 16pp 차이로 증폭되는 서사가 설득력 있다.
5. **실로봇 tri-view 획득이 현실적**: 캘리브레이션 리그 없이 핸드헬드 3장, 물체당 1분 미만.

## 11. 약점 (Weaknesses)

1. **물체 어휘가 유계(bounded)** — 저자도 인정하는 최대 한계. 타깃 물체마다 tri-view 1회 셋업이 필요하므로 **임의의 미지 물체에 대한 zero-shot 배포가 원천적으로 불가**하다. 이는 "privileged supervision"이며 open-vocabulary를 지향하는 VLA 흐름과 상충한다.
2. **지시문 = 단일 타깃 물체 가정** — 그래서 LIBERO-Long에서 DreamVLA(89.5)에 87.5로 뒤지고, scene-image 정렬(90.0)에도 뒤진다. 연쇄 장기 태스크는 장면 수준 모델링을 요구한다.
3. **m(ℓ) 획득 절차가 불투명** — "지시문이 참조하는 타깃 물체를 얻는다"고만 하고, 시뮬레이션은 태스크 정의로 페어링된다. 즉 **타깃 식별이 데이터셋 메타데이터로 주어진 것**이며, grounding 실패 시의 견고성은 평가되지 않았다.
4. **실로봇 평가 범위 협소** — 태스크 3종, 타깃 물체 2종(banana/potato), 조건당 30 trial. +32pp라는 헤드라인 수치의 통계적 근거는 얇다.
5. **하이퍼파라미터·비용 미보고** — λ_tri, λ_geo 값과 민감도, VGGT 페어링 층 선택 근거, 학습 시간 오버헤드가 없다. 4개 층 선택이 임의적으로 보인다.
6. **코드 미공개** — "will be publicly available" 상태라 재현성 검증 불가.
7. **A3a가 LIBERO-Long에서 이기는 현상에 대한 처방 부재** — scene + object 감독을 함께 쓰는 하이브리드는 자연스러운 다음 수인데 실험되지 않았다.

## 12. 총평 (Overall Assessment)

| 항목 | 평가 |
|---|---|
| **Novelty** | ★★★★☆ — "정렬 타깃을 instruction-aware하게"는 개념적으로 단순하지만 아무도 명시적으로 하지 않던 각도 |
| **Technical depth** | ★★★☆☆ — 구성요소(VAE latent 회귀 + VGGT 코사인 정렬)는 모두 기성품 조합. 새 손실·새 아키텍처는 없음 |
| **Experimental rigor** | ★★★★☆ — 통제된 ablation, focus/canonical-view 분해, 불리한 셀 명시가 좋음. 실로봇 표본은 얇음 |
| **Practical impact** | ★★★☆☆ — 추론 비용 0은 큰 장점이나 bounded object vocabulary가 적용 범위를 크게 제한 |
| **Writing quality** | ★★★★☆ — 문제 규정→방법→검증의 논리가 매우 깔끔 |

**종합**: 논문의 진짜 기여는 LIBERO 93.9%가 아니라 **"3D 감독은 무엇에 대한 감독이어야 하는가"** 라는 질문을 던진 것이다. LIBERO에서의 +1.1pp(93.9 vs 92.8)는 그 자체로는 미미하지만, 같은 변경이 실로봇 가림에서 26pp 차이(54 vs 28)로 증폭되는 것을 보임으로써 시뮬레이션 벤치마크가 이 축의 차이를 거의 측정하지 못한다는 점까지 함께 드러낸다. 반대로 tri-view 사전 준비 요구는 zero-shot 일반화를 포기하는 대가이며, 저자도 이를 명시한다. 실용적 다음 단계는 tri-view를 오프라인 자산 대신 생성 모델로 합성해 어휘 제약을 푸는 것이다. **종합 평점 8.0/10.**

---

## 부록: 예상 날카로운 질문

| # | 질문 | 핵심 답변 요점 |
|---|---|---|
| 1 | m(ℓ) 타깃 물체는 어떻게 얻나? grounding 모델이 틀리면? | 시뮬레이션은 태스크 정의로 tri-view가 고정 페어링, 실로봇은 물체별 사전 촬영. 즉 **주어진 메타데이터**이며 grounding 실패 강건성은 미평가 |
| 2 | tri-view가 오프라인이면 시점 변화에 무관한 건 당연한 것 아닌가? | 맞다. 그것이 설계 의도이자 동시에 한계 — 감독은 관측 불변이지만 그 대가가 bounded vocabulary |
| 3 | LIBERO-Long에서 scene-image 변형에 지는 것은 방법의 반례 아닌가? | 부분적으로 그렇다. 논문도 인정하며 연쇄 장기 태스크는 전역 맥락이 필요하다고 해석. scene+object 하이브리드 미실험 |
| 4 | VGGT 정렬 층을 왜 4개, 왜 그 위치인가? | 근거·민감도 분석 없음. 임의 선택으로 보임 |
| 5 | λ_tri, λ_geo 민감도는? | 미보고 |
| 6 | 345M이 93.9%면 백본 크기가 문제가 아니라 감독이 문제라는 뜻인가? | 논문의 함의는 그렇지만, LIBERO 자체의 포화도 고려해야 함 (상위권이 93~94에 몰려 있음) |
| 7 | 실로봇 +32pp는 물체 2종·30 trial 기준인데 신뢰 가능한가? | 표본이 얇다. 다만 scene-image 통제군이 같은 조건에서 −29pp 붕괴한 대비는 방향성 근거로 유효 |
| 8 | SAM crop 변형이 96.5인데, crop이 tri-view보다 훨씬 싸다. 실용적으로는 crop이 낫지 않나? | 깨끗한 시뮬레이션에서는 타당. 그러나 가림 상황에서 현재 관측 기반 crop은 정보 자체가 손실되므로 canonical view가 필요 |
| 9 | 학습 오버헤드(VAE/VGGT teacher forward)는? | 미보고. 추론 오버헤드만 "negligible"로 언급 |
| 10 | 다중 물체 지시("A를 B 위에 놓아라")는? | 단일 타깃 매핑 가정이라 미지원. 저자가 한계로 명시 |

<!-- VERIFIED: pdf -->
