# Event-VLA: Action-Conditioned Event Fusion for Robust Vision-Language-Action Model

> **arXiv 2606.29384** (ShanghaiTech / HKUST-GZ / UMich / SJTU, 2026-06-28)
>
> ⚠️ **주의 — 동명이인 논문**: 트래커에 이미 있는 **EventVLA** (arXiv 2606.20092, QwenOFT + Keyframe Evidence Memory, "event"=키프레임 이벤트 메모리)와는 **완전히 다른 논문**이다. 이 논문의 "Event"는 **이벤트 카메라(event camera)** 를 의미하며, 저조도 환경에서 RGB가 붕괴할 때 이벤트 스트림을 액션 경로에 주입해 강건성을 확보하는 연구다. 구분을 위해 본 트래커에서는 하이픈 표기 "Event-VLA"(`event_vla.yaml`)를 사용한다.

## 1. 한 줄 요약

OpenVLA(Prismatic-7B) 백본을 유지한 채, 이벤트 카메라 스트림을 PREI(Physical Residual Event Integration) 3채널 잔차 맵으로 압축하고 백본 **바깥에서** gated cross-attention + query-guided routing으로 액션 경로에만 주입하는 이벤트 강화 VLA. 정상 조명 LIBERO 96.5%로 RGB SOTA급 유지, 신규 저조도 벤치마크 LIBERO-Cross LL-Severe에서 95.6%로 MM-ACT(69.6%)·OpenVLA-OFT(61.2%) 대비 압도적, 실제 Franka near-dark에서 52.5%(π₀ 15.0%).

## 2. 문제 정의

- 기존 VLA는 잘 조명된 안정적 RGB 관측을 암묵적으로 가정 → 조도 변화, 센서 노이즈, 모션 블러 하에서 semantic grounding과 액션 예측이 붕괴.
- 이벤트 카메라는 로그 밝기 변화만 비동기적으로 감지 → high dynamic range, 저지연, 저조도에서도 모션/엣지 단서 보존.
- 핵심 질문(인터페이스 문제): **비동기 고주파 이벤트 스트림을 사전학습 VLA에 "어떻게" 통합할 것인가?** 전역 semantic 토큰 공간에 직접 병합(unified encoding)하면 사전학습 prior 훼손 + 지연 폭증, adapter 융합은 성능 부족.

## 3. 핵심 기여

1. **PREI**: 이벤트 활동을 action-time 물리 잔차로 분해하는 경량 표현 — instantaneous(τ=3ms decay, 최근 모션), salient(τ=10ms, 국소 정규화된 두드러진 활동), persistent(윈도우 내 이벤트 카운트, 윤곽 유지) 3채널, 윈도우 H=40ms, ρ=tanh 정규화.
2. **Action-conditioned event interface**: 이벤트 토큰을 백본 self-attention에 넣지 않고, 백본 통과 후 gated cross-attention(Z_fused = Norm(Z_llm + Γ⊙Z_attn))으로 융합한 뒤 common/action/event 3종 learnable query로 액션 헤드와 보조 이벤트 헤드에 라우팅.
3. **LIBERO-Cross**: LL-Mild(2.32EV↓, SNR 15.5dB)/LL-Dark(3.24EV↓, -0.2dB)/LL-Severe(4.78EV↓, -11.9dB, blur 3px) 3단계 점진적 저조도 벤치마크 + 학습된 RGB-to-event 시뮬레이터(v2e pseudo-event 감독, leakage 통제).
4. 시뮬레이션 + 실제 Franka(DAVIS 이벤트 카메라) 양쪽에서 검증.

## 4. 아키텍처

- **백본**: Prismatic-7B, OpenVLA(OXE 사전학습) 가중치로 초기화, LoRA rank 32로 미세조정. 입력 = [RGB 토큰; proprio 토큰; 언어 토큰; 액션 placeholder; common/action/event 쿼리].
- **이벤트 인코더**: PREI 맵 → 패치 특징 → VLA hidden 차원 투영. 동결된 DINO teacher로부터 feature distillation(N-ImageNet 네이티브 쌍 + v2e로 합성한 LIBERO 쌍, cos 유사도 손실 λg=0.7/λp=0.3, 50 epoch). 이벤트 토큰 Y_e는 **백본에 절대 입력되지 않음**.
- **융합**: Z_attn = CrossAttn(W_q Z_llm, W_k/v(Y_e + T_pos)); 게이트 Γ = σ(f_gate(Z_llm, Q_c, Q_a, Q_e, Z_attn)); 잔차 결합 후 Norm.
- **라우팅**: common+action 쿼리 → 액션 헤드(MLP), common+event 쿼리 → 보조 이벤트 헤드(deconv 디코더, 미래 PREI 예측). 라우팅은 fused 토큰과 쿼리 토큰의 단순 concat(parameter-free).
- **추론 지연**: no-event 162.2ms 대비 query routing +2.157ms만 추가(RTX 4090). unified encoding은 +62.874ms.

## 5. Action Head Category

**regression**. 액션 헤드는 MLP로 K=8 액션 청크를 연속값 직접 회귀하며 L1 손실(L_act)로 학습. diffusion/flow-matching 아님. 보조 이벤트 헤드(L_evt + L_deriv, λ=0.1/0.3)는 학습 시 정규화 전용이며 추론에는 불필요.

## 6. 학습 데이터 & 레시피

- LIBERO 서브벤치마크별 개별 학습: batch 64, action chunk 8, event history 8, ~140k step, 8x H100.
- AdamW lr 5e-5(50k step에서 0.1배 감쇠), weight decay 0.01, 보조 이벤트 손실 5k step warm-up.
- 이벤트 관측: LIBERO는 네이티브 이벤트가 없어 v2e pseudo-event로 U-Net식 RGB→event 시뮬레이터(f_prei)를 학습해 평가 시 직접 PREI 예측. 시뮬레이터는 정책과 독립 학습·동결, 시각 관측만 조건(언어/상태/GT 액션/성공 라벨 미사용, 미래 프레임 접근 금지)으로 leakage 통제.
- 실세계: Franka Research 3, 4개 태스크(red tape/tennis ball → box, milk → plate, ring the bell) × 20 demo = 80 시연, 다양한 조명 조건 포함.

## 7. LIBERO 정상 조명 결과 (Table 1)

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 | 95.4 |
| ResVLA / MM-ACT | — | — | — | — | 96.3 / 96.3 |
| **Ours w/o event** | 94.4 | 99.2 | 96.8 | 94.4 | 96.2 |
| **Ours** | 94.2 | **99.4** | **97.4** | **94.8** | **96.5** |

- 이벤트 경로를 꺼도(w/o event) 96.2%로 거의 동일 → 인터페이스가 사전학습 RGB-언어 액션 경로를 훼손하지 않음(Q2 검증). 각 서브벤치마크 500 trial(태스크당 50 rollout).

## 8. LIBERO-Cross 저조도 결과 (Table 2)

| Level | π₀ | OpenVLA-OFT | MM-ACT | **Ours** |
|---|---|---|---|---|
| LL-Mild | 91.7 | 93.9 | 95.9 | **96.1** |
| LL-Dark | 89.3 | 91.6 | 93.0 | **96.5** |
| LL-Severe | 61.9 | 61.2 | 69.6 | **95.6** |

- 핵심 결과: RGB 정책은 LL-Severe에서 25~35%p 붕괴하는 반면 Event-VLA는 정상 조명과 거의 동일한 95.6% 유지(Spatial 95.6/Object 97.2/Goal 97.8/Long 92.0).

## 9. 실세계 Franka 결과 (Table 4, 10)

| Method | Normal | Low-Light | Near-Dark | Avg |
|---|---|---|---|---|
| π₀ | 75.0 | 55.0 | 15.0 | 48.3 |
| OpenVLA-OFT | 70.0 | 57.5 | 12.5 | 46.7 |
| Ours w/o queries | 70.0 | 65.0 | 45.0 | 60.0 |
| **Ours** | 72.5 | **70.0** | **52.5** | **65.0** |

- 실제 DAVIS 이벤트 카메라 + wrist ZED + 외부 Orbbec, RTX 4090 배포. 조건당 40 trial(4 태스크 × 10). near-dark에서 RGB 정책은 사실상 무력(12.5~15%)한데 Event-VLA는 52.5%. 실패 사례: 대상 물체의 대비·상대 모션이 약해 이벤트 활성이 불충분하면 localization 모호 → 실패.

## 10. Ablation (Table 3, 9 — LIBERO-Cross LL-Severe)

- **표현**: no event 60.6% → time surface 91.2% → **PREI 95.6%**. Distillation 정렬 품질도 PREI 우위(R@1 81.4/75.9 vs 73.6/67.8, CKA 95.3 vs 94.2).
- **인터페이스**: unified encoding 95.1%(+62.9ms), RGB/event adapter 94.2%(+1.9ms), **query routing 95.6%(+2.2ms)** — 성능·지연 동시 최적.
- **쿼리**: w/o common 94.5, w/o event 95.2, full 95.6 → 3종 쿼리 모두 기여.
- **정규화**: 없음 94.8, w/o mask 95.1, full(masked future-PREI + derivative) 95.6 → 보조 미래 이벤트 감독이 일관된 이득.

## 11. 한계 & 토의

- LIBERO-Cross의 이벤트는 **시뮬레이터 산출물** — 실제 이벤트 카메라의 노이즈, 트리거 동역학, 캘리브레이션 오차를 완전히 반영하지 못함(저자 인정). 시뮬레이션 수치의 대외 비교 가능성에 유의.
- 실세계 평가가 4개 태스크 × 3개 조명으로 소규모; 장기 horizon·다양한 물체 검증 필요.
- 조명 열화에 특화 — occlusion 등 다른 열화 유형은 미검증. 이벤트 카메라 하드웨어 + 동기화가 배포 비용 증가.
- 정적 장면(모션 없음)에서는 이벤트 활성이 사라지므로 근본적으로 RGB semantic prior에 의존해야 함; 실패 사례가 이를 시사.
- LIBERO 수치는 서브벤치마크별 개별 학습(single-suite) 기준 — multi-suite 단일 모델과의 비교는 조건이 다름.

## 12. VLA-Tracker 관점에서의 의의

- 트래커 최초의 **이벤트 카메라 융합 VLA**. 기존 EventVLA(2606.20092, 키프레임 메모리)와 이름만 유사할 뿐 문제 설정·방법·벤치마크가 전혀 다름 — YAML `name: Event-VLA`, 파일 `event_vla.yaml`로 분리 등재.
- LIBERO 96.5%는 OpenVLA 백본 계열로는 상위권(OpenVLA-OFT 95.4% 상회)이나 순위표 기여보다는 **로버스트니스 축**(LIBERO-Cross, 저조도 실세계)의 기여가 본질.
- "새 modality를 백본 밖에서 액션 경로에만 주입"하는 인터페이스 설계는 tactile(Tactile-VLA, VLA-Touch)·stereo(StereoVLA) 등 타 modality 확장 연구와 비교할 만한 일반 원칙 — 지연 +2.2ms로 modality 추가라는 실용성이 강점.
- action_head_category: **regression** (L1 청크 회귀 MLP). 벤치마크 등록: `libero`(Table 1), `real_world`(Table 4). LIBERO-Cross는 트래커 표준 벤치마크가 아니라 eval_conditions 주석으로만 기록.

<!-- VERIFIED: pdf -->
