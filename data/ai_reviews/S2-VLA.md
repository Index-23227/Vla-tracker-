# S²-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation

> arXiv 2606.27872 · East China Normal University / Shanghai Jiao Tong University · 2026-06-26

## 1. 한 줄 요약

GRU 기반 belief state가 태스크 진행 단계를 추적하며 시각·의도·액션 세 attention 경로의 융합 비율을 동적으로 게이팅(SSGAA)하는 2B VLA로, LIBERO 평균 98.2%(Long 96.4%)와 SimplerEnv-Bridge 78.1%로 7B급 모델들을 능가하는 SOTA를 달성.

## 2. 문제 정의

- 기존 VLA는 고정 가중치로 시각·언어·액션 표현을 융합하는 **정적 융합(static fusion)** 구조라, 정밀 위치잡기 단계(시각 중요)와 계획 단계(의미 의도 중요)에 동일한 융합 비율을 적용.
- 이로 인한 초기 단계의 결정 편향이 long-horizon 태스크에서 액션 체인을 따라 전파·증폭되어 누적 오류로 태스크 실패를 유발.
- 각 액션이 정적 스냅샷에서 생성되어 실행 이력과 시간적 일관성을 반영할 명시적 메커니즘이 부재.

## 3. 핵심 기여

1. 대규모 액션 시퀀스 사전학습 없이, 사전학습 VLM(Qwen3-VL)에서 belief-state-space 인터페이스를 통해 직접 로봇 정책을 생성하는 S²-VLA 프레임워크.
2. 태스크 진행을 동적 belief state로 모델링하고, 이를 게이팅 네트워크로 시각·언어·액션 표현 융합에 반영하는 **State-Space Guided Adaptive Attention (SSGAA)** 메커니즘.
3. 2B 경량 아키텍처로 LIBERO·SimplerEnv·실세계 ALOHA에서 7B급 기존 VLA를 능가하는 long-horizon 성능 입증.

## 4. 아키텍처

- **백본**: Qwen3-VL (2B). ViT 시각 특징 F_t + 언어 토큰 + 학습 가능한 intent 토큰을 하나의 시퀀스 X로 결합, H개 Transformer 레이어 통과.
- **Belief state**: 경량 GRU f_φ가 최근 K스텝 액션 A_{t−K:t−1}과 proprioception P_t로 은닉 상태를 재귀 갱신, b_t = W_b·o_t + β_b. 별도 감독 없이 액션 예측 loss만으로 end-to-end 학습되며 태스크 단계·실행 편차 정보를 자연스럽게 인코딩.
- **SSGAA 액션 헤드 (24-layer)**: 세 병렬 경로 — (i) 액션 시퀀스 self-attention(시간적 일관성, zero-init A^(0) ∈ R^{T×K×d}), (ii) 저수준 시각 cross-attention(시각 토큰 은닉 상태 C_vis에서 공간 세부 유지), (iii) 고수준 intent cross-attention(intent 토큰 은닉 상태 C_ite에서 태스크 의미). belief state가 MLP+Softmax로 게이트 (g_vis, g_ite, g_act)를 산출해 융합 (Eq. 7-8).
- **액션 디코딩**: 마지막 SSGAA 레이어 출력에 LayerNorm + 선형 사영으로 K-step 액션 청크를 병렬 예측 (Eq. 9).

## 5. Action Head Category

**regression** — 순수 L2(MSE) 액션 예측 loss(Eq. 10)로 연속 액션 청크를 직접 회귀하는 병렬 디코딩 헤드. diffusion/flow-matching 미사용 (결론에서 향후 확장으로만 언급).

## 6. 학습 & 추론

- **학습**: 정규화 항 없는 순수 액션 예측 MSE loss로 Qwen3-VL 백본·GRU belief 모듈·SSGAA를 전부 end-to-end 공동 최적화. H100 4장.
- **추론**: 시간 재귀 루프 — GRU 상태 갱신 → 적응 게이트 산출 → 융합 표현으로 다음 액션 예측·실행. 7GB VRAM만으로 배포 가능, throughput **80.8 Hz** (OpenVLA-OFT 71.4, CronusVLA 8.7, TraceVLA 4.3 대비, Table 5).

## 7. LIBERO 결과 (Table 1)

- Spatial **98.4** / Object **99.6** / Goal **98.4** / Long **96.4** / **Avg 98.2** — 전 서브셋 SOTA.
- 2B로 OpenVLA-OFT 7B(97.1), CronusVLA 7B(97.0), MemoryVLA 7B(96.7), π₀ 3B(94.2)를 상회. 특히 Long에서 96.4로 OpenVLA-OFT(94.5), CronusVLA(94.0) 대비 우위 — 오류 누적 완화 효과.

## 8. SimplerEnv-Bridge 결과 (Table 2)

- WidowX 4개 태스크 평균 **78.1%**: Spoon-on-Towel 83.3, Carrot-on-Plate 87.5, Stack-Cube 41.7, Eggplant-in-Basket 100.0.
- MemoryVLA(71.9), π₀-Beta(68.4), CogACT-Large(57.3)를 능가하는 SOTA.

## 9. 실세계 ALOHA 결과

- ALOHA 이중팔 모바일 플랫폼(30Hz, 멀티뷰+손목 카메라)에서 pick-and-place, stacking, 책상 정리, 젓가락 전달(양팔 handover) 4개 태스크 평가.
- ACT, π₀-FAST 대비 전 태스크 우수한 성공률(Fig. 3) — SSGAA의 실세계 전이 능력 입증. 정확한 수치는 그림으로만 제시.

## 10. Ablation 결과 (Table 3, 4)

- **게이팅 위치**: w/o Gate 95.0 → Gate@All 94.4(악화), Gate@12 **96.4**(+1.4 최고). 중간 레이어(6/12/18)만 유효, 다층 조합(6,12,18)은 94.4로 오히려 불안정.
- **Belief 가이드**: Gate@12에서 b_t를 학습 상수 벡터로 대체(w/o State)하면 95.8 — 이득은 게이팅 자체가 아니라 belief state에 의한 **동적** 변조에서 옴.
- **백본 통제 비교 (Table 4, LIBERO-Long)**: 동일 QwenVL 계열에 FAST/OFT/PI/GR00T 헤드 이식 시 88.4~93.8에 그침. Qwen3-VL+Adapter-Pro 2B 94.4 대비 S²-VLA 96.4 — 아키텍처 설계 자체의 기여 입증.
- **게이팅 해석 (Fig. 4)**: 정밀 위치잡기 구간에서 g_visual 상승, 서브태스크 전환점에서 g_intent 피크, 정속 이동 구간에서 g_action 지배 — 단계 적응적 융합이 실제로 발현.

## 11. 한계 & 토의

- 액션 생성이 단순 L2 회귀라 멀티모달 액션 분포 모델링에 한계 — 저자들도 diffusion/flow matching으로의 SSGAA 확장을 향후 과제로 명시.
- 실세계 성공률이 그림으로만 제시되어 정량 비교 곤란, 코드 공개 여부 미언급.
- 게이팅 최적 레이어(12)가 실험적으로 결정된 값이라 다른 백본/스케일에서의 일반성 미검증. CALVIN 등 다른 표준 벤치마크 부재.

## 12. VLA-Tracker 관점에서의 의의

- LIBERO 평균 98.2로 리더보드 최상위권 진입 — 특히 2B라는 크기 대비 효율이 두드러짐(7GB VRAM, 80.8Hz).
- belief state 기반 동적 게이팅은 MemoryVLA(메모리 뱅크)·CronusVLA(멀티프레임)와 다른 제3의 시간 문맥 접근으로, "phase-aware fusion" 계열의 대표 사례.
- action_head_category=regression인 모델 중 최고 수준의 LIBERO 성적으로, diffusion 헤드 없이도 long-horizon SOTA가 가능함을 보여주는 데이터 포인트.

<!-- VERIFIED: pdf -->
