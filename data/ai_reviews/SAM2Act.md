# SAM2Act: Integrating Visual Foundation Model with a Memory Architecture for Robotic Manipulation

**Fang, Grotz, Pumacay, Wang, Fox, Krishna, Duan, 2025 (arXiv:2501.18564v4, U. Washington / NVIDIA / AI2 / UC San Pablo)**

> 작성 시점에서 venue 정보가 명확히 확인되지 않아 venue 표기는 생략. 본 리뷰는 arXiv:2501.18564v4 기준.

## 한 줄 요약
RVT-2의 multi-view coarse-to-fine transformer 위에 SAM2 image encoder + multi-resolution upsampling을 결합하여 RLBench 18 task 평균 86.8%를 달성하고, SAM2의 memory bank/encoder/attention을 차용한 SAM2Act+로 spatial memory 의존 태스크(MemoryBench) 평균 94.3%를 기록한 keyframe-based 3D manipulation policy.

## 핵심 기여
- SAM2의 multi-resolution embedding을 cascade convex upsampler로 통합해 keyframe pose 예측의 정밀도 향상.
- MemoryBench라는 비-Markov spatial memory 평가 벤치마크를 새로 제안 (reopen_drawer, put_block_back, rearrange_block).
- SAM2의 video tracking용 메모리 메커니즘(Memory Bank/Encoder/Attention)을 manipulation policy로 변환 — 기존 BC 방법이 풀지 못한 시퀀스 의존 작업을 해결.

## 배경
3D manipulation policy 계열(PerAct voxel, RVT 다중뷰, RVT-2 coarse-to-fine)은 keyframe pose 예측을 통해 정밀 조작에서 진전을 보여왔지만, 저자들은 (i) 환경 perturbation에 대한 일반화, (ii) 비-Markov 시퀀스 의존 태스크에서의 spatial memory 두 축이 미흡하다고 지적한다 (§1). SAM-E[26] 등이 SAM 인코더의 object-centric feature를 도입했으나 단일 해상도이고 메모리 메커니즘은 없다. SAM2의 memory attention/bank/encoder는 비디오 객체 추적에서 검증된 구조이므로, 저자들은 이를 manipulation policy의 heatmap·관측 임베딩 회상 메커니즘으로 재해석한다 (§1, §4.2). MemoryBench는 RLBench의 Markov 가정을 의도적으로 위반하여 두 action history가 같은 ot로 수렴하지만 이후 다른 행동을 요구하도록 설계된 새 벤치마크다 (§3.1, P(o_{t+1}|o_t,a_t) 식).

## 방법론
- **Backbone (Fig. 3, Fig. 4)**: 입력 RGB-D 5뷰(128×128, front/left/right/overhead/wrist)로 point cloud 재구성 후 3개의 가상 뷰로 렌더링. 각 가상 뷰의 RGB를 복제해 SAM2 image encoder에 통과시키고, LoRA(rank 16) fine-tuning으로 도메인 적응. SAM2가 3 해상도의 multi-resolution embedding을 제공한다 (§4.1).
- **Multi-resolution upsampling (Fig. 4, 본문 식)**: cascade convex upsampler 3단으로 feature map을 점진적 2배씩 upsample. 각 단계에서 X^{l+1} = LayerNorm(U(X^l) ⊕ E^l)로 SAM2의 동일 해상도 embedding과 element-wise add. Coarse branch는 translation heatmap만, fine branch는 추가로 rotation/gripper도 예측.
- **Multi-View Transformer**: CLIP text encoder로 언어 토큰화, virtual image patchify, SAM2 lowest-resolution embedding을 token으로 변환 후 4 layer self-attn(view 별) + 4 layer joint attn (Appendix A).
- **SAM2Act+ (Algorithm 1, §4.2)**: coarse branch 위에 Memory Encoder, Memory Attention(self+cross attn 4 layer, RoPE 2D, dim 128), Memory Bank(view 별 FIFO queue, max M memory, dim 64) 추가. 예측된 translation heatmap을 mask analog로 사용해 MVT의 unconditioned embedding과 합성하여 memory feature 생성. Stage 1에서 SAM2Act 사전학습 후 image encoder/MVT/fine branch를 freeze하고 coarse branch만 메모리와 함께 fine-tune (§4.2 Training).

## 실험 결과
### 시뮬레이션
- **RLBench 18 tasks (Table 1)**: SAM2Act 평균 86.8% ± 0.5, RVT-2(81.4%)·SAM-E(70.6%)·RVT(62.9%)·PerAct(49.4%)를 모두 능가. 9/18 task에서 1위. 가장 큰 개선은 Insert Peg(40→84, +44pp, 약 2.1×)와 Sort Shape(35→64, +29pp). Stack Cups 78%, Place Cups 47%로 RVT-2 대비 향상. Close Jar 99.0, Push Buttons 100.0, Sweep to Dustpan 99.0으로 거의 포화.
- **The Colosseum 일반화 (Table 2)**: 13개 perturbation 평균 -4.3% ± 3.6%로 RVT-2(-19.5%) 대비 압도적. Light Color +4.5%, Table Color +1.1%, Distractor +1.7% 등 일부 perturbation에서는 오히려 성능 향상. RO-Texture에서는 +24.7%로 학습 시보다 더 높은 성공률을 보이는 흥미로운 현상. All Perturbations 동시 적용 시 -58.3% (RVT-2 -77.9%).
- **Ablation (Table 2 일부)**: SAM2 → SAM 교체 시 -20.7%로 회귀해 video pretraining의 가치를 입증. Multi-res input 제거 시 -19.1%로 회귀 — 두 설계 모두 robustness의 핵심.
- **MemoryBench (Table 3)**: 3 task 평균 SAM2Act+ 94.3% ± 9.0 (Reopen Drawer 84.0, Put Block Back 100.0, Rearrange Block 99.0). 메모리 없는 SAM2Act 55.0%, RVT-2 54.0% 대비 +39.3%pp 격차. Random agent 기댓값(33/25/50%)을 명백히 상회. SAM2Act 단독은 Rearrange Block에서만 82% 도달(다른 task는 30~48%)하여 명시적 메모리 없이는 비-Markov 추론 불가능함을 보여줌.

### 실세계
- **Real-robot (Table 4)**: Franka Panda + Robotiq 2F-85 + RealSense D455. 4 task에서 SAM2Act 75% vs RVT-2 43%. 고정밀 (a) lamp 점등 0/10 → 6/10 (RVT-2 ID/OOD 모두 0). Push button sequence 9/10 (in-dist) & 9/10 (OOD). Memory task (d) "push the same button"에서 SAM2Act+ in-distribution 7/10, OOD 6/10 (RVT-2 4/10, 2/10).

## 한계
- **연속 제어 미지원**: Keyframe-based pose 예측 정책으로 dexterous continuous control(예: 손가락 제어)로의 확장이 제한됨 (§6).
- **고정 메모리 윈도우**: Memory window 길이가 task별로 고정되어 가변 길이 시퀀스에 약함 (§6).
- **Semantic 메모리 부재**: 저자들이 직접 실험한 결과 메모리에 spatial 정보는 잘 저장되지만 색상 등 semantic 정보 보존은 어려웠다고 명시 (§6).
- **계산 비용**: 32× H100/A100 GPU 사용으로 학습 비용이 크고 SAM2 encoder param이 정확히 공개되지 않음(YAML "not disclosed").
- **벤치마크 커버리지 협소**: LIBERO·CALVIN 평가가 없어 다른 VLA 계열과의 직접 비교가 제한적. RLBench·Colosseum·MemoryBench 모두 RVT-2 후속작 생태계 위주.

## 총평
RVT-2가 정의한 multi-view 2.5D 키프레임 회귀 패러다임에 (i) SAM2의 풍부한 멀티해상도 시각 표현, (ii) 메모리 모듈을 결합해 RLBench·Colosseum·MemoryBench 3축에서 SOTA를 달성한 단단한 후속작이다. 특히 Insert Peg에서 2× 가까운 개선과 Colosseum 환경 perturbation에서 -4.3%로 거의 robust한 점은 vision foundation model의 표현이 manipulation policy의 일반화에 직접 기여함을 보여주는 강한 증거다. MemoryBench라는 새 벤치마크 제안 자체도 비-Markov 의존 작업의 평가 방법론으로 가치가 크다.

설계상 RVT-2 위에 plug-in 형태로 SAM2 encoder + cascade upsampler + memory 모듈을 얹은 구조라 다른 multi-view 3D policy(예: 3D Diffuser Actor, Act3D)에도 이식 가능성이 높다. 이는 향후 SAM3, DINOv3 같은 더 강력한 vision foundation model로의 자연스러운 업그레이드 경로를 열어둔다는 점에서 미래지향적 설계라 평가할 수 있다.

## 예상 질문
1. **왜 SAM2 → SAM으로 바꾸면 성능이 크게 떨어지는가?**
   - Table 2에서 SAM2 → SAM 교체 시 Colosseum 평균 -20.7%로 RVT-2 수준으로 회귀.
   - SAM2의 video pretraining에서 얻은 시간적/멀티해상도 표현이 manipulation 일반화에 핵심이라는 해석. 단일 해상도 SAM 임베딩만으로는 cascade upsampler의 multi-scale fusion 효과를 누릴 수 없다.
2. **SAM2Act+의 메모리는 정확히 무엇을 저장하는가?**
   - 각 view별 FIFO queue에 (predicted heatmap을 conv downsampling한 mask analog) ⊕ (MVT의 unconditioned image embedding)으로 만든 spatial feature map을 저장 (§4.2, Algorithm 1).
   - SAM2 image encoder가 아닌 MVT 출력을 사용하는 이유: 언어·view·spatial을 융합한 표현이 행동 회상에 더 적합 (§4.2 Training).
   - 차원: memory feature는 64-dim, MVT 출력 128-dim과 align.
3. **RVT-2 대비 추가된 학습 비용은?**
   - LoRA(rank 16) fine-tune이라 SAM2 encoder는 거의 frozen에 가깝게 유지.
   - 32× H100/A100 학습 셋업이 기본이지만 16/8 GPU로 줄여도 동일 batch size로 공정 비교 (Appendix B).
4. **MemoryBench task가 비-Markov임을 어떻게 보장하는가?**
   - §3.1의 P(o_{t+1}|o_1,a_1,...,o_t,a_t) = P(o_{t+1}|o_t,a_t) Markov 가정을 의도적으로 위반.
   - 두 action history가 같은 o_t로 수렴하지만 이후 다른 행동이 필요하도록 task 디자인. 언어 instruction을 표준화해 우회 단서 차단.

<!-- VERIFIED: pdf -->
