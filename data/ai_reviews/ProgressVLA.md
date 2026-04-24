# ProgressVLA: Progress-Guided Diffusion Policy for Vision-Language Robotic Manipulation

## 1. 배경 및 동기

기존 VLA는 대부분 **progress awareness가 없어** hand-crafted 종료 heuristic에 의존하고, long-horizon cascaded sub-goal 태스크에서 갈 길을 잃는다는 문제 의식에서 출발한다(Sec. I). Diffusion policy는 multi-modal trajectory를 잘 생성하지만 sampling이 conditioning에만 이끌리지, "지금 얼마만큼 목표에 가까워졌는가"라는 dense signal이 없다. 저자는 Table I에서 **progress-guided sampling을 적용한 diffusion policy가 Pearson r=0.722→0.934, steps 90.4→77.3, success 92.7→93.6**으로 개선되는 것을 motivating experiment로 보이며, "progress estimator를 inverse-dynamics world model을 통해 diffusion sampling에 classifier guidance로 주입하자"는 아이디어를 제안한다.

## 2. 방법론

세 개 컴포넌트 + RL 파인튜닝(Sec. IV, Fig. 2).

- **Progress Estimator $P$ (Sec. IV-A)**: DINOv2 patch feature를 사용해 $p = P(l, o_0, o_t) \in [0,1]$을 출력. Ground-truth는 정규화된 timestep $p^* = t/T$이며 L1 loss $\mathcal{L}_\text{prog}=|p-p^*|$로 훈련(Eq. 2). expert demonstration이 단조롭게 진행된다는 가정을 활용.
- **Action-Dynamics-Oriented World Model (Sec. IV-B, Fig. 3)**: UniVLA 구조를 차용. Encoder(inverse dynamics) $a_z = E(o_t, o_{t+N})$, Decoder(forward dynamics) $o_{t+N} = D(o_t, a_z)$. 목표는 $\mathcal{L}_\text{world} = \sum_t \|o_{t+N} - o^*_{t+N}\|^2 + \text{KL}(a_z, \mathcal{N}(0,I))$로 latent action 분포를 정규화(Eq. 5).
- **Joint Fine-tuning (Sec. IV-C)**: $p_{t+N} = P(l, o_0, D(o_t, a^z_{t:t+N}))$, $\mathcal{L}_\text{joint} = \|p_{t+N}-p^*_{t+N}\|$. 총 loss $\mathcal{L}_\text{ft} = \mathcal{L}_\text{world}+\mathcal{L}_\text{prog}+\mathcal{L}_\text{joint}$(Eq. 8).
- **Progress-Guided Diffusion (Sec. IV-D)**: Latent Action Expert는 **DiTA-style causal Transformer**로 latent action chunk $a^z_{t:t+N}$을 생성, Action Decoder가 이를 low-level action으로 디코딩. 역전파 가능한 classifier guidance: $x_{\tau-1} = \mu_\theta(x_\tau,\tau,c) + s\nabla_{x_\tau}\hat p_{t+N} + \sigma_\tau \epsilon$(Eq. 11). 종료는 progress threshold로 판정.
- **RL Fine-tuning (Sec. IV-E, Fig. 4)**: Online buffer에서 성공 에피소드의 단조성 위반을 $\mathcal{L}_\text{mono}=\sum_{t\in I_\text{anom}}\max(0, \epsilon-(\hat p_{t'}-\hat p_t))$로 교정(Eq. 14–15). Policy는 KL-constrained improvement $\pi^*(a|s) \propto \pi_\theta(a|s)\exp(\tfrac{1}{\alpha}Q(s,a))$(Eq. 18)로 업데이트.
- **Compute**: batch 2048, 8×H20 GPU(256/GPU), lr $1\times 10^{-4}$(Sec. V-A). OXE pretraining + LIBERO/CALVIN fine-tuning + CALVIN A/B/C online RL.

## 3. 실험 결과

**Table II (CALVIN ABC→D, S-RGB)**: Pretrained ProgressVLA (Full)이 1/2/3/4/5-in-a-row 각각 **95.2 / 84.8 / 73.6 / 67.2 / 52.0**, 평균 길이 **3.73**. GR-MG(S-RGBD+G-RGBD+P, 4.04)에는 못 미치나 S-RGB 단일 입력에서는 Dita(3.61), GHIL-Glue(3.69)를 상회. Classifier guidance 제거(w/o cg)는 3.57, pretrained evaluator 없이 scratch CG만 주면 3.61로, **"reliable evaluator → CG gradient 품질"이라는 저자의 가설을 확인**.

**Table III (LIBERO)**: Spatial 88.2 / Object 96.4 / Goal 87.2 / Long 66.2, 평균 **84.5** (Full). w/o cg 81.5 → w/ cg 83.3 → Full 84.5로 단계별 증가. LIBERO-Long에서 OpenVLA(53.7) 대비 +12.5%p로 long-horizon 효과가 두드러짐.

**Table IV (Real-robot, ARX AC-One dual-arm, 5 tasks × 20 trials)**: Octo 23% / 1.30m / 187.9 step → ProgressVLA w/o CG 66% / 0.96m / 100.8 step → w/ CG **76% / 0.81m / 53.3 step**. Pick peach 35→90, Open drawer 20→90으로 CG가 효율성과 신뢰도를 모두 끌어올림.

**Table V (Progress estimator generalization)**: Lighting shift에서 from-scratch는 Pearson 0.809/Stop 3.6(붕괴)이지만 pretrained+finetune는 **0.953/80.8**. Novel objects도 0.810→**0.972**. **OXE pretraining이 progress signal의 외란 robust성에 결정적**임을 증명.

**Table I (Motivating exp.)**: progress guidance on/off 차이로 Pearson 0.722→0.934 확보.

## 4. 한계 및 미해결 문제

- 진행도 라벨을 "정규화된 timestep $t/T$"로 단순화했기에, expert demo가 일부 구간에서 비단조(예: 재시도/후퇴)일 경우 편향이 발생. RL 단계의 $\mathcal{L}_\text{mono}$가 이를 사후 교정하지만 근본적 라벨 노이즈는 잔존.
- Parameters가 YAML에 "TODO"로 남아 있을 정도로 규모 정보가 미공개(본 논문도 구체 파라미터 수 미기재).
- Classifier guidance를 위해 diffusion step마다 world model + progress estimator의 forward + backward가 필요해, per-step 비용이 크다(real-robot에서 step 수가 줄어도 per-step latency는 측정 미공개).
- CALVIN 성능은 S-RGBD 계열(GR-MG 4.04)에 비해 여전히 아래 — depth 정보 활용 미비.
- Progress-guided termination threshold의 민감도 분석이 부족.

## 5. 총평

- **Novelty**: ★★★☆☆ — "normalized timestep을 progress proxy로 쓰고, inverse-dynamics world model을 거쳐 diffusion sampling에 gradient를 역전파"하는 조합은 의미 있는 integration이지만, 진행도 예측/분류기 guidance/world model/RL 파인튜닝 각 요소는 독립적으로 존재해 왔음.
- **Practical impact**: ★★★★☆ — Real-robot 5 tasks에서 success 23→76%, steps 187.9→53.3의 폭은 실전적. 특히 executable path length가 60% 감소한다는 건 에너지·시간 제약이 있는 deployment에서 직접적 가치.

## 6. 예상 질문

**Q1. progress label을 $t/T$로 정의하는 것은 위험하지 않은가?**
A. Sec. IV-A의 가정: expert demo가 "approximately monotonic"하다는 큐레이션 전제에 기댄다. 비단조성은 Sec. IV-E의 $\mathcal{L}_\text{mono}$(Eq. 15)로 online buffer에서 사후 교정하는 구조. 완벽한 해결책은 아니지만, 수작업 progress annotation 부재에서의 현실적 타협.

**Q2. Classifier guidance의 gradient는 어떻게 구체적으로 흐르는가?**
A. $\hat p_{t+N} = P(l, z_0, D(o_t, x_\tau))$가 $x_\tau$에 대해 미분 가능(world model decoder $D$를 통과)이므로 $\nabla_{x_\tau}\hat p_{t+N}$을 얻어(Eq. 10–11), diffusion reverse step의 평균에 strength $s$를 곱해 더한다.

**Q3. Pretrained evaluator가 왜 scratch evaluator보다 나은가?**
A. Table V에서 pretrained는 lighting shift Pearson 0.953 vs scratch 0.809로, distribution shift 하에서 progress signal이 훨씬 안정적. noisy evaluator는 gradient noise를 유발해 CG sampling을 교란함(Sec. V-C).

**Q4. ABC→D zero-shot에서 GR-MG 같은 RGBD 모델을 왜 못 이기는가?**
A. Table II의 입력 비교: GR-MG는 S-RGBD+G-RGBD+P 풀세트. ProgressVLA는 **S-RGB 단일**이므로 이 handicap을 감안하면 Dita/SuSIE 대비 우위가 오히려 공정한 비교. Depth 통합은 future work.

<!-- VERIFIED: pdf -->
