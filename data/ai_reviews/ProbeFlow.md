# ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models

## 1. 배경 및 동기

Flow Matching(FM) action head가 VLA의 SOTA 성능을 견인하고 있지만, 50-step Euler 같은 고정 ODE solver가 실시간 로봇 제어의 주요 병목이라는 것이 논문의 핵심 관찰이다(Sec. I). Table I에서 보이듯 50-step Euler는 MetaWorld에서 **전체 328.7 ms 중 235.7 ms(71%)를 FM action head에서 소비**한다. 기존 VLA 가속 연구는 대부분 VLM backbone 측 quantization/token compression에 집중했지만, action head의 iterative ODE는 손대지 못한 사각지대였다. 저자는 FM 경로가 "gross transit 구간의 직선 + precise grasping 구간의 급격한 곡률"로 구성된다는 기하학적 특성을 활용해, **어떤 파라미터 업데이트도 없이** 동적으로 ODE step 수를 조절하는 training-free wrapper를 제안한다.

## 2. 방법론

베이스는 Evo-1 + frozen InternVL3-1B VLA에 탑재된 8-layer DiT, hidden 1024, horizon $T=50$ FM action head다(Sec. V-A). ProbeFlow는 이 위에 얹히는 추론 전용 스케줄러이다.

- **Lookahead Linearity Probe (Sec. IV-B, Fig. 2)**: 초기 속도 $v_\text{start} = v_\theta(x_0, 0, c)$를 계산한 뒤 큰 step $\Delta t_\text{probe}=0.5$로 탐색 상태 $x_\text{probe}$를 만들고 그 지점의 속도 $v_\text{probe}$와의 cosine similarity $S = v_\text{start}^\top v_\text{probe} / (\|v_\text{start}\|\|v_\text{probe}\|)$를 curvature proxy로 쓴다(Eq. 5–8). 로컬 truncation error가 $\|e_\text{trunc}\| \propto (\Delta t)^2 |dv_t/dt|$에 의해 곡률에 비례한다는 수치 해석 결과(Eq. 4)가 이론적 근거다.
- **Dynamic Step Scheduler (Sec. IV-C, Algorithm 1)**: $N = \text{clip}(N_\text{min} + \lfloor (1-S)/\epsilon \rfloor \cdot \Delta N, N_\text{min}, N_\text{max})$로 이산화(Eq. 9). $N = N_\text{min}$인 선형 구간에서는 $x_1 = x_\text{probe} + v_\text{probe} \cdot (1-\Delta t_\text{probe})$로 중간 적분을 전부 생략(Eq. 10). 곡률이 큰 구간에서는 dense Euler로 전환하되 $v_\text{start}$는 재활용.
- **Hyperparameters**: $N_\text{min}=2, N_\text{max}=10, \Delta N=2, \epsilon=0.008$을 MetaWorld/LIBERO 양쪽에 고정(Sec. V-A). 완전 training-free이므로 어떤 파라미터도 재학습되지 않음.

## 3. 실험 결과

**Table I (MetaWorld MT50)**: Fixed-Euler $N=50$이 82.5±1.2%/235.7 ms일 때, ProbeFlow는 평균 2.6 step으로 **83.2±1.8% / flow solver 15.9 ms (14.8× 가속)**, total 116.5 ms(end-to-end 2.8× 가속). 동일 compute budget의 $N=3$ fixed Euler는 72.4%로 붕괴(rigid schedule의 한계). RK45는 NFE 폭증으로 2923 ms, 63.0%로 실용 불가. AB2 65.6 ms / 78.8%로 저조.

**Table II (LIBERO)**: Fixed-Euler $N=50$ 92.5±1.1%/278.7 ms, ProbeFlow 평균 4.5 step / **88.7±1.5% / 32.7 ms (8.5× 가속)**. 흥미롭게도 LIBERO는 probability path가 더 휘어있어($N=50$에서도 평균 4.5 step 필요), 동일 $\epsilon$에서 MetaWorld보다 3.8%p 성능 손실 발생.

**Table III–IV (Sensitivity to $\epsilon$)**: MetaWorld는 $\epsilon=0.008$에서 83.2%/15.9 ms가 최적. LIBERO는 $\epsilon=0.002$로 조이면 평균 14.1 step / 92.0%(N=50 베이스라인 92.5%와 통계적 동률)로 회복되면서도 60% 이상 가속.

**Fig. 4 (Probe horizon ablation)**: $\Delta t_\text{probe} \le 0.4$는 근시안적이어서 success rate <5%로 catastrophic collapse, $\ge 0.6$은 over-sensitive해서 $N$이 20까지 급증, $\Delta t_\text{probe}=0.5$에서 83.2%로 sweet spot.

**Table V (실세계 "Pick-and-Place", UFACTORY xArm7 + Inspire hand)**: Fixed-Euler N=50은 270.3 ms / 8/10 success. ProbeFlow는 평균 2.1 step / **12.26 ms / 7/10**로 22× action-head 가속을 실기기에서 재현.

## 4. 한계 및 미해결 문제

- 저자 스스로 지적하듯 $\epsilon$은 domain-level hyperparameter이며 완전한 zero-shot drop-in은 아니다(Sec. V-E). MetaWorld 최적값을 LIBERO에 그대로 쓰면 -3.8%p 성능 손실.
- $\Delta t_\text{probe}$가 고정 상수이므로 극도로 변동이 심한 contact-rich 궤적에서는 적응 실패 가능성이 남아있다(Sec. VI, future work).
- Baseline이 Evo-1 + InternVL3-1B 한 가지 VLA에 국한되어 있으며, π0·GR00T·RDT 등 대형 flow-matching VLA에서의 검증은 미포함.
- Rectified Flow로 사전 학습된 policy처럼 이미 trajectory가 곧게 편 경우에는 이득이 줄어들 가능성(Sec. IV-A).

## 5. 총평

- **Novelty**: ★★★★☆ — "훈련 없이, geometry만으로 곡률을 한 번의 lookahead로 추정"한다는 발상은 놀라우리만치 단순하지만, AdaFlow(보조 variance estimator 훈련 필요)·consistency models·rectified flow 등과 뚜렷한 차별점을 만든다.
- **Practical impact**: ★★★★★ — 어떤 pre-trained FM VLA에도 그대로 얹을 수 있고, 실세계에서 flow solver 지연을 270→12 ms로 줄이는 효과는 즉시 배포 가치가 높다.

## 6. 예상 질문

**Q1. 단 한 번의 lookahead만으로 trajectory curvature를 정말 근사할 수 있는가?**
A. Eq. 4에서 truncation error가 $|dv_t/dt|$에 비례함을 활용한다. 시작점과 $\Delta t_\text{probe}=0.5$ 지점의 속도 cos 유사도는 해당 구간 평균 곡률의 단조 proxy로 기능하며, Fig. 3의 Basketball/Button Press 시각화가 이 가설을 지지한다.

**Q2. 왜 LIBERO에서 MetaWorld보다 step이 더 많이 필요한가?**
A. Sec. V-C의 분석에 따르면, LIBERO는 long-horizon semantic reasoning이 필요해 expert action distribution이 본질적으로 더 multi-modal이다. 따라서 학습된 FM trajectory도 더 휘어 있어($N=50$ baseline에서조차 평균 4.5 step 필요) 더 촘촘한 적분을 요구한다.

**Q3. RK45처럼 고전 adaptive solver 대비 이점은?**
A. RK45는 step마다 6 NFE를 요구해 MetaWorld에서 2923 ms로 폭발(Table I). ProbeFlow는 step당 최대 1 NFE만 추가되므로, 무거운 neural vector field에 본질적으로 더 유리하다(Sec. II-C).

**Q4. $N_\text{min}=2$로 잡은 근거는?**
A. Sec. V-E. 초기 evaluation $v_\text{start}$와 lookahead $v_\text{probe}$를 버리지 않고 재활용해야 probe의 computational cost를 완전 상각할 수 있으므로, 이 두 evaluation을 소비하려면 최소 $N=2$가 이론적 하한이다.

<!-- VERIFIED: pdf -->
