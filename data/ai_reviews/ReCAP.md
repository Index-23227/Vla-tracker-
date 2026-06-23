# ReCAP: Retrieve, Don't Retrain — Extending Vision-Language-Action Models to New Tasks at Test Time

> **한 줄 요약**: VLA 정책을 새 task에 적응시킬 때 매번 teleoperation + per-task fine-tuning(~24 GPU-hours/task)이 필요하던 비용을, **retrieval pool에 값싼 pool-embodiment 데모(예: 사람 손 비디오)를 추가하는 것**만으로 대체. Cosmos Policy(WAM) 위에 retrieval-conditioned residual action parameterization을 두고 한 번 학습 후 동결(frozen)하면, 새로운 task는 pool indexing만으로 흡수된다. PushT cross-embodiment에서 34.9% vs baseline 6.0%, RoboTwin 2.0 unseen 5종에서 31.5% vs 26.0% (강한 baseline 대비), real-robot에서 close-cabinet/place-bottle 두 unseen task가 baseline 0%/10%에서 30%/80%로 상승.

---

## 1. 배경 및 동기

### 기존 연구의 구조적 한계
- 범용 VLA (OpenVLA, π0.5, GR00T N1.6, Cosmos Policy 등)는 새 embodiment·task마다 **teleop 데이터 + per-task fine-tuning**을 요구
- 데이터 측 비용: teleop이 사람 손 데모 대비 **약 18배 느림**
- 컴퓨트 측 비용: 최신 world-action model(WAM) 한 task fine-tuning에 **~24 GPU-hours**
- 두 비용이 task마다 누적되어 확장성이 무너진다

### 핵심 질문
- 새 task 적응을 **파라미터 업데이트 없이 retrieval로 대체**할 수 있는가?
- 사람 손 비디오 같이 **cheap embodiment 데모**가 robot teleop을 일부 대체할 수 있는가?
- WAM의 future-image objective는 retrieval과 어떻게 시너지를 내는가?

📌 [Figure 1 삽입] — 상단: per-task fine-tune (24 GPU-hours × N tasks). 하단: ReCAP은 frozen policy + retrieval pool 확장만으로 새 task 흡수.

---

## 2. 방법론 심층 분석

### 2.1 시스템 개요

세 가지 핵심 구성:
1. **WAM 백본 (Cosmos Policy)**: action latent + future image latent를 동일한 video sequence로 denoising
2. **Retrieval Conditioning**: 매 control step에서 pool에서 (state, action) chunk를 검색, clean latent로 prepend
3. **Residual Action Parameterization**: action latent가 pool action 위의 잔차(residual)만 학습

### 2.2 Backbone: Cosmos Policy (WAM)

- Video generation 기반 정책. action + future observation을 하나의 denoised video로 출력
- 일반 VLA(action-only head)와 달리 **L_state**(future-image flow-matching loss)가 존재
- 언어 명령은 cross-attention으로 주입

### 2.3 Retrieval-Conditioned Input (식 1)

$$\pi_\theta(s_t^{query},\; s_{t':t'+H}^{pool},\; a_{t':t'+H}^{pool}) \mapsto \hat{a}_{t:t+H}^{query},\; \hat{s}_{t+H}^{query}$$

- 검색된 pool chunk는 **clean latent**로 인코딩되어 temporal axis에 prepend
- query side action·observation은 noise에서 denoising
- 아키텍처 수정 없음 — 기존 I2V conditioning(single clean frame)을 **clean state-action sub-sequence**로 확장

### 2.4 Residual Action Parameterization (식 3)

$$\hat{a}_{t:t+H}^{query} = a_{t':t'+H}^{pool} + \Delta a_{t:t+H}$$

- Pool action이 이미 coarse motion을 제공 → 정책은 **embodiment gap만큼의 차이**만 학습
- 이 차이는 action label로 잘 안 드러나지만 픽셀(접촉 방식, gripper closing)에서는 명확 → state prediction이 dense visual signal 제공

### 2.5 Joint Flow-Matching Loss (식 2)

$$L(\theta) = \lambda \cdot L_{act}(\hat{a}^{query}, a^{query}) + L_{state}(\hat{s}^{query}_{t+H}, s^{query}_{t+H})$$

- Action과 future-image를 **하나의 flow-matching**으로 joint supervise
- Standard VLA는 L_state가 없어 retrieval의 visual consistency 신호를 못 받음 → 본 논문이 강조하는 retrieval × WAM 시너지의 근거

### 2.6 Retrieval Mechanism (§4.2)

- 후보 set $C_t^{traj}$: composite initial-frame descriptor $\psi_0$ (언어 embedding + SAM 3 object positions + 초기 proprioception)으로 top-K trajectory 선택
- Index distance d: object pose / proprioception / upcoming action chunk의 weighted L2 + DINOv3 image feature cosine distance
- **학습 시에만** upcoming action chunk를 distance에 포함, inference 시 drop
- Inference 시 새 pool $D_{test}^{pool}$이 active pool을 replace, 매 step re-index

📌 [Figure 2 삽입] — ReCAP framework: 현재 obs → retrieval → (state, action) chunk + 현재 obs를 WAM에 conditioning → next action + next obs denoising

---

## 3. 실험 설정

### 3.1 PushT Cross-Embodiment
- 2D PushT: T-shape block을 goal pose로 push
- Target = **triangle**, Pool = **disc** (다른 contact dynamics)
- Train: ±45°에서 100 paired demos
- Test: -60°~+60°까지 15° 간격 9개 angle, 그 중 **7개가 unseen**
- Test-time pool: 5° 해상도 disc 데모

### 3.2 RoboTwin 2.0 Simulation
- Target = **Aloha-Agilex** (dual-arm), Pool = **UR5**
- Train: 5 paired tasks (PCB, OM, DB, MP, GR)
- Test (unseen): 5 tasks (MPP, PBS, CB, HM, LP)
- Pool progression: 11 → 17 → 23 → 29 → 35 tasks (strict superset)

### 3.3 Real Robot
- Pool = **사람 손 비디오** (VR wrist tracking)
- Target = teleop robot
- Train: open-cabinet 단일 task만
- Test (unseen): close-cabinet, place-bottle

---

## 4. 핵심 결과

### 4.1 RoboTwin 2.0 (Table 1)

| Method | Seen Avg | Unseen Avg |
|---|---|---|
| Baseline (Cosmos Policy) | 32.5 | 4.0 |
| Retrieval Only | 25.5 | 26.0 |
| Co-training (EgoBridge/STRAP 류) | 27.0 | 10.0 |
| **ReCAP (Ours)** | **43.5** | **31.5** |

- 가장 강한 baseline 대비 **seen +11.0pt, unseen +5.5pt** (강 baseline은 co-training이 아닌 retrieval-only)
- Co-training은 seen에서 baseline보다도 떨어짐 — pool 데이터를 joint train할 때 embodiment confusion 발생
- Retrieval Only는 가까운 pool trajectory가 이미 target에 근접한 task에서만 경쟁력, 그 외엔 residual learning이 격차를 메움

### 4.2 PushT Cross-Embodiment (Fig. 5a, Table 2)

| Backbone | No Retrieval | + Retrieval |
|---|---|---|
| π0.5 | 6.6% | 25.1% |
| **Cosmos (WAM)** | 6.0% | **34.9%** |

- Retrieval은 두 backbone 모두 향상시키지만 **WAM에서 이득이 더 큼**
- L_state 제거 시 27.4%로 하락 → **future-image objective가 retrieval과 paired될 때만 정보를 갖는다**는 주장의 증거

### 4.3 Test-Time Pool Progression (Fig. 7)

- RoboTwin unseen avg가 pool size에 따라 **monotonic** 상승: 11(9.0) → 17(18.5) → 23(19.5) → 29(22.0) → 35(31.5)
- 모든 5개 held-out task가 pool에 포함된 시점에 supervised unseen-task avg와 매칭
- **파라미터 업데이트 없이** indexing만으로 coverage 확장 가능 입증

### 4.4 Real Robot (Fig. 8b)

| Task | Baseline | ReCAP |
|---|---|---|
| Open cabinet (trained) | 90% | 90% |
| Place bottle (unseen) | 10% | **80%** |
| Close cabinet (unseen) | 0% | **30%** |

- Baseline은 unseen task에서 학습된 open-cabinet 동작으로 collapse
- ReCAP은 frozen policy가 사람 손 trajectory를 따라 unseen task를 수행 — **embodiment gap이 큰 데도 작동**

---

## 5. 비교 — 기존 retrieval/test-time-adaptation 접근

| 차원 | EgoBridge / STRAP (co-training) | Retrieve-then-Steer | **ReCAP** |
|---|---|---|---|
| Frozen at test time | ✗ (joint train) | ✓ | ✓ |
| New task 추가 비용 | 재학습 | 없음 (online memory) | **없음 (pool append)** |
| Backbone | 일반 VLA | π0/π0.5/CogACT (생성형) | **Cosmos WAM** |
| Visual consistency | ✗ | ✗ | ✓ (L_state) |
| Pool embodiment | target과 동일 | self success memory | **다른 embodiment (사람 손 / UR5)** |
| Action 표현 | direct | sampler guidance | **residual on pool action** |

---

## 6. 강점

1. **확장성**: 새 task당 GPU-hours 비용이 0 (단순 데모 추가). 24h × N → 0 × N.
2. **데이터 비용 18× 절감**: 사람 손 비디오가 teleop을 부분적으로 대체.
3. **명확한 ablation**:
   - π0.5 vs Cosmos backbone 비교 → WAM이 retrieval과 더 시너지
   - L_state ablation → image objective의 retrieval-dependent 효과 입증
   - Pool progression → 파라미터 업데이트 없이 monotonic coverage 확장
4. **Residual parameterization**의 ML적 정당성: embodiment gap은 action label에서 약하고 픽셀에서 강함, 따라서 state prediction이 supervision 보완.

---

## 7. 약점·한계

1. **Pool curation 비용 미정량화**: 사람 손 데모 수집·정렬·indexing pipeline의 실제 비용은 18× 단순 비교만으로 부족.
2. **Retrieval 실패 시 catastrophic**: pool에 적절한 chunk가 없을 때 정책 거동은 표면적으로만 다뤄짐.
3. **Cosmos Policy 의존성**: WAM이 없는 경량 VLA에서는 L_state 이득을 못 받음. π0.5 결과(25.1%)는 ReCAP 풀 버전(34.9%)에 비해 ~10pt 낮음.
4. **Pool indexing 계산 비용**: control step마다 SAM 3 + DINOv3 + L2 검색. inference latency는 명시되지 않음.
5. **RoboTwin task 셋이 작음** (seen 5 + unseen 5). MPP 등 unseen task에서 5.0%만 달성하여 강한 일반화는 아직 부족.
6. **Embodiment gap 한계**: 사람 손 ↔ robot처럼 gripper geometry가 다른 케이스에 대한 systematic 분석 부족.

---

## 8. ML 관점에서의 함의

- **RAG의 로보틱스 외삽**: LLM RAG가 파라미터 대신 외부 store로 지식을 확장하듯, 로봇 정책도 동일 패러다임 가능함을 실증.
- **Residual learning + retrieval**: ICL/RAG 문헌의 retrieval-augmented prediction을 continuous control + cross-embodiment에 일반화.
- **World model과 retrieval의 결합**: future-image prediction이 단순 보조 loss가 아니라 retrieval의 visual consistency check 역할로 재해석됨. WAM이 retrieval에 필수적인 미래 정책 방향성을 시사.
- **Data-centric robotics**: 모델 크기보다 pool 품질·다양성이 결정적이라는 관점 강화.

---

## 9. 후속 연구 방향

1. **Pool 자동 큐레이션**: Active learning으로 unseen task에 가장 유익한 pool 데모를 식별/수집
2. **Retrieval 실패 검출**: confidence-aware retrieval, fallback policy
3. **Multi-modal pool**: 사람 손·다른 robot·시뮬레이션 데이터를 통합 indexing
4. **Hierarchical retrieval**: trajectory level → chunk level → frame level의 multi-scale 검색
5. **경량 backbone에서의 적용**: WAM 없이 visual consistency signal을 얻는 대안 objective 탐색
6. **Long-horizon composition**: 여러 pool chunk를 조합하여 합성 task 수행

---

## 10. 재현·실용 노트

- 코드는 project page(`recap-robot.github.io`)에서 공개 의도, 본 PDF 시점 기준 별도 코드 release 명시 없음 (open_source: false로 표기)
- 학습 데이터: PushT 100 paired demos / RoboTwin 5 paired tasks / real-robot single task — **상대적으로 적은 paired 데이터**
- 의존성: Cosmos Policy backbone, SAM 3, DINOv3
- Inference 비용: control step당 retrieval re-run, sub-sequence conditioning으로 인한 추가 토큰

---

## 11. 결론

ReCAP은 "**VLA 확장 = retrieval pool 확장**"이라는 새로운 패러다임을 WAM 기반에서 실증한다. Cosmos Policy + residual action + joint flow-matching loss의 조합으로, **frozen policy가 사람 손 비디오 같은 cheap embodiment 데모를 indexing**하여 unseen task를 흡수한다. PushT(34.9% vs 6.0%), RoboTwin 2.0(unseen 31.5% vs 26.0%), real-robot(place-bottle 80% vs 10%) 모두에서 일관된 향상을 보였고, pool progression 실험으로 파라미터 업데이트 없는 capability 확장을 입증했다. 한계로는 WAM backbone 의존성, retrieval 실패 분석 부족, embodiment gap 일반화 미완 등이 남는다.

---

## 12. 핵심 인용

> "...our retrieval augmented policy is trained once on paired demonstrations from the target embodiment (query) and a cheaper embodiment (pool, e.g., human-hand video), then frozen. New tasks are added at deployment by appending pool-side demonstrations to a retrieval pool."

> "We parameterize the action latents as a residual over retrieved trajectories: retrieval supplies the coarse high-level motion and task progression, while the policy learns only the embodiment-specific dynamics needed to execute the behavior on the target robot."

> "Cheap pool-embodiment data at deployment can therefore stand in for new target-embodiment demonstrations on tasks unseen during fine-tuning."

---

<!-- VERIFIED: pdf -->
