# Geo-VLA: Geometry-Aware Vision-Language-Action Planning via Internalization of Map Semantics

> **한 줄 요약**: 단일 전방 카메라 기반 driving VLA가 차선 연결성·도로 곡률·교차로 구조 같은 **정적 도로 기하/위상**을 잘 표현하지 못한다는 문제를 지적하고, 오프라인 지도 주석으로 만든 **Geo-QA**(3,000개)로 VLA backbone을 contrastive + QA instruction tuning(LoRA)으로 사전 적응시킨 뒤 원래 action decoder를 재학습하는 plug-and-play 프레임워크를 제안. 추론 시 HD map 불필요. NAVSIM v1에서 DynVLA 91.0 → **92.1 PDMS**, ReCogDrive 90.8 → 91.2.

---

## 1. 배경 및 동기

- 최근 driving VLA(ReCogDrive, DynVLA 등)는 foundation model의 semantic reasoning을 활용하지만, 도로 정보를 카메라 이미지로만 얻는다.
- 차선 경계·교차로 연결성·회전 제한 같은 구조적 지도 요소는 단일 전방 이미지의 외관으로 추론하기 어려워, 곡선로·교차로에서 궤적이 expert와 도로 기하에서 이탈(Figure 1).
- HD map encoder를 추론에 넣는 직접적 해법은 (1) 지도 수집·유지 비용, (2) localization·검색·인코딩 추가로 인한 복잡도/연산 증가, (3) VLA 입력 인터페이스 변경이라는 문제가 있음.
- **핵심 질문**: 지도 의미를 추론 입력이 아닌 **학습 감독 신호**로만 써서, VLA가 지도 입력 없이도 도로 기하·위상을 표현하도록 만들 수 있는가?

---

## 2. 문제 정식화

- 시점 t 입력: 전방 이미지 I_t, ego 과거 궤적 H_t = [w_{t−m}, …, w_{t−1}] (w = (x, y, heading)), ego 상태 s_t(속도·가속도 등), 고수준 navigation command c_t.
- 출력: 미래 waypoint 시퀀스 Ŷ_t = [ŵ_t, …, ŵ_{t+n}].
- VLA = vision-language backbone Φ_θ + action decoder Π_ψ. Geo-VLA는 Φ_θ → Φ*로 교체하고 Π_ψ는 설계를 유지한 채 재학습.

---

## 3. Geo-QA 데이터셋

- NAVSIM 전방 프레임 I_i + 로컬 오프라인 지도 주석 M_i + 사전 정의 카테고리 r_i를 GPT-5.4에 넣어 질문 Q_i와 답 A_i를 생성. 답 A_i는 contrastive learning용 지도 의미 텍스트 T_i로도 재사용.
- **5개 카테고리** (Table 4): lane structure, road geometry(방향·곡률), intersection structure, drivable boundary(좌/우 도로 경계 거리 등), road topology(합류·분기·허용 기동).
- 총 **3,000개** 레코드, conversation 형식 JSONL. 예: "좌우 도로 경계까지 거리는?" → "왼쪽 약 2.9 m, 오른쪽 약 2.5 m".
- 완전한 HD map 재구성을 요구하지 않고, 궤적의 가능 영역/방향에 영향을 주는 **국소 도로 관계**만 추출.
- 대조군으로 동일 규모의 **dynamic-object QA**(교통 참여자·운동 상태)도 구성.

---

## 4. 방법론: 2단계 학습

### Stage 1 — Geometry-Aware Pretraining
- Vision encoder와 LLM에 **rank-16 LoRA** adapter 삽입, 보조 projection head g_v, g_t와 함께 최적화. Action decoder는 고정.
- **Contrastive alignment**: pooled 이미지/텍스트 feature를 ℓ2 정규화 후 대칭 InfoNCE (L_con = ½(L_i2t + L_t2i)).
- **QA instruction tuning**: 답 토큰에 대한 autoregressive loss L_qa.
- 전체 L_pre = L_con + λ·L_qa. 역할 분담: contrastive는 장면 간 도로 구조 표현을 조직화하고, instruction tuning은 그 표현을 기하 QA에 쓸 수 있게 만듦.

### Stage 2 — Planning Fine-Tuning
- 적응된 backbone Φ*를 **고정**하고 geometry-aware hidden state F_geo를 산출.
- **Action decoder ψ만** 각 baseline의 원래 imitation-learning 목표·전처리·action 표현·스케줄 그대로 학습. 추가 planner loss 없음.
- 추론: Φ*가 Φ_θ를 대체, projection head·Geo-QA·지도 주석은 모두 제거 → baseline과 동일한 입력 인터페이스.

---

## 5. 실험 설정

- NAVSIM v1: navtrain 학습, navtest 평가, 공식 non-reactive simulation 프로토콜.
- 지표: NC, DAC, TTC, Comfort, EP, PDMS (PDMS는 장면별 계산 후 평균이라 성분 평균으로 재현 불가).
- Baseline 호스트: ReCogDrive, DynVLA (두 모델 모두 저자 재현값 사용, 원 논문 학습/평가 프로토콜 준수).
- Best-of-N 및 후보 선택 방식은 비교에서 제외(single-trajectory protocol).

---

## 6. 주요 결과 (Table 1)

| Method | NC | DAC | TTC | C | EP | PDMS |
|---|---|---|---|---|---|---|
| ReCogDrive† | 98.20 | 97.50 | 94.80 | 100.00 | 87.50 | 90.8 |
| DriveFine | 98.80 | 98.60 | 96.20 | 100.00 | 86.90 | 91.8 |
| LaST-VLA | 98.70 | 97.90 | 95.60 | 100.00 | 86.80 | 91.3 |
| DynVLA† | 98.00 | 97.20 | 94.20 | 100.00 | 85.20 | 91.0 |
| ReCogDrive + Geo-VLA | 98.40 | 98.00 | 95.00 | 100.00 | 88.00 | 91.2 |
| **DynVLA + Geo-VLA** | 98.50 | 98.20 | 95.30 | 100.00 | **89.00** | **92.1** |

- DynVLA + Geo-VLA가 **92.1 PDMS**로 단일 카메라 VLA planner 중 SOTA (이전 최고 DriveFine 91.8 대비 +0.3).
- DynVLA에서 DAC 97.20 → 98.20, TTC 94.20 → 95.30, EP 85.20 → 89.00, Comfort 100 유지.
- 서로 다른 action 생성 메커니즘의 두 planner 모두에서 개선 → plug-and-play 호환성 주장.

---

## 7. Ablation: 지도 인코딩 방식 비교 (Table 2)

| Host | Variant | 추론 오버헤드 | PDMS |
|---|---|---|---|
| ReCogDrive | Baseline | 동일 | 90.8 |
| ReCogDrive | HD map encoder | High | 91.9 |
| ReCogDrive | Lane detection | Moderate | 91.2 |
| ReCogDrive | Geo-QA | 동일 | 91.2 |
| DynVLA | Baseline | 동일 | 91.0 |
| DynVLA | HD map encoder | High | 92.4 |
| DynVLA | Lane detection | Moderate | 91.7 |
| DynVLA | Geo-QA | 동일 | 92.1 |

- HD map encoder가 최고 PDMS지만 localization·검색·인코딩 오버헤드가 큼.
- Geo-QA는 ReCogDrive에서 lane detection과 동률, DynVLA에서 lane detection보다 +0.4, HD map 대비 각각 0.7/0.3 PDMS 이내 — **추가 추론 비용 없이** HD map에 근접.

---

## 8. Ablation: 정적 vs 동적 QA (Table 3, Table 6)

- DynVLA 계획 성능: QA 없음 91.0 / dynamic QA 90.7 / **static QA 92.1**. 동일 규모 dynamic QA는 오히려 소폭 하락 → 이득은 샘플 수나 일반 QA 감독이 아니라 **정적 도로 구조 감독**에서 나옴.
- ReCogDrive QA probe (GPT-Score): 원본 static 63.10 / dynamic 70.30 / 평균 66.70; dynamic QA 학습 시 61.10 / 72.90 / 67.00; static QA 학습 시 **75.10 / 73.60 / 74.35**.
- 해석: 차량·보행자·운동 상태는 이미 backbone의 시각 prior로 잘 포착되지만, 차선 연결성·도로 경계·교차로 구조는 궤적 감독만으로 학습되기 어려우면서 가능 궤적을 직접 제약한다.

---

## 9. 정성적 분석 (Figure 4)

- 회전 장면에서 ReCogDrive는 회전 안쪽으로 파고들며 expert 경로에서 이탈하는 반면, Geo-VLA는 heading 변화를 따라 의도한 차로 통로에 가깝게 유지.
- Lane detection 변형은 곡선로에서 안쪽으로 drift하지만 Geo-QA 변형은 도로 곡률을 따라감 — lane detection은 국소 차선 단서를 별도 추론 branch로 추가할 뿐이고, Geo-QA는 차선 구조·곡률·연결성을 학습 단계에서 감독하기 때문이라고 해석.

---

## 10. 강점

1. **추론 인터페이스 보존**: 지도 의미를 학습 신호로만 사용해 HD map·지도 텍스트·추가 encoder 없이 배포 가능 — 실용적 가치가 큼.
2. **깔끔한 인과 분리 실험**: 동일 규모 dynamic QA 대조군(Table 3)과 QA probe(Table 6)로 "정적 도로 구조 감독"이 핵심임을 설득력 있게 보임.
3. **배포 비용 대비 성능 비교**: HD map / lane detection / Geo-QA를 동일 조건에서 비교하며 오버헤드 축을 함께 제시.
4. **적은 데이터**: 3,000개 QA와 LoRA만으로 1 PDMS 이상의 개선.
5. **호스트 무관성**: 두 개의 서로 다른 VLA planner에 적용해 일관된 개선.

---

## 11. 한계 및 논의

1. **개선폭의 편차**: ReCogDrive에서는 +0.4 PDMS에 그쳐, 호스트에 따라 효과 크기가 크게 다름. 분산·시드 반복이 보고되지 않아 0.3~0.4 수준 차이의 유의성은 불명확.
2. **재현 baseline 의존**: 비교 대상 ReCogDrive/DynVLA가 저자 재현값(†)이므로 원 논문 수치와의 괴리 가능성.
3. **평가 범위**: NAVSIM v1 non-reactive 프로토콜만 사용 — 저자도 인정하듯 상호작용 교통(closed-loop, Bench2Drive 등)에서의 동작은 미검증. NAVSIM v2(EPDMS)도 없음.
4. **오프라인 지도 의존**: Geo-QA 품질·커버리지가 오프라인 지도 주석과 GPT-5.4 생성 품질에 묶임. 생성 QA의 오류율 검증이 없음.
5. **Decoder 설명 부족**: 호스트 action decoder의 구체적 구조(예: diffusion/token)는 원 논문에 위임되어 있어 "다른 action 생성 메커니즘" 주장의 세부가 본문에서 드러나지 않음. 코드 공개 언급 없음.

---

## 12. 총평 및 예상 질문

- **총평**: "지도는 학습 때만, 추론은 카메라만"이라는 명확한 설계로, 작은 QA 데이터와 LoRA만으로 driving VLA의 정적 도로 기하 인식을 강화했다. 개선폭은 크지 않지만 오버헤드 0이라는 점과 dynamic QA 대조 실험이 논문의 설득력을 받쳐준다.
- **예상 질문**
  - Q1. Stage 2에서 backbone을 완전히 고정하는 대신 joint fine-tuning하면 기하 표현이 유지되는가, 아니면 망각되는가?
  - Q2. Geo-QA 규모를 3,000에서 늘리면 PDMS가 계속 오르는가(스케일링 곡선)?
  - Q3. Static + dynamic QA를 함께 쓰면 static 단독(92.1)보다 나은가?
  - Q4. 다중 카메라 VLA나 closed-loop 평가에서도 동일한 이득이 재현되는가?

<!-- VERIFIED: pdf -->
