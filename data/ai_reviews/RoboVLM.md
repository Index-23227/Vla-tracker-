# RoboVLMs: What Matters in Building VLA Models for Generalist Robots

> **한 줄 요약**: 8 VLM backbone × 4 policy architecture × 600+ 실험을 통해 VLA 설계 4대 질문(Why/Which/How/When)을 체계적으로 검증하고, KosMos+Policy-Head 구성이 CALVIN/SimplerEnv에서 SOTA를 달성함을 보임 (ICLR 2025).

---

## 1. 배경 및 동기

- VLA 설계 공간(backbone, action space, history aggregation, cross-embodiment data 사용 시점)이 방대하나 **공정한 비교 연구가 부재** (Sec. I)
- 저자들은 4가지 핵심 질문으로 정리: ① Why VLA? ② Which backbone? ③ How to formulate? ④ When to use cross-embodiment data? (Fig. 2)
- 모든 코드/모델/데이터셋을 robovlms.github.io에 공개

---

## 2. 방법론: RoboVLMs 통합 프레임워크

### 분석 축 (Fig. 1b)
| 축 | 선택지 |
|---|--------|
| VLM Backbone | LLaVA, Flamingo, KosMos, PaliGemma, Qwen-VL, MoonDream, UForm, Phi (8종) |
| Action Space | Discrete (token) vs Continuous |
| History | One-Step vs Interleaved vs Policy-Head |
| Training Objective | MSE+BCE vs Flow Matching |
| Cross-Embodiment | Pre-train / Co-train / Post-train |

### 평가 환경 (Sec. II.A)
- **CALVIN**: 34 tasks, 24K teleop demos, ABCD splits, 1~5 consecutive-task 평균길이(Avg.Len.)
- **SimplerEnv**: WidowX+Bridge, Google Robot real-to-sim
- **Real Robot**: Kinova Gen3 + Robotiq 2F-85, 105 tasks, 70K trajectories, 20 평가 task × 5 setting

---

## 3. 실험 결과 (PDF 수치 인용)

### Backbone × Structure ablation (Table I, CALVIN ABCD→D)
| Backbone | Structure | Avg.Len. |
|---|---|---|
| LLaVA | One-Step Disc | 1.85 |
| LLaVA | Policy-Head Cont | 2.71 |
| Flamingo | One-Step Disc | 1.22 |
| Flamingo | Policy-Head Cont | 4.09 |
| KosMos | One-Step Disc | 0.55 |
| **KosMos** | **Policy-Head Cont** | **4.49** |
| PaliGemma | Policy-Head Cont | 4.42 |

→ **Continuous + Policy-Head + KosMos가 모든 형태 중 최고**.

### CALVIN ABCD→D, Extended Table 2: KosMos P.H. (RoboVLMs) Avg.Len. **4.25**, GR-1 3.06 → **+1.19 task** 향상 (Sec. II.B).

### SimplerEnv Google Robot (Extended Table 3b)
| Method | Pick Coke Can Avg | Move Near | Drawer Avg |
|---|---|---|---|
| RT-2-X | 0.787 | 0.779 | 0.250 |
| OpenVLA-7b | 0.163 | 0.462 | 0.356 |
| **RoboVLMs** | **0.970** | **0.888** | **0.551** |

### Training objective (Table IIa, ABCD): Flow Matching Chunk Avg.Len. 4.09 vs MSE+BCE Chunk 4.04 → 거의 동등.

### MoE (Table IIb, ABC): MoE+MSE 3.69 vs no-MoE 3.57 → MoE는 **unseen scene(ABC)에서만 일관된 이득**, ABCD에서는 오히려 손해.

---

## 4. 핵심 Findings

- **Finding 1.1/1.2**: VLA는 generalist policy의 유망한 경로이며 real-robot에서도 강건 (Sec. II.B).
- **Finding 2**: KosMos / PaliGemma처럼 **충분한 VL pre-training을 받은 backbone이 결정적** (Extended Table 5).
- **Finding 3.1**: Continuous action + multi-step history + Policy-Head가 최적.
- **Finding 3.3**: Flow Matching ≈ MSE+BCE (One-Step-Continuous에서); chunk 실행이 first/ensemble보다 안정적.
- **Finding 4** (Sec. II.E, 본문에서 확인): Cross-embodiment **pre-training**의 일관된 이득은 없음. **Post-training**이 가장 효과적이며, 동일 robot/task data가 가장 큰 부스트.

---

## 5. 한계 및 비판적 고찰

1. **Compute cost**: 8 backbone × 4 structure × 다수 data scale → 재현 비용이 매우 큼 (RoboVLMs가 32 A100을 쓴다고 YAML에 기재).
2. **CALVIN/SimplerEnv 위주**: 두 sim benchmark 위주이므로 dexterous/contact-rich에서의 일반성은 미검증.
3. **MoE 결론의 분기**: ABC vs ABCD에서 정반대 결론 → "언제 MoE를 쓸지" 판단 기준이 추가 필요.
4. **YAML 불일치**: 본 PDF는 LIBERO 평가를 포함하지 않음 (CALVIN/SimplerEnv/Real만 다룸). 그러나 `data/models/robovlm.yaml`에는 LIBERO 4-suite 점수(91.8/93.4/86.8/64.2)가 기재됨 → **출처 검증 필요**.

---

## 6. 총평 및 예상 날카로운 질문

**강점**: "무엇이 중요한가?"에 대한 600+ 실험 기반의 실증적 답. 프레임워크가 오픈소스. **약점**: 결론의 시효성, LIBERO 미평가.

| # | 질문 | 핵심 답 |
|---|------|---------|
| 1 | KosMos가 PaliGemma 대비 우위인 이유는? | Table I에서 두 backbone 모두 P.H.+Cont로 4.4+ 달성, KosMos가 0.07 차이로 미세 우위. PDF는 "두 backbone이 distinguished" 표현 사용 (Finding 2). |
| 2 | Cross-embedding pre-training이 효과적이지 않다는 결론은 다른 작업과 모순되지 않나? | 저자들은 in-domain 데이터로 post-training 시 큰 이득(+50%, Sec. II.E)을 확인. Pre-training만으로는 부족하다는 의미. |
| 3 | LIBERO 점수가 paper에 없는데 어떻게 보고됐나? | PDF 기준 LIBERO 미평가. 추가 검증 필요. |

<!-- VERIFIED: pdf -->
