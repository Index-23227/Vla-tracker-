# 🏗️ Physical AI Benchmark Dashboard — 전체 프로젝트 설계서

## 1. 프로젝트 비전 & 차별화

### 프로젝트명 (후보)
- **VLA-Tracker** — 간결하고 기능이 명확
- **PhysicalAI-Observatory** — 관측소 컨셉, 학술적 뉘앙스
- **VLA-Pulse** — 트렌드의 "맥박"을 짚는다는 의미

### 핵심 컨셉
> "VLA 모델의 벤치마크 성능을 추적하고, AI가 매주 자동으로 분석 코멘터리를 생성하는 살아있는 대시보드"

### 기존 프로젝트와의 차이점

| 기존 프로젝트 | 한계 | 우리의 차별화 |
|---|---|---|
| VLA-Arena | 자체 벤치마크의 raw 수치만 제공 | 다중 벤치마크 크로스 분석 |
| VLABench | 자체 평가 프레임워크 | 모든 주요 벤치마크 통합 뷰 |
| sota.evomind-tech.com | LIBERO/CALVIN/Meta-World 수치 나열 | AI 생성 인사이트 + 시계열 추적 |
| awesome-VLA 리스트들 (5개+) | 정적 논문 링크 모음 | 동적 업데이트 + 분석 + 시각화 |
| Open VLM Leaderboard | VLM에 집중, VLA action 성능 미포함 | Physical AI (VLA) 전문 |

### 핵심 가치 제안 (토큰 집약적 요소)
1. **AI Weekly Digest**: 매주 새 논문/모델의 벤치마크 결과를 Claude로 분석
   - "π₀.5가 LIBERO Long Horizon에서 OpenVLA-OFT를 12.3%p 앞섰다. 이는 flow matching 기반 action head가 autoregressive 방식보다 long-horizon에서 우위를 보여주는 추세를 확인해준다."
   - 토큰 소모: 주당 ~20-30K tokens (논문 5-10편 × 분석당 3-5K)

2. **Cross-Benchmark Synthesis**: 모델 A가 LIBERO에서 1위인데 CALVIN에서는 3위인 이유를 AI가 분석
   - 토큰 소모: 모델당 ~5K tokens

3. **Research Trend Timeline**: 월별/분기별 트렌드 자동 생성
   - "2025 Q3: Diffusion policy → Flow matching 전환 가속", "2025 Q4: 3D representation 통합 VLA 급증"
   - 토큰 소모: 월간 ~10K tokens

4. **Model Card AI Summary**: 각 모델에 대한 구조화된 1-page 요약 자동 생성
   - Architecture, Training data, Key innovation, Limitations, Comparison
   - 토큰 소모: 모델당 ~3K tokens

---

## 2. 기술 스택 & 아키텍처

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                      │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ /data        │  │ /analysis    │  │ /dashboard    │  │
│  │  - benchmarks│  │  - weekly    │  │  - React app  │  │
│  │  - models    │  │  - model     │  │  - Charts     │  │
│  │  - papers    │  │  - trends    │  │  - GitHub     │  │
│  │  (JSON/YAML) │  │  (Markdown)  │  │    Pages      │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                   │           │
│  ┌──────┴─────────────────┴───────────────────┘          │
│  │              GitHub Actions                            │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────────┐         │
│  │  │ Scraper │→ │Claude API│→ │ Build & Deploy│         │
│  │  │ (daily) │  │(analysis)│  │  (weekly)     │         │
│  │  └─────────┘  └──────────┘  └──────────────┘         │
│  └────────────────────────────────────────────           │
└─────────────────────────────────────────────────────────┘
```

### 기술 선택 & 이유

#### 데이터 수집 파이프라인 (Python)
```
스택: Python 3.11+ / requests / BeautifulSoup / PyYAML
이유: arxiv API, Papers with Code API, GitHub scraping에 가장 적합
```

- **arxiv API**: 매일 `cs.RO`, `cs.AI`, `cs.CV` 카테고리에서 VLA 관련 키워드로 새 논문 검색
- **Papers with Code API**: 벤치마크 결과 자동 수집 (LIBERO, CALVIN, Meta-World, RLBench 등)
- **직접 수집**: 주요 논문 Table에서 벤치마크 숫자를 YAML/JSON으로 수동 입력 (초기)
  → 이후 Claude Vision으로 테이블 자동 파싱 자동화 가능

#### AI 분석 엔진 (Claude API)
```
스택: Anthropic Python SDK / claude-sonnet-4-20250514
이유: 비용 효율 + 충분한 분석 품질 (Opus는 비용 부담)
```

분석 파이프라인:
1. 새 벤치마크 데이터 수집
2. 기존 데이터와 diff 계산
3. Claude에게 구조화된 프롬프트로 분석 요청
4. 결과를 Markdown + JSON으로 저장
5. Git commit & push

#### 대시보드 프론트엔드 (React + Recharts)
```
스택: React 18 / Recharts / Tailwind CSS / GitHub Pages
이유: 무료 호스팅 + 인터랙티브 차트 + 빠른 개발
```

- **Recharts**: 시계열 차트 (모델별 성능 추이)
- **Tailwind CSS**: 빠른 스타일링
- **GitHub Pages**: 무료, 자동 배포
- **JSON data fetch**: `/data/*.json`을 빌드 시 import → 정적 사이트로 서빙

#### 자동화 (GitHub Actions)
```
스택: GitHub Actions (cron + workflow_dispatch)
```

- **Daily job**: arxiv 스크래핑, 새 논문 감지
- **Weekly job**: Claude API로 분석 생성, 대시보드 리빌드 & 배포
- **On-demand**: 수동 트리거로 특정 모델/논문 분석

### 핵심 데이터 스키마

```yaml
# data/models/openvla.yaml
name: OpenVLA
version: "1.0"
organization: Stanford / UC Berkeley / TRI / Google DeepMind / PI / MIT
paper_url: https://arxiv.org/abs/2406.09246
code_url: https://github.com/openvla/openvla
date: 2024-06-13
architecture:
  backbone: PrismaticVLM (SigLIP + DinoV2)
  llm: Llama-2 7B
  action_head: autoregressive (tokenized)
  action_space: continuous (7-DoF)
training:
  dataset: Open X-Embodiment
  num_trajectories: 970K
  compute: "64 A100 GPUs, ~14 days"
benchmarks:
  libero:
    libero_spatial: 84.2
    libero_object: 88.4
    libero_goal: 72.0
    libero_long: 53.4
    source: "Table 1, original paper"
    date_reported: 2024-06-13
  simpler_env:
    google_robot_pick: 56.2
    source: "Table 2, original paper"
tags: [open-source, 7B, autoregressive, manipulation]

# data/benchmarks/libero.yaml
name: LIBERO
full_name: "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"
paper_url: https://arxiv.org/abs/2306.03310
categories:
  - libero_spatial: "Spatial relationship understanding"
  - libero_object: "Object generalization"
  - libero_goal: "Goal specification"
  - libero_long: "Long-horizon tasks (10 sequential steps)"
metric: success_rate
higher_is_better: true
evaluation_protocol: "50 rollouts per task suite, average across 10 task suites"
```

### Claude API 프롬프트 설계 (핵심 — 여기서 토큰 투입)

```python
WEEKLY_ANALYSIS_PROMPT = """
You are an expert Physical AI researcher analyzing VLA model benchmarks.

## Context
Here are the latest benchmark results added this week:
{new_results_yaml}

Here is the existing leaderboard data:
{existing_leaderboard_yaml}

## Task
Generate a weekly analysis report in the following JSON structure:

{
  "week": "2026-W12",
  "headline": "One-line summary of the biggest development",
  "key_findings": [
    {
      "finding": "Specific observation about benchmark trends",
      "evidence": "Numbers and comparisons supporting this",
      "implication": "What this means for the field"
    }
  ],
  "model_spotlight": {
    "model": "Name of the most notable model this week",
    "why_notable": "Explanation",
    "strengths": ["..."],
    "limitations": ["..."]
  },
  "trend_update": {
    "current_trend": "What direction the field is moving",
    "supporting_evidence": ["..."]
  },
  "benchmark_gaps": [
    "What benchmarks are missing or insufficient"
  ]
}

Be specific with numbers. Avoid vague statements.
Compare against previous SOTA explicitly.
"""

MODEL_CARD_PROMPT = """
Generate a structured model card for the following VLA model:
{model_yaml}

Include:
1. **Architecture Diagram** (text description of data flow)
2. **Key Innovation**: What's new compared to prior work?
3. **Training Recipe**: Dataset, compute, key hyperparameters
4. **Benchmark Analysis**: Where it excels vs. where it struggles
5. **Comparison**: Explicit comparison with top 3 competitors
6. **Practical Notes**: Inference speed, memory requirements, ease of fine-tuning
7. **Research Impact**: How this advances the field

Format as Markdown. Be concise but specific.
"""
```

---

## 3. 레포 구조

```
physical-ai-dashboard/
├── README.md                    # 프로젝트 소개 (아래 별도 섹션)
├── CONTRIBUTING.md              # 기여 가이드
├── LICENSE                      # MIT License
│
├── data/                        # 구조화된 벤치마크 데이터
│   ├── models/                  # 모델별 YAML 파일
│   │   ├── openvla.yaml
│   │   ├── pi0.yaml
│   │   ├── pi0_fast.yaml
│   │   ├── octo.yaml
│   │   ├── rt2x.yaml
│   │   ├── cot_vla.yaml
│   │   └── ...
│   ├── benchmarks/              # 벤치마크 정의
│   │   ├── libero.yaml
│   │   ├── calvin.yaml
│   │   ├── metaworld.yaml
│   │   ├── simpler_env.yaml
│   │   └── rlbench.yaml
│   └── leaderboard.json         # 자동 생성된 통합 리더보드
│
├── analysis/                    # AI 생성 분석 (Claude output)
│   ├── weekly/                  # 주간 분석 리포트
│   │   ├── 2026-W10.md
│   │   ├── 2026-W11.md
│   │   └── 2026-W12.md
│   ├── model-cards/             # 모델별 AI 분석
│   │   ├── openvla.md
│   │   ├── pi0.md
│   │   └── ...
│   └── trends/                  # 월간/분기 트렌드
│       ├── 2026-Q1.md
│       └── ...
│
├── scripts/                     # 자동화 스크립트
│   ├── scrape_arxiv.py          # 새 논문 감지
│   ├── scrape_benchmarks.py     # 벤치마크 수집
│   ├── generate_analysis.py     # Claude API 호출
│   ├── build_leaderboard.py     # leaderboard.json 생성
│   └── validate_data.py         # 데이터 무결성 체크
│
├── dashboard/                   # React 대시보드
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── LeaderboardTable.jsx
│   │   │   ├── PerformanceChart.jsx
│   │   │   ├── ModelCompare.jsx
│   │   │   ├── WeeklyDigest.jsx
│   │   │   ├── TrendTimeline.jsx
│   │   │   └── BenchmarkSelector.jsx
│   │   ├── data/                # 빌드 시 /data에서 복사
│   │   └── styles/
│   └── public/
│
├── .github/
│   └── workflows/
│       ├── daily-scrape.yml     # 매일 새 논문 체크
│       ├── weekly-analysis.yml  # 주간 분석 생성 + 배포
│       └── on-demand.yml        # 수동 트리거
│
└── docs/
    ├── data-format.md           # 데이터 포맷 가이드
    ├── adding-model.md          # 새 모델 추가 방법
    └── architecture.md          # 기술 아키텍처
```

---

## 4. README 작성 전략

### README의 원칙 (검증된 전략)
- **30초 규칙**: 방문 후 30초 안에 "이게 뭐하는 프로젝트인지" 알 수 있어야 함
- **Show, Don't Tell**: 스크린샷/GIF가 설명보다 강력
- **Honest Limitations**: 한계를 솔직하게 적으면 역설적으로 신뢰 상승

### README 구조

```markdown
# 🔬 VLA-Tracker
### Real-time benchmark tracking & AI-powered analysis for Vision-Language-Action models

[screenshot/GIF of the dashboard here - 가장 중요!]

[![Weekly Analysis](badge)](link)
[![Models Tracked](badge)](link)  
[![Last Updated](badge)](link)
[![Stars](badge)](link)

---

## ⚡ What is this?

VLA-Tracker automatically tracks **[N] VLA models** across **[M] benchmarks**
and generates **AI-powered analysis** every week.

**Unlike static awesome-lists**, this project:
- 📊 **Tracks performance over time** — see how SOTA evolves week by week
- 🤖 **AI-generated insights** — Claude analyzes why Model A beats Model B
- 🔄 **Auto-updated** — GitHub Actions scrapes new results daily
- 📈 **Interactive dashboard** — filter, compare, and explore at [live-link]

> 💡 **Latest finding (Week 12, 2026):**  
> "Flow matching VLAs now dominate 4/5 LIBERO suites.
> The last holdout — LIBERO-Long — shows autoregressive models
> still lead by 3.2%p, suggesting long-horizon planning
> benefits from sequential token generation."

## 🏆 Current Leaderboard (LIBERO)

| Rank | Model | Spatial | Object | Goal | Long | Avg |
|------|-------|---------|--------|------|------|-----|
| 🥇 | PLD | - | - | - | - | **99.0** |
| 🥈 | SimpleVLA-RL | **99.1** | **99.2** | **98.5** | **98.5** | **98.83** |
| 🥉 | X-VLA | 98.4 | 98.8 | 97.6 | 96.4 | **97.8** |

*Updated: 2026-03-19 · [Full leaderboard →](link)*

## 📖 Weekly AI Analysis

Latest: **[Week 12: Flow matching의 지배가 시작됐다](link)**

> 지난주 공개된 3편의 논문이 모두 flow matching 기반 action head를 채택했다...

[Past reports →](link)

## 🚀 Quick Start

### Browse the Dashboard
👉 **[vla-tracker.github.io](link)** — no installation needed

### Use the Data
```python
import yaml
with open('data/models/pi0.yaml') as f:
    model = yaml.safe_load(f)
print(model['benchmarks']['libero']['libero_long'])  # 72.8
```

### Contribute
See [CONTRIBUTING.md](link) — we welcome:
- 📝 New model benchmark data (just add a YAML file!)
- 🐛 Bug reports
- 💡 Analysis feedback

## 📊 Tracked Benchmarks

| Benchmark | Tasks | Models | Focus |
|-----------|-------|--------|-------|
| LIBERO | 4 suites | 35 | Manipulation generalization |
| CALVIN | ABC→D | 17 | Long-horizon, language |
| Meta-World | ML-10/45 | - | Multi-task dexterity |
| SimplerEnv | 5 tasks | 15 | Sim-to-real transfer |
| RoboTwin v1/v2 | 50 tasks | 7 | Dual-arm, bimanual |
| RLBench | 18 tasks | 3 | Diverse manipulation |

## 🤖 How AI Analysis Works

1. GitHub Actions scrapes new papers & benchmarks daily
2. Claude (Sonnet) analyzes the delta vs existing data
3. Generates structured insights with specific numbers
4. Commits analysis to `/analysis/weekly/`
5. Dashboard rebuilds automatically

**Token budget**: ~50K tokens/week → affordable on Pro plan

## ⚠️ Limitations (being honest)

- Benchmark numbers are manually verified from papers — errors possible
- AI analysis may occasionally misinterpret results
- Not all models report on all benchmarks (fair comparison is hard)
- Real-world performance ≠ benchmark performance
- Updates depend on paper availability and our review cycle

## 📜 Citation

If you use this data in your research:
```bibtex
@misc{vla-tracker-2026,
  title={VLA-Tracker: AI-Powered Benchmark Dashboard for Physical AI},
  author={Hyeongjin Kim},
  year={2026},
  url={https://github.com/xxx/vla-tracker}
}
```

## 🙏 Acknowledgments

Built with data from the awesome VLA research community.
Special thanks to the authors of [VLA-Arena], [VLABench], [LIBERO], [CALVIN].
```

### README 핵심 포인트
1. **첫 줄 아래 바로 스크린샷** — 이것만으로 star 전환율 2-3배 차이
2. **"Latest finding" 박스** — 이 프로젝트가 "살아있다"는 증거
3. **현재 리더보드 테이블** — 즉각적 가치 제공
4. **Limitations 섹션** — 신뢰도 상승 (검증된 전략)
5. **Citation 블록** — 학술 커뮤니티 기여 유도

---

## 5. GitHub Star 마케팅 전략

### Phase 1: 씨앗 (0→100 stars, Week 1-2)

**1-1. 퍼스트 서클 활성화**
- HOLI 랩 동료들 → 직접 메시지로 star 요청 (솔직하게)
- YAX (연세대 Physical AI 학회) 멤버들 → 슬랙/카톡 공유
- ICLR/NeurIPS 논문 co-author들 → 이메일
- 목표: 50-80 stars → GitHub trending 후보

**1-2. 한국 AI 커뮤니티**
- 모두의연구소, AI Korea Slack, TensorFlow Korea Facebook
- "VLA 연구하시는 분들을 위해 만들었습니다" — 도메인 특정 커뮤니티
- 한국어 소개 글 작성 (블로그 or 벨로그)

### Phase 2: 이륙 (100→500 stars, Week 3-6)

**2-1. 글로벌 AI 커뮤니티**
- **Reddit r/MachineLearning** — "Show" flair로 포스트
  - 제목: "[P] VLA-Tracker: Auto-updated benchmark dashboard with AI analysis for Physical AI models"
  - 내용: 대시보드 스크린샷 + 가장 흥미로운 AI 분석 인사이트 1개
  
- **Twitter/X** — 핵심 타겟 팔로워
  - VLA 연구자들을 태그: Physical Intelligence, Stanford NLP, UC Berkeley RAIL
  - 주간 분석 결과 중 가장 놀라운 발견을 트윗
  - 예: "Did you know? Autoregressive VLAs still beat diffusion on LIBERO-Long by 3.2%p. Our AI analysis explains why → [link]"

- **Hacker News** — "Show HN"
  - 타이틀: "Show HN: AI-Powered Dashboard Tracking VLA Robot Model Benchmarks"
  - HN은 GitHub 링크 선호 (브랜드 URL보다)

**2-2. 학술 커뮤니티 침투**
- 논문 저자에게 직접 연락: "귀 논문의 벤치마크를 VLA-Tracker에 추가했습니다. 혹시 결과에 오류가 있으면 PR이나 이슈로 알려주세요."
  → 저자가 보통 star를 눌러줌 + 주변에 공유

### Phase 3: 지속 성장 (500→2000+ stars, Month 2-6)

**3-1. 콘텐츠 플라이휠**
```
매주 AI 분석 발행
  → Twitter/Reddit에 흥미로운 인사이트 공유
  → 인용/RT 발생
  → 레포 방문 증가
  → Star 증가
  → GitHub Trending 노출
  → 더 많은 방문
  → (반복)
```

**3-2. 커뮤니티 기여 유도**
- "Good First Issue" 라벨: 새 모델 데이터 추가 (YAML 파일 1개만 만들면 됨)
- CONTRIBUTING.md를 매우 친절하게 작성
- PR merge 시 감사 코멘트 + contributor badge

**3-3. 주기적 콘텐츠 이벤트**
- "2026 Q1 VLA 트렌드 리포트" — 매 분기 발행 → 그때마다 재공유
- "ICML 2026 VLA 논문 벤치마크 비교" — 학회 시즌 활용
- "VLA Model of the Month" — 월간 선정 → 해당 팀/저자가 공유

### Star 성장 현실적 예측

```
Week 1-2:  50-100 stars  (친지인, 연구실, 학회)
Week 3-4:  100-300 stars (Reddit/Twitter 첫 반응)
Month 2:   300-500 stars (Hacker News + 지속적 업데이트)
Month 3-4: 500-1000 stars (학술 커뮤니티 입소문)
Month 6:   1000-2000 stars (콘텐츠 플라이휠 작동)
Month 12:  2000-5000 stars (VLA 분야 "필수 참고" 정착)
```

핵심: **매주 업데이트**가 가장 중요한 성장 드라이버.
정적 레포는 초기 바이럴 이후 성장 멈춤, 살아있는 레포는 복리 성장.

---

## 6. 상세 실행 로드맵

### Week 0: 준비 (1일)
- [ ] GitHub 레포 생성 (public)
- [ ] 레포명, description, topics 설정
- [ ] MIT License 추가
- [ ] .gitignore 설정 (Python + Node)

### Week 1: 데이터 기반 구축 (3-5일)
- [ ] `data/benchmarks/` — LIBERO, CALVIN, Meta-World, SimplerEnv 정의
- [ ] `data/models/` — 핵심 10개 모델 수동 입력
  - OpenVLA, π₀, π₀.5, π₀-FAST, CoT-VLA, Octo, RT-2-X, GR-1, UniVLA, HybridVLA
- [ ] `scripts/build_leaderboard.py` — YAML → leaderboard.json 변환
- [ ] `scripts/validate_data.py` — 데이터 무결성 체크

### Week 2: AI 분석 파이프라인 (3-5일)
- [ ] `scripts/generate_analysis.py` — Claude API 연동
- [ ] 첫 번째 주간 분석 생성 & 검수
- [ ] 모델 카드 5개 생성 (top 5 모델)
- [ ] 프롬프트 튜닝 (구체적 숫자 비교가 나오도록)

### Week 3: 대시보드 MVP (5-7일)
- [ ] React 프로젝트 세팅 (Vite + Tailwind)
- [ ] LeaderboardTable 컴포넌트
- [ ] PerformanceChart 컴포넌트 (Recharts)
- [ ] WeeklyDigest 컴포넌트 (Markdown 렌더링)
- [ ] GitHub Pages 배포 설정
- [ ] **README 작성 (스크린샷 포함)**

### Week 4: 자동화 & 런칭 (3-5일)
- [ ] GitHub Actions — weekly-analysis.yml
- [ ] GitHub Actions — daily-scrape.yml (arxiv 모니터링)
- [ ] 대시보드 최종 점검 & 배포
- [ ] **Phase 1 마케팅 시작**
  - 연구실/학회 공유
  - 한국 AI 커뮤니티 포스트

### Week 5-8: 확장 & 마케팅
- [ ] 모델 수를 10 → 84로 확장
- [ ] Reddit r/MachineLearning "Show" 포스트
- [ ] Twitter/X 스레드 작성
- [ ] Hacker News "Show HN" 시도
- [ ] 논문 저자들에게 이메일
- [ ] 첫 분기 트렌드 리포트 발행

---

## 7. 비용 분석

### Claude API 토큰 사용량 (월간 추정)

| 작업 | 빈도 | 토큰/회 | 월간 총 토큰 |
|------|------|---------|-------------|
| 주간 분석 | 4회/월 | 5K input + 3K output | 32K |
| 모델 카드 (신규) | 4-8개/월 | 3K input + 2K output | 40K |
| 월간 트렌드 | 1회/월 | 10K input + 5K output | 15K |
| 논문 요약 | 20편/월 | 2K input + 1K output | 60K |
| **합계** | | | **~150K/월** |

→ Claude Pro ($20/월) 또는 API 직접 사용 ($3-5/월 수준)으로 충분!

**또는** claude.ai 대화에서 직접 분석 생성 후 복사 → $0 (현재 남는 토큰 활용!)

### 기타 비용
- GitHub Pages 호스팅: **무료**
- GitHub Actions: **무료** (2,000분/월)
- 도메인 (선택): $10-15/년

---

## 8. 장기 진화 방향

### Phase A (Month 1-3): MVP
- 정적 데이터 + 수동 업데이트 + AI 분석
- GitHub Pages 대시보드

### Phase B (Month 3-6): 자동화
- arxiv 자동 스크래핑
- Claude Vision으로 논문 테이블 자동 파싱
- 완전 자동 weekly pipeline

### Phase C (Month 6-12): 플랫폼화
- 커뮤니티 submit 기능 (PR 기반)
- API 제공 (JSON endpoint)
- 학회별 특집 (ICML, CoRL, RSS, ICLR 시즌 리포트)

### Phase D (Month 12+): 확장
- VLA 넘어서 Embodied AI 전체로 확대
- 실시간 알림 (새 SOTA 달성 시 자동 트윗/알림)
- 연구자 프로필 페이지 (누가 어떤 벤치마크에서 기여했는지)
- 스폰서십 가능 (학회/기업으로부터)

---

## 9. 핵심 성공 요소 요약

1. **살아있는 콘텐츠** — 매주 업데이트가 생명
2. **AI 인사이트의 품질** — 뻔한 분석이 아닌 구체적 숫자 비교
3. **도메인 전문성** — 형진님의 VLA 연구 경험이 데이터 curation 품질 보장
4. **README 임팩트** — 첫인상에서 가치를 즉시 전달
5. **커뮤니티 참여** — 데이터 기여를 쉽게 만들기 (YAML 1개)
6. **지속적 마케팅** — 주간 분석을 SNS에 꾸준히 공유
