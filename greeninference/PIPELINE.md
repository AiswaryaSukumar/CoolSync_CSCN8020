# GreenInference — 파이프라인 및 동작 흐름

## 1. 실행 방법

```bash
# 프로젝트 디렉터리로 이동
cd greeninference/greeninference

# 의존성 설치 (최초 1회)
npm install

# 개발 서버 시작 → http://localhost:5174
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 결과 미리보기
npm run preview
```

**주요 의존성**
| 패키지 | 버전 | 역할 |
|--------|------|------|
| React | 19.2.0 | UI 프레임워크 |
| Vite | 7.3.1 | 번들러 / 개발 서버 |
| Recharts | 3.7.0 | 차트 시각화 |
| Framer Motion | 12.35.0 | 애니메이션 |
| Tailwind CSS | 3.4.4 | 스타일링 |
| Lucide React | 0.577.0 | 아이콘 |

---

## 2. 프로젝트 구조

```
greeninference/
├── src/
│   ├── lib/                     ← 핵심 비즈니스 로직
│   │   ├── simulate.js          # 시뮬레이션 엔진
│   │   ├── telemetry.js         # CSV 파싱 & 텔레메트리 처리
│   │   ├── telemetrySchema.js   # 스키마 감지 & 검증
│   │   ├── calibration.js       # 모델 캘리브레이션
│   │   ├── overheadModel.js     # 시설 오버헤드 산출
│   │   ├── capacityManager.js   # 용량 계획
│   │   ├── routingPolicy.js     # 지역 라우팅 추천
│   │   ├── sustainabilitySignals.js  # 전력망 탄소 신호
│   │   └── benchmarks.js        # 벤치마크 프로파일
│   ├── components/              ← UI 컴포넌트 (14개)
│   ├── sections/                ← 페이지 섹션 (4개)
│   ├── App.jsx                  # 루트 컴포넌트
│   └── main.jsx                 # 진입점
├── public/
├── dist/                        # 빌드 산출물
├── vite.config.js
└── package.json
```

---

## 3. 전체 데이터 파이프라인

GreenInference는 두 가지 입력 경로를 지원합니다.

```
┌─────────────────────────────────────────────┐
│              사용자 입력                      │
│  ① 시뮬레이션 설정    ② CSV 텔레메트리 업로드  │
└──────────┬──────────────────┬───────────────┘
           │                  │
           ▼                  ▼
    [시뮬레이션 엔진]     [텔레메트리 파서]
     simulate.js          telemetry.js
           │                  │
           └────────┬─────────┘
                    ▼
           [에너지 산출 & 메트릭]
          Wh/req · J/token · CO2e
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
   [오버헤드    [용량 계획]  [라우팅
    산출]       capacity     정책]
   overhead     Manager    routing
   Model.js                Policy.js
          │         │         │
          └─────────┼─────────┘
                    ▼
           [의사결정 지원 엔진]
           detectCoordinationSignals()
           buildDecisionSupport()
                    │
                    ▼
          [시각화 & 추천 출력]
    차트 · KPI · 알림 · 라우팅 추천
```

---

## 4. 경로별 상세 흐름

### 경로 ①: 시뮬레이션 파이프라인

```
사용자 설정 (토큰 수, 모델, 냉각 전략)
  │
  ▼
토큰 수 → 복잡도 클래스 → 프롬프트 위험도 → 전력 배수
  │
  ▼
3단계 타임라인 생성 (1800ms, 50ms 간격)
  ├── PREFILL    (0–300ms)    : 고전력 연산
  ├── DECODE     (300–1400ms) : 지속 연산
  └── RECOVERY   (1400–1800ms): 전력 감소
  │
  ▼
열 모델 시뮬레이션
  inlet_t = inlet_t-1 + heat×0.08 − cooling_effect
  │
  ▼
전력 적분 → 에너지 계산 (Wh, 컴포넌트별)
  GPU / CPU / DRAM / NIC
  │
  ▼
오버헤드 산출 (overheadModel.js)
  냉각: 42–72% / 전력손실: 12–26% / 유휴예비: 8%
  │
  ▼
탄소 계산
  CO2e = 총 Wh × 전력망 탄소 집약도 (gCO2/kWh)
  │
  ▼
의사결정 신호 감지
  스파이크 위험 / 온도 상승률 / 냉각 지연 / 조정 갭
  │
  ▼
RunSummary + 알림 + 추천 출력
```

**냉각 전략 비교**
| 전략 | 특징 |
|------|------|
| `reactive` (반응형) | 열 스파이크 이후 냉각 개입 → 지연 발생 |
| `coordinated` (조정형) | 사전 예측 기반 냉각 선제 대응 → 안정적 |

---

### 경로 ②: 텔레메트리 파이프라인

```
CSV 파일 업로드
  │
  ▼
스키마 자동 감지 (telemetrySchema.js)
  ├── legacy_minimal       : t, gpuW, inlet, coolingKw
  ├── canonical_minimal    : time_ms, gpu_power_w, inlet_temp_c, cooling_kw
  └── full_stack_v1        : 시설/랙/서버 전체 계층 + 메타데이터
  │
  ▼
CSV 파싱 & 정규화 (telemetry.js)
  컬럼 별칭 처리 (time_ms, t, T → 통합 t 필드)
  필수 필드 검증
  │
  ▼
페이즈 구간 추론
  PREFILL: 전체 시간의 앞 17%
  DECODE:  전체 시간의 17–78%
  BASELINE: 나머지
  │
  ▼
실측 에너지 적분
  컴포넌트별 Wh 계산 (GPU / CPU / DRAM / NIC / 냉각)
  │
  ▼
관측 오버헤드 산출
  buildObservedOverheadAttribution()
  │
  ▼
캘리브레이션 (calibration.js)
  추정값 vs 실측값 비교
  modelWhMultiplier · prefillMultiplier 등 보정 계수 산출
  오차율: beforeError → afterError
  신뢰도: HIGH / MEDIUM / LOW
  │
  ▼
RunSummary (실측 메트릭) + 보정된 모델 파라미터
```

---

## 5. 핵심 모듈 요약

| 모듈 | 핵심 함수 | 역할 |
|------|-----------|------|
| `simulate.js` | `runSimulation()`, `buildDecisionSupport()`, `compareRuns()` | 시뮬레이션 엔진, 전략 비교, 의사결정 |
| `telemetry.js` | `parseTelemetryCsv()`, `computeTelemetryMetrics()` | CSV 파싱, 실측 메트릭 계산 |
| `telemetrySchema.js` | `detectTelemetrySchema()`, `computeLayerConfidence()` | 스키마 감지, 데이터 품질 평가 |
| `calibration.js` | `calibrateRunAgainstTelemetry()` | 모델-실측 간 오차 보정 |
| `overheadModel.js` | `estimateFacilityOverhead()` | 시설 에너지 오버헤드 분해 |
| `capacityManager.js` | `buildCapacityPlan()` | 처리량, 큐 지연, 안정성 모델링 |
| `routingPolicy.js` | `buildRoutingRecommendation()` | 지역별 탄소 비교 및 라우팅 결정 |
| `sustainabilitySignals.js` | `buildSustainabilitySignals()` | 전력망 탄소/수자원 위험 신호 |
| `benchmarks.js` | `buildBenchmarkResult()` | 실행 간 벤치마크 스냅샷 비교 |

---

## 6. 주요 데이터 구조

### RunSummary (실행 결과)
```js
{
  strategy,        // "reactive" | "coordinated"
  tokens,          // 입력 토큰 수
  complexity,      // "LIGHT" | "MEDIUM" | "HEAVY"
  promptRisk,      // 프롬프트 위험도
  peakGpuW,        // 최대 GPU 전력 (W)
  peakInlet,       // 최대 인렛 온도 (°C)
  avgCoolingKw,    // 평균 냉각 전력 (kW)
  stabilityScore,  // 안정성 점수 (0–100)
  whPerRequest,    // 요청당 에너지 (Wh)
  jPerToken,       // 토큰당 에너지 (J)
  co2ePerRequest,  // 요청당 탄소 배출 (gCO2e)
  lagMs            // 냉각 응답 지연 (ms)
}
```

### CalibrationState (캘리브레이션 상태)
```js
{
  status,               // "SEEDED" | "CALIBRATED"
  modelWhMultiplier,    // 전체 에너지 보정 계수
  prefillMultiplier,    // PREFILL 단계 보정
  decodeMultiplier,     // DECODE 단계 보정
  siteOverheadMultiplier, // 시설 오버헤드 보정
  beforeError,          // 보정 전 오차율 (%)
  afterError,           // 보정 후 오차율 (%)
  confidence            // "HIGH" | "MEDIUM" | "LOW"
}
```

---

## 7. 전력망 탄소 집약도 프리셋

| 지역 | 탄소 집약도 (gCO2/kWh) | 특징 |
|------|----------------------|------|
| Quebec | 50 | 수력 발전 중심 |
| Ontario | 110 | 원자력 혼합 |
| US Average | 380 | 화석 연료 혼합 |
| Alberta | 650 | 화석 연료 의존 |

---

## 8. UI 컴포넌트 계층

```
App
├── Navbar
├── KpiCard[]              ← 상단 KPI 요약
├── ChallengeSection       ← 문제 정의
├── EnergyDynamicsSection  ← 열 지연 시각화
├── ArchitectureSection    ← 시스템 아키텍처
└── PrototypeDemoSection   ← 메인 인터랙티브 데모
    ├── 시나리오 / 모델 선택
    ├── 냉각 전략 토글 (reactive ↔ coordinated)
    ├── CSV 텔레메트리 업로드
    ├── CalibrationPanel   ← 캘리브레이션 결과
    ├── CapacityControlCard ← 용량 제어
    ├── 전력 & 온도 차트
    ├── OverheadAttributionCard ← 오버헤드 분해
    ├── RoutingRecommendationCard ← 라우팅 추천
    └── BenchmarkPanel     ← 실행 간 비교
```
