# PRD — TINN 시멘트 수화 4D 플랫폼 v2 (구현 우선 재작성)

문서 버전: 2.0-draft / 작성일: 2026-07-14
목적: 이 문서 하나만으로 Claude(Fable)가 빈 저장소에서 플랫폼 전체를 처음부터 재구현할 수 있어야 한다.
최상위 원칙: **구현이 우선이다.** 검증은 각 마일스톤의 "완료 조건"에 명시된 최소 세트만 수행하고, 그 외의 검증·증적(evidence)·QC 산출물은 만들지 않는다.

---

## 0. 배경 — v1 회고와 v2 설계 원칙

### v1에서 달성한 것 (재사용할 자산)

v1(2026-07 개발분)은 다음을 실제로 동작시켰고, 이 설계들은 v2에 그대로 계승한다:

- 주기 경계 3D 복셀 RVE에서 해상(resolved)/부분해상(fractional)/서브그리드 입자의 3계층 고체 표현.
- 트랜잭션 스테핑: copy-on-write 트라이얼 → 전체 불변식 검사 → 원자적 커밋 또는 완전 롤백.
- float64 성분/상/물/부피 원장(ledger)과 보존 오차 ~1e-25 mol 수준의 수치 폐쇄.
- 연결 액체 클러스터(6-이웃, 주기) + overlap-matrix 기반 보존적 인벤토리 리매핑.
- 의존성 없는 Zarr-v2 디렉터리 체크포인트와 비트단위 재시작 동등성.
- Parrot–Killoh 4상 동역학 2종 프리셋(Elakneswaran 2018, CemGEMS 2021)의 검증된 구현.
- xGEMS `ChemicalEngine`을 격리 프로세스에서 호출하는 어댑터와 PC(Cemdata) 번들 스모크 성공
  (재평형: pH 13.596951, I=0.366219 molal / C3S+물 스모크: pH 12.661942, CSHQ+CH 생성).
- 가변 조성 용액상(CSHQ)을 고정 화학식으로 재해석하지 않는 product-parcel 원장 개념.

### v1이 실패한 방식 (v2에서 금지)

- **모듈 폭증**: 소스 약 90개 모듈(~2.3MB), 테스트 80여 파일. 상당수가 특정 게이트 전용 일회성 모듈
  (`g6c_*`, `g7c_*`, `opc_0d_evidence`, `vendor_neutral_evidence`, `public_operator_evidence` 등).
- **검증 관료제**: evidence registry JSON, QC 정책 파일, 관측 연산자 차터 등 산출물이 코드보다 빨리 늘었다.
- **점진 마이그레이션 비용**: 스칼라 C3S alpha로 시작해 다상(phase-vector) 상태로 이행하며
  호환 뷰·스키마 마이그레이션·이중 원장이 누적됐다.
- **검증 실패가 전체를 블로킹**: XCT 세그멘테이션 QC 실패로 과학 검증이 0건인 채 코드만 확장됐다.
- **문서 부패**: README와 로드맵이 실제 코드 진행보다 수 게이트 뒤처졌다.

### v2 설계 원칙 (전 마일스톤 공통 강제 규칙)

1. **구현 우선.** 기능이 동작하는 스모크 런이 먼저다. 검증은 §6의 최소 세트만.
2. **모듈 상한.** `src/tinn/` 아래 파이썬 모듈은 **최대 16개**. 초과가 필요하면 기존 모듈을 삭제·통합한 뒤 추가한다.
   게이트 이름·마일스톤 이름을 딴 모듈 금지 (예: `g6c_*.py` 금지).
3. **처음부터 다상(multiphase).** 상태는 첫 커밋부터 phase-vector 기반. 스칼라 alpha 전용 경로,
   호환 뷰, 스키마 마이그레이션 코드를 만들지 않는다.
4. **테스트 예산.** 전체 테스트 파일 ≤ 12개, 총 테스트 ≤ 150개. 한 기능에 한 테스트가 원칙.
5. **문서는 2개.** 이 PRD와 README.md. 별도 아키텍처/검증계획/증적 문서를 만들지 않는다.
   결정 변경은 PRD를 직접 수정한다.
6. **검증 실패는 기록하고 전진.** 최소 검증(§6)에서 sanity band를 벗어나면 요약 리포트에 기록하되,
   수치 보존(§6.1)만 통과하면 다음 마일스톤 진행을 막지 않는다. 수치 보존 실패만 블로킹이다.
7. **git 필수.** 첫 커밋 전 `git init`. 마일스톤마다 최소 1커밋. `code_version`은 실제 git 해시.

---

## 1. 제품 정의

### 1.1 한 문장 정의

배합(순수 C3S 또는 OPC 4상 클링커), PSD, w/c, 온도, 시간 스케줄을 입력받아
**질량이 보존되는 시간 의존 3D 미세구조(상 분포·물 분배·공극)와 0D 화학 상태(GEMS 연결 시 pH·상 조성 포함)**를
결정론적·재시작 가능하게 출력하는 시뮬레이션 플랫폼.

### 1.2 입력

| 입력 | 형식 | 필수 여부 |
|---|---|---|
| 결합재 레시피 | 상별 질량분율 (C3S 단독 또는 C3S/C2S/C3A/C4AF + 미배정 잔여) | 필수 |
| PSD | 로그 구간 부피분율 테이블 (합성 기본값 제공) | 필수(기본값 허용) |
| w/c (또는 w/C3S) | 질량비 | 필수 |
| 온도 | K, 등온 | 필수 |
| 동역학 | `tabulated` alpha(t) 테이블 **또는** `pk` 프리셋 이름 | 필수 |
| RVE | 격자 크기(32³/64³), 복셀 크기(0.5–1.0 µm), 시드 | 필수 |
| 화학 백엔드 | `stoichiometric`(합성) 또는 `gems3k`(PC 번들 경로) | 필수 |
| 시간 스케줄 | 출력 시각 리스트 + dt 정책(최소 dt, 최대 재시도) | 필수 |

### 1.3 출력

- 시각별 3D 상태: 상별 점유율 배열, 액체/기체 모세관 분율, particle/cluster 라벨.
- 시계열 요약(JSON): 상별 alpha, 상별 mol, 물 분배(자유/겔/결합), 총·모세관 공극률, 화학수축 진단,
  accept/reject 카운트, 최대 원장 오차.
- GEMS 백엔드 사용 시: 클러스터별 pH, 이온강도, 상별 원소 조성(parcel).
- Zarr-v2 체크포인트(재시작 가능) + 분석 요약 + PNG 슬라이스 이미지(선택).

### 1.4 명시적 제외 (v2 범위 밖 — 코드도 만들지 않는다)

TINN/서로게이트 학습, 건조/RH 경계·수분 유출, dry-residue 정책, 기존 수화물 재용해,
SCM(슬래그·플라이애시 등), XCT/SEM 실험 검증 파이프라인, supervoxel/FVM 수송, 민감도 연구 프레임워크,
GUI. 이들을 위한 훅·플래그·빈 인터페이스도 미리 만들지 않는다 (YAGNI).

---

## 2. 아키텍처

### 2.1 모듈 구성 (상한 16개)

```text
src/tinn/
  config.py       # Pydantic 설정 스키마 + 단위/범위 검증 + config hash
  registry.py     # 상/성분 레지스트리 (ID, 화학식, 몰질량, 몰부피+basis, 겔공극률)
  geometry.py     # 주기 RVE 초기화: 입자 배치, 3계층 분할, 초기 포화
  kinetics.py     # KineticsModel: TabulatedKinetics + ParrotKilloh(2 프리셋)
  dissolution.py  # 상별 목표 용해량의 사이트 배분 + 소진 사이트 재배분
  transport.py    # 액체 클러스터 라벨링 + overlap-matrix 리매핑
  backend.py      # ReactionBackend 프로토콜 + StoichiometricBackend(합성 다상)
  gems.py         # GemsBackend: 격리 프로세스 xGEMS 워커 + PC 번들 감사/해시
  morphology.py   # 내부/외부 분할 부피 배치, 용량 부족 시 reject
  state.py        # SimulationState: dense 배열 + 테이블 + 원장 (처음부터 phase-vector)
  ledger.py       # 성분/상/물/부피 원장 갱신과 불변식 검사
  engine.py       # 트랜잭션 오케스트레이터: trial → 검사 → commit/rollback, dt 적응
  storage.py      # Zarr-v2 체크포인트 저장/로드/검증/원자적 교체
  analysis.py     # 읽기 전용: 공극률, 연결도(percolation), 상 분율 시계열, 슬라이스 PNG
  cli.py          # run / restart / validate-config / report 서브커맨드
  __init__.py
```

### 2.2 인터페이스 계약 (v1 D1–D12 압축 계승)

- **KineticsModel**: 구간 [t, t+dt]에 대한 상별 목표 alpha 벡터만 반환. 위치·화학·형태 무관여.
  - `TabulatedKinetics`: 단조 보간, 시간 증가·alpha∈[0,1] 검증.
  - `ParrotKilloh`: 상별 min(R_ng, R_df, R_hs) 적분, Blaine/온도/RH/물접근 보정.
    프리셋 `pk_elakneswaran_2018`(기본), `pk_cemgems_2021`. RH<0.55 컷오프.
- **ReactionBackend**: 클러스터별 (기존 용액 인벤토리 + 이번에 방출된 원소)를 받아
  (잔여 용액 인벤토리, 신규 침전 parcel 목록, 상태/잔차)를 반환. 미반응 클링커는 절대 입력하지 않는다.
  좌표를 모른다. 계산 불가 필드는 NaN + `not_available`이지 0이 아니다.
  - `StoichiometricBackend`: 4상 각각에 고정 합성 화학식(C3S→C1.7SH4+1.3CH 등 config 명시)을 적용하는
    결정론 백엔드. GEMS 없이 전체 파이프라인을 구동·테스트하기 위한 것.
  - `GemsBackend`: xGEMS ChemicalEngine. 트라이얼당 클러스터당 1회 호출, 격리 작업 디렉터리
    (ipmlog.txt/xGEMS.log 오염 방지), 소스 번들 해시 전후 검증, 클링커 상 억제(bound=0),
    cold-start 실패 시 trial reject.
- **MorphologyModel**: parcel의 벌크 외피 부피(V_skel/(1-ε_gel))를 소스 입자 주변
  내부(inner)/외부(outer) 분할로 배치. 부피를 버리거나 임의 이동 금지 — 용량 부족은 reject.
- **TransportModel**: 액체분율 임계 + 면 전도 기준의 6-이웃 주기 클러스터. 화학·배치 무관여.
- **Engine**: 유일한 상태 변경 주체. trial 생성 → kinetics → dissolution → backend → morphology →
  ledger/불변식 → commit 또는 rollback+dt 축소. 거부 사유는 안정 식별자
  (`placement_capacity`, `insufficient_water`, `cluster_dryout`, `backend_failure`, `balance_*`).

### 2.3 상태 모델 (v1 state_vector 압축, 처음부터 다상)

**Dense (z,y,x 순, float64):**
`anhydrous_fraction[phase]`, `hydrate_fraction[phase]`(벌크 외피), `capillary_liquid`,
`capillary_gas`(고정 RVE 잔여), `particle_id`(int64, -1=없음), `cluster_id`(int64, 스냅샷-로컬).
복셀 항등식: Σ고체 + Σ수화물 + 액체 + 기체 = 1 (허용오차 내, 클리핑 금지).

**테이블:** particles(3계층 공통), subgrid_bins, clusters(+원소 인벤토리 벡터),
product_parcels(시각·클러스터·상별 불변 행: 상량, 원소 매핑, 골격/벌크 부피 — 누적 이중 원장 금지),
remap_events(overlap 행렬 기록).

**헤더:** `kinetic_phase_ids`(순서 고정), `phase_alpha`/`initial_phase_mol`/`unmet_mol` 벡터,
time, dt, config_hash, code_version(git), backend_id, RNG 전체 상태, accept/reject 카운트.

**물 원장:** 자유 모세관수 / 겔수(수화물 외피 내부, 점유율 비가산) / 결합수(상 조성 유래) — 상호배타,
mol이 권위이고 kg은 파생.

**체크포인트:** Zarr-v2 디렉터리 스토어, 무압축 raw chunk + JSON 테이블, manifest 체크섬,
임시 디렉터리 작성→검증→원자적 rename. 재시작은 무중단 실행과 비트단위 동일해야 한다.

---

## 3. 마일스톤 (각각 독립 세션 1~2회 분량, 순서 고정)

각 마일스톤의 **완료 조건(DoD)** = ① 명시된 스모크 런 성공 ② 명시된 최소 테스트 통과 ③ git 커밋.
그 이상의 테스트·문서·검증을 추가하지 않는다.

### M0 — 골격과 초기화
- 패키지 골격, `config.py`, `registry.py`, `geometry.py`, `cli.py`의 `validate-config`.
- 32³ 합성 PSD로 다상(C3S 단독 배합도 4상 스키마의 특수형) RVE 초기화.
- DoD: 목표 고체분율·w/c 달성(오차 보고), 동일 시드 → 동일 배열 해시. 테스트 ~15개.

### M1 — 보존 코어 (합성 백엔드 전체 루프)
- `state/ledger/dissolution/transport/backend(합성)/morphology/engine/storage` 전체 연결.
- 32³ C3S-물 런: alpha 0→0.35, 롤백 포함, 체크포인트/재시작 비트단위 동등.
- DoD: §6.1 수치 보존 전 항목 통과. 테스트 ~50개 (v1의 G/K/L/A/C/P 매트릭스에서 핵심만 발췌).

### M2 — P&K 동역학과 4상 구동
- `kinetics.py`에 ParrotKilloh 추가, NIST SRM 114q 레시피(60/14/7/10/9, Blaine 381.8 m²/kg)로
  4상 alpha 스케줄 생성, 합성 백엔드로 3D 런.
- DoD: 두 프리셋이 §6.2의 P&K 앵커 값 재현(단위 테스트), 4상 3D 런에서 상별 원장 폐쇄. 테스트 ~20개.

### M3 — GEMS 0D 프로브
- `gems.py`: PC 번들 로드, 격리 워커, 해시 감사. 0D 누적 화학 프로브
  (P&K가 방출한 양만 평형화, 클링커 억제).
- DoD: §6.2 GEMS 앵커 2건 재현(회귀 스냅샷), C3S 0D 프로브에서 원소·물 수지 폐쇄. 테스트 ~15개.

### M4 — GEMS→3D 결합
- GemsBackend를 engine에 연결: 클러스터별 방출 원소 집계 → xGEMS → parcel 커밋 →
  형태 배치. CSHQ는 endmember 합으로 검증하고 고정 화학식 재해석 금지.
  기존 수화물 재용해는 비활성(백엔드 입력에서 제외).
- DoD: 32³ OPC 런 1/3/7일 시점 완주(또는 명시적 feasibility ceiling 보고),
  §6.1 폐쇄 + §6.3 sanity band 리포트 생성. 테스트 ~25개.

### M5 — 분석과 리포트
- `analysis.py`: 총/모세관 공극률 시계열, 액체 percolation 여부, 상 분율, 중앙 슬라이스 PNG,
  `cli report` 서브커맨드로 요약 JSON+이미지 일괄 생성.
- DoD: M4 런 산출물에서 리포트 생성, §6.3 sanity band 자동 판정 포함. 테스트 ~15개.

**마일스톤 밖 작업 금지 목록**: M1 전에 GEMS 코드 작성 금지, M4 전에 analysis 작성 금지,
어떤 시점에도 §1.4 제외 항목 착수 금지.

---

## 4. 핵심 알고리즘 명세 (구현 시 그대로 따를 것)

### 4.1 P&K 상별 속도 (day⁻¹)

R_ng = (K1/N1)(1−α)[−ln(1−α)]^(1−N1),
R_df = K2(1−α)^(2/3) / [1−(1−α)^(1/3)],
R_hs = K3(1−α)^N3, 제어속도 = min(세 후보).
보정: Blaine 비 × Arrhenius(프리셋별 기준온도 293.15/298.15 K) × RH 인자 × 물접근 인자.
프리셋별 차이: 2018은 전체 제어속도에 표면 스케일링, 2021은 nucleation/growth에만.
총 클링커 alpha = 4상 질량가중 alpha / 4상 분율 합 (전체 시멘트 질량 아님).
alpha 시드·적분 스텝은 config 명시 수치 정책.

### 4.2 용해 배분

상별 목표 Δn = 초기 mol × Δalpha. 접근 가능(액체 인접) 사이트에 기하 표면적 가중으로 배분,
소진 사이트 잔여는 접근 가능 사이트에 반복 재배분. `phase_id` 필터 필수.
표면적은 상대 가중치일 뿐 절대 속도가 아니다. 미달성분은 `unmet_mol`로 기록 (은폐 금지).

### 4.3 클러스터 리매핑

이전/신규 액체장의 물리적 겹침부피 Ω_ab로 성분·잔여수를 비례 배분(merge/split 동시 지원).
인벤토리 보유 클러스터의 겹침 0 = `cluster_dryout` reject.

### 4.4 부피 규약

수화물 벌크 외피 = 골격/(1−ε_gel), basis(`solid_skeleton`|`bulk_envelope`)는 레지스트리 필수 필드,
누락은 하드 에러. 겔수는 질량 원장에만 계상. capillary_gas는 고정 RVE 잔여이고
화학수축은 동일 원장의 진단값(이중 계상 금지).

---

## 5. 비기능 요구

- Python ≥3.11, 코어 의존성 numpy+pydantic만. zarr/h5py 불요(자체 Zarr-v2 writer).
  gems 모듈만 xgems(pybind11)를 지연 import — 미설치 시 다른 전 기능 정상.
- 결정론: 동일 config+seed → 동일 결과. RNG는 `numpy.random.Generator(PCG64)` 상태 전체 저장.
- 성능 목표: 32³ 합성 런 < 5분, 64³ < 1시간 (일반 데스크톱). 미달 시 기능보다 프로파일링 우선.
- 플랫폼: Windows 우선 (`py` 런처), 경로 구분자 하드코딩 금지.
- 오류 정책: 모르는 상/성분/단위/basis는 추측하지 않고 거부. 침묵 폴백 금지.

---

## 6. 최소 실험값·검증 세트 (이것이 전부이며, 추가하지 않는다)

### 6.1 수치 보존 (블로킹 — 유일하게 진행을 막을 수 있는 검증)

| 항목 | 기준 |
|---|---|
| 성분/원소 수지 | \|e\| ≤ atol + 1e-8·\|M\| (글로벌 float64 원장) |
| 복셀 점유 항등식 | 전 복셀 오차 ≤ 1e-12, 음수·클리핑 없음 |
| 물 수지 | 자유+겔+결합 = 총량, 동일 rtol |
| 배치 수지 | 백엔드 부피 = 요청 = 배치 부피 (허용오차 내), 손실·이동 없음 |
| 롤백 불변성 | 거부 트라이얼 후 커밋 상태 해시 불변 |
| 재시작 동등성 | 중단/재시작 = 무중단, dense 배열·원장·RNG 비트단위 동일 |

### 6.2 재현 앵커 (회귀 테스트 — 문헌/기측정 값과의 일치)

| 앵커 | 값 | 용도 |
|---|---|---|
| P&K 파라미터 표 | Elakneswaran 2018 (doi:10.3390/app8122597) 발표 표 그대로 | kinetics 단위 테스트 입력 |
| P&K 변형 프리셋 | Kulik 2021 / CemGEMS 문서 값 | 제2 프리셋 테스트 |
| OPC 레시피 | NIST SRM 114q: C3S 60, C2S 14, C3A 7, C4AF 10, 기타 9 wt%; Blaine 381.8 m²/kg | 표준 4상 픽스처 (9%는 미배정 유지, 발명 금지) |
| GEMS 앵커 1 | PC 번들 저장 DBR 재평형 → pH 13.596951, I 0.366219 molal | gems 어댑터 회귀 스냅샷 |
| GEMS 앵커 2 | 1.0 g C3S + 0.5 g H2O + O2 시드 → pH ≈ 12.66, 주생성상 CSHQ+Portlandite | gems 어댑터 회귀 스냅샷 |
| C3S 픽스처 | 1.0 g C3S, w/C3S 0.50, 293.15 K, 표면 스케일링 중립(1.0) | 최소 스모크 조건 |

### 6.3 Sanity band (비블로킹 — 리포트에 pass/warn만 기록)

M4 이후 OPC 런의 자동 판정 항목. 벗어나면 warn으로 기록하고 계속 진행한다.
목적은 "물리적으로 터무니없지 않음"의 확인이지 과학적 검증이 아니며, 리포트에 그렇게 명기한다.

| 관측치 | band (20 °C, w/c≈0.5 OPC 문헌 통상 범위) |
|---|---|
| 총 클링커 alpha @ 1 d / 7 d / 28 d | 0.25–0.55 / 0.50–0.80 / 0.65–0.90 |
| C3S alpha ≥ C2S alpha (모든 t) | 순서 위반 없음 |
| CH(Portlandite) 질량 | 단조 증가 (재용해 off이므로) |
| 모세관 공극률 | 단조 감소, 최종 < 초기 액체분율 |
| 화학수축 진단 @ 28 d | 0.03–0.08 mL/g(반응 시멘트) 자릿수 |
| 클러스터 pH (GEMS) | 12.4–13.9 |

### 6.4 하지 않는 것

QXRD/TGA/XCT 데이터셋 파이프라인, 캘리브레이션/홀드아웃 절차, 불확도 전파, 해상도 앙상블 연구,
evidence registry. 이것들은 v2 완료 후 별도 프로젝트로 다룬다.

---

## 7. Fable 재구현 실행 계획

1. **새 저장소에서 시작한다.** v1 코드는 참조하되 복사하지 않는다(참조 경로:
   `C:\Users\solmo\TINN platform`). 특히 v1의 `pk_kinetics.py`, `storage.py`, `transport.py`는
   알고리즘 참조 가치가 높다.
2. 세션 구성: 마일스톤당 1개 세션을 기본으로 하고, 각 세션 시작 시 이 PRD 전문과
   직전 마일스톤 커밋 로그를 컨텍스트로 준다.
3. 각 세션 지시문 템플릿: "PRD §3의 M_n을 구현하라. DoD의 스모크 런과 테스트만 작성하고,
   §0의 설계 원칙(모듈 상한 16, 테스트 예산, 문서 2개)을 위반하지 마라."
4. 리뷰 게이트: 각 마일스톤 커밋 후 `/code-review` 1회, 발견된 correctness 이슈만 수정.
5. GEMS PC 번들(`PC-dat.lst` 외 4파일)과 xGEMS 설치는 M3 시작 전에 사용자가 새 저장소에
   복사·확인한다. 미준비 시 M3–M4를 보류하고 M5를 합성 백엔드 산출물로 선진행한다.

## 8. 성공 기준 (v2 전체)

- [ ] M0–M5 전체 DoD 통과, 총 모듈 ≤ 16, 총 테스트 ≤ 150.
- [ ] 합성 백엔드만으로: 4상 OPC 32³ 런 + 재시작 + 리포트가 GEMS 설치 없이 동작.
- [ ] GEMS 백엔드로: OPC 32³ 런이 1일 이상 시뮬레이션 시간을 완주하고 §6.3 리포트 생성.
- [ ] 신규 사용자가 README만으로 30분 내 첫 런 재현.
- [ ] 이 PRD가 코드와 일치 (불일치 발견 시 코드가 아니라 PRD를 먼저 갱신했는지 확인).
