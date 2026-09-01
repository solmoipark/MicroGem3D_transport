# TINN v4.0/RT 과학·공학 타당성 검토

검토일: 2026-08-20 · 검토 대상: `TINN_ reactive transport` (rt-dev, f7be88b → RT-W2a 커밋) · 검토자: Claude (외부 리뷰)

---

## 0. 검토 범위와 방법

PRD.md v4.0-RT(2026-08-20 개정 전문), README.md, rt-dev 커밋 이력 5건(RT-W0 포트/PRD 개정 → RT-W1 모드 B → W1 리뷰 수정 → RT-W2a 모드 C 봉인계), 소스 16모듈 중 핵심(`transport.py`, `engine.py`, `kinetics.py`, `config.py`, `ledger.py`), RT 테스트(`test_rt.py` 전문 + 스위트 구성), 실행 산출물(`runs/rt0~rt2` 앵커 런, `rt1_demo_tau720` 모드 B 데모 요약)을 직접 읽고, 핵심 모델링 선택 각각을 반응-수송·시멘트화학 문헌과 대조했다. 판정 기준은 네 가지다: 물리·화학적 근거, 수치해석적 건전성, 문헌 정합성, 로드맵의 순서·실행 가능성.

현재 상태 스냅샷:

| 구분 | 항목 | 상태 |
|---|---|---|
| 완료 | M0–M5, v2.1(SCM), v2.2(전평형 재용해), rev.2, v3.0 E1–E3b(endmember 원장·염 담체) | 플랫폼 기반, f7be88b |
| 완료 | RT-W0(포트·앵커), RT-W1(모드 B), RT-W2a(모드 C 봉인계, FORMAT_VERSION 4) | rt-dev, 테스트 247개 그린 |
| 남음 | RT-W2b(GEM 경제: dirty flag·연령 상한·호출 예산) | config는 이미 거부 게이트로 예약 |
| 남음 | RT-W3(경계 저수조), RT-W4(검증 응용: FA zoning / Ca 용출 / 탄산화) | 설계 명세만 존재 |
| 문서화된 장기 경로 | 종별 Nernst–Planck 영전류 사영 | v2 업그레이드로 명시 |

## 1. 종합 판정

**완료된 작업과 남은 로드맵 모두 과학·공학적으로 타당하다.** 블로킹 수준의 물리·수치 결함은 발견하지 못했다. 두 수송 모드는 확립된 방법론의 정합적 구현이고(모드 C는 혼합셀/구획 반응-수송 + GEM 연산자 분할, 모드 B는 1차 교환 폐포), 채택된 단순화(공통 D₀, 겔수 무저장, 도메인 즉시혼합, 도메인 id 순 용량 청구)는 전부 선례가 있으며 그 한계가 PRD에 이례적으로 정직하게 문서화되어 있다. 남은 업그레이드도 순서가 옳다 — GEM 경제(W2b)가 미세 분할의 실용성 전제이고, 경계 저수조(W3)가 응용(W4)의 전제라는 의존 구조가 정확히 반영되어 있다. 아래 §4의 권고 7건은 개선·주의 사항이지 결함 지적이 아니다.

## 2. 완료 작업의 타당성

### 2.1 기반 플랫폼 (M0–M5, v2.1–v3.0)

Parrot–Killoh 구현은 표준형 그대로다. 세 속도식(R_ng, R_df, R_hs)의 수식, 프리셋 파라미터(C3S: K1 1.5, N1 0.7, K2 0.05, K3 1.1, N3 3.3, Ea 41.57 kJ/mol 등), RH 인자 ((RH−0.55)/0.45)⁴와 컷오프 0.55, 물접근 인자 (1+3.333(H·w/c−α))⁴에 클리핑을 더한 형태 모두 Parrot–Killoh/Lothenbach 계보의 발표 표와 일치하고, 두 프리셋의 차이(2018은 전 속도 표면 스케일, 2021은 핵생성·성장만)를 출처와 함께 분리한 것도 옳다. 클러스터별 GEMS snapshot 전평형은 국소평형가정(LEA)의 공간판으로, GEMS 기반 열역학 수화 모델의 표준 관행을 3D로 일반화한 것이다. CSHQ/CNASH 가변 조성 고용체를 고정 화학식으로 붕괴시키지 않는 endmember 원장(E1/E2)은 고용체 열역학(예: 비이상 Ca(OH)₂–SiO₂ 고용체 모델 [20])과 정합하는 올바른 상태 표현이다. 염 담체의 용해도 제어(E3b) — 속도식을 발명하지 않고 접근 가능 전량을 평형에 내놓는 규칙 — 는 석고·반수석고류의 빠른 표면 제어 용해를 감안하면 열역학 수화 모델 관행과 일치하는 결정이며, "근거 없는 시간상수 삭제"의 판단 과정 자체가 모범적이다. 겔 상대확산계수 0.0025는 Garboczi–Bentz의 C-S-H 1/400 [17], 겔공극률 0.28은 Powers 계보의 표준값이고, Kozeny-Carman을 리포트 전용으로 격리한 것도 적절하다.

한 가지 알려진 구조적 한계(코드 결함 아님): P&K형 동역학은 황산염 소진↔C3A 반응 같은 화학→동역학 피드백을 받지 않는다. E3로 황산염을 넣어도 클링커 알파 스케줄은 변하지 않는다는 뜻인데, 이는 P&K류 모델의 공통 전제("정상 석고 배합 OPC")이고 PRD §1.2 말미의 비교성 주의가 이미 이 방향을 가리키고 있다. 저황산염·과황산염 배합을 다루게 되면 이 한계를 리포트 주석으로 승격할 것을 권한다.

### 2.2 모드 B — 속도제한 재평형 (RT-W1)

**물리적 근거가 문헌으로 뒷받침된다.** C-(A-)S-H의 재평형은 느리다: 용액 농도는 첫 1일에 급변한 뒤 14일 이후엔 미미하게 변하지만, 고체 내부의 Si/Al 재분배와 조성 재조정은 10년 스케일까지 관측된다 [1]; 준안정 평형 도달에 수개월~3년이 걸리고 Ca/Si가 높을수록 빨라진다 [2]. 반면 CH·AFt 같은 결정상의 용해·성장은 표면 제어로 빠르다. 따라서 "다-endmember(고용체) 채널만 전역 τ, 단일-DC 결정상은 τ=0(전량 제공)"이라는 기본 배선은 문헌이 지지하는 방향이고, CH 유보가 pH 완충을 파괴한다는 PRD의 논거도 옳다.

**수학적 성격도 건전하다.** f=min(1, Δt/τ)는 1차 교환 ODE(dn/dt = −n/τ + 침전)의 명시적 이산화로, 스텝당 회전량이 n·Δt/τ → 교환율 n/τ로 수렴하는 일관된(1차 정확) 스킴이다. f가 (config, 시도 dt)의 순수함수라 dt 반감 재시도와 재시작 결정론이 공짜로 성립하는 설계도 좋다. 흥미로운 부수 효과로, 포화 용액에서는 offered분이 GEMS에 의해 그대로 재석출되므로 순용해율이 자동으로 작아진다 — 즉 선형 구동력 유사의 포화 의존성이 창발한다. 강한 불포화에서는 순용해율이 n/τ로 상한되는데, 이는 표면적·(1−Ω)에 비례하는 TST형 속도식과 다른 거동이지만, 모드 B의 목적이 용해 속도론이 아니라 "침전 시점 조성의 보존 대 재평형"의 회전율 폐포라는 점에서 타당하고, τ는 어차피 보정 파라미터로 남는다. 실측 데모(τ=720 h, 168 h 런)가 sanity band를 통과하면서 전평형 궤적과 분기하는 것도 의도된 거동이다.

주의 2건은 §4 권고에 정리했다(요지: Δt ≥ τ에서 f=1로 조용히 전평형에 회귀하는 것은 "침묵 폴백 금지" 원칙과 긴장 관계; 선형 f는 Δt/τ가 큰 스텝에서 지수형 1−e^(−Δt/τ) 대비 교환을 과대평가 — Δt≪τ 운용이면 무시 가능).

### 2.3 모드 C — 도메인 분할 + 그래프 확산 (RT-W2a)

**아키텍처 선택이 문헌 관행과 정확히 맞는다.** "연결 클러스터 ∩ 정적 타일 = well-mixed 반응기, 반응기 사이 유효 확산"은 혼합셀(compartment/mixing-cell) 반응-수송의 정석이며, GEMS3K 자체가 "조성이 변한 제어부피마다 평형 솔버를 호출하는 연산자 분할 RT"를 위해 설계된 커널이다 [9]. OpenGeoSys-GEM 등 GEM 결합 RT 코드들이 같은 구조를 쓴다 [10]. per-voxel FVM 대신 도메인 그래프를 선택한 것은 비용 구조(GEM 10–60 s/스텝 vs 수송 연산자 0.1–0.4 s 목표)에서 합리적이고, "타일=격자면 현행과 비트 동일"이라는 자명 분할 게이트는 회귀 안전성의 모범이다.

**수치 코어가 옳게 구현되었다.** 직접 확인한 사항: (i) backward Euler를 플럭스 형태로 적용해 — 같은 float를 한쪽에서 빼고 다른 쪽에 더함 — 원소 보존이 부동소수점 수준에서 구성적으로 정확하다. (ii) 계 행렬 diag(W)+ΔtL은 M-행렬이라 양성이 보장되고, CG 더스트 음수는 결정론 보정 후 1e-11 상대 초과 시 하드 실패한다. (iii) 명시적 FTCS 배제 판단이 옳다 — 0.5 µm 복셀·D₀=1e-9 m²/s에서 안정 한계는 도메인 크기에 따라 10⁻⁸~10⁻⁶ h 오더로, 목표 dt(시간 단위)와 6자릿수 이상 괴리한다(PRD의 3e-6 h는 대표 도메인 가정의 추정치이고 결론은 동일). (iv) 2-도메인 해석 앵커의 대수식 Δc⁺=Δc/(1+Δt·λ), λ=(D₀G/p)(1/W₁+1/W₂)를 독립 유도로 확인했고 테스트가 rel 1e-12로 고정한다. (v) 반대칭·양성·균일농도 정지성·결정론의 링 불변식 테스트, 합성·GEMS 양쪽의 1-도메인 비트 동일 게이트, 분할 런의 재시작 비트 동일까지 — 수송 연산자의 검증 스위트로서 강력하다.

**폐포 선택들의 타당성.** 전 원소 공통 단일 D₀: "원소 원장에는 화학종 D가 정의되지 않는다(추측=발명)"는 진단이 정확하고, 등속 이동은 국소 전기중성을 근사 보존하는 소박한 폐포다. 문헌 정합도 좋다 — 순수수(탈이온수) 용출에서는 Fick 확산과 Nernst–Planck 전기확산의 예측 차이가 무시할 수준이고, 강한 이온성 용액(질산암모늄, 폐기물 용액)에서 2–4배 괴리가 난다 [4]. 즉 현행 폐포는 봉인계·순수수 문제에 적합하고, NP 영전류 사영을 v2 경로로 문서화한 것은 옳은 우선순위다(NP 기반 시멘트 RT 모델은 확립된 계보가 있다 [5][6]). 겔수 전도-무저장은 후기 재령에서 용질 저장용량을 약간 과소평가해 농도 응답을 빠르게 만드는 방향의 단순화지만, "cluster_inventory = 모세관 용액 원장"이라는 기존 계약과 정합적이며 문서화되어 있다. D_rel 피드백 금지(그래프가 이미 기하 저항)는 이중 계상을 정확히 짚었다.

**알려진 근사 2건(문서 보강 권고).** 첫째, TPFA 전달률 T=D₀·G/p는 도메인 중심 간 거리를 타일 피치 p로 고정하는데, 클러스터∩타일 도메인이 타일의 구석 슬리버일 때 이 가정이 국소적으로 O(1) 오차를 낼 수 있다. 앵커 2의 정확성은 "타일별 상수장" 전제 위의 것이므로, 이 한계를 PRD §4.6.2에 한 줄 명시하고 타일 크기를 해상도 노브로 쓰는 현행 방침을 유지하면 된다(혼합셀 방법의 공통 근사다). 둘째, Lie T→R 1차 분할은 SNIA 표준이고 [7] "화학 폐포 오차가 지배" 논거도 수긍하지만, W3 이후 클로깅(공극 폐색)형 시나리오에서는 분할 오차와 시간해상도가 민감해진다는 것이 변동 공극률 RT의 교훈이므로 [8], 그 국면에서 dt 정책을 재점검할 것.

**작은 설계 특이점 2건(결함 아님).** 도메인 id 순 순차 용량 청구는 결정론을 확보하는 대신 용량 희소 국면에서 저-id(원점 쪽 타일) 도메인에 성장을 우선 배정하는 공간 편향을 잠재적으로 갖는다 — 근포화 클러스터에서만 발현하므로 동결 이벤트 카운트로 관찰하면 충분하다. trace-water 판정의 모드 C 재앵커(전역 총수 → 습윤 도메인 평균)는 미세 분할에서의 전역 동결을 막기 위해 필요했고 합리적이지만, 판정 기준이 분할 해상도에 약결합되므로 동결 도메인 수를 진단 지표로 유지하는 게 좋다(현행 metrics에 이미 있음).

### 2.4 산출물 실측 확인

모드 B 데모(168 h)의 sanity band: 총 클링커 알파 0.355@24h / 0.614@168h(밴드 내), 주용액 pH 12.75–13.10(밴드 내), 모세관 공극률 단조 감소, 화학수축 0.081 mL/g warn(밴드 0.03–0.08의 상단 근접 — FA/번들 몰부피 기원으로 PRD가 이미 규명한 영역). 미량수 포켓 pH 이상치 7건(최대 15.3)은 §6.3이 "진단 노이즈, 주용액 분리 판정"으로 이미 처리한 범주다 — pH 15는 물리적으로 무의미한 값이므로 이 분리 판정 방식이 옳다. RT-W2의 앵커 재고정에서 dense_hash 불변(물리 무변경)을 실측으로 입증하고 full_hash 이동 원인(boundary_water_mol 해시 편입)을 특정한 방식은 회귀 관리의 모범이다.

## 3. 남은 업그레이드의 타당성

### 3.1 RT-W2b — GEM 경제 (dirty flag · 연령 상한 · 호출 예산)

설계가 문헌 관행과 정확히 일치한다. "조성 드리프트가 임계 이하인 반응기의 평형을 미룬다"는 것은 GEMS3K가 상정한 결합 방식("조성·T·P가 변한 제어부피만 평형 호출") [9] 그 자체이고, 저장소 시뮬레이션 쪽의 adaptive-tolerance 스킵(변화율 5% 이하 셀의 지화학 생략으로 3–15배 가속) [12], on-demand 학습 [11], 서로게이트+질량수지 런타임 검증 [13][18] 계열과 비교하면 **스킵 시에도 원소는 정확하고 근사는 상 조합의 갱신 시점뿐이라는 점에서 오히려 보수적(안전한) 편**이다. 강제 범주(첫 접촉·리매핑·경계 결합·동결), lexsort(−드리프트, id) 무기아 선별, 경제 스냅샷의 체크포인트 동승(재시작 비트 동일 유지)까지 — 침묵 노화를 막는 장치가 계획에 이미 들어 있다.

리스크 1건: 스킵 동안 방출 원소가 인벤토리에 직가산되므로, 재평형(해동) 시점의 스냅샷 델타가 커져 공간충전·물 재배치 동결 게이트를 때릴 확률이 올라간다. dt에 비례하지 않는 델타라 거부가 아닌 동결로 처리되는 현행 설계가 안전판이긴 하나, 동결 빈발은 곧 수화 정체다. dirty_rtol을 낮게(≤1e-3) 시작하고, 계획된 지표(n_deferred, max_r_deferred)에 "동결 이벤트와의 상관"을 한 줄 추가해 감시하기를 권한다.

### 3.2 RT-W3 — 경계 저수조

고스트 Dirichlet 노드 + 반셀 전도 결합(T_AR=2·D₀·G_AR/p)은 리포트 네트워크 솔버의 Dirichlet 규칙과 동일한 표준 처리이고, 플럭스를 `boundary_exchanged_elements`로 부호 계상해 §6.1 폐합 항등식에 선반영해 둔 것(원장 변경 0)은 좋은 선행 설계다. 물 교환을 유보하고 용질만 교환하는 것도 포화 용출 문제의 정의와 정확히 부합한다.

과학적 유의점 셋. 첫째, 고정 조성 배스는 "연속 갱신(무한 희석) 배스"의 극한으로, 정적/주기 갱신 배스 실험보다 공격적인 경계조건이다 — 초기 구현으로 타당하지만 검증 비교 시 이 성격을 문서화해야 한다. 둘째, morphology/dissolution의 주기 랩 잔존 불일치(용질만 비주기)는 노출면 인접의 고체 이벤트가 반대면과 위상적으로 이어질 수 있음을 뜻한다 — W3 착수 전 설계 리뷰 항목으로 이미 잡혀 있으니, 그 리뷰에서 "열화 전선 폭 대비 RVE 깊이"의 정량 허용 기준을 정하면 된다. 셋째, 규모 실행 가능성은 오히려 유리하다: 순수수 용출의 CH 전선은 √t로 전진하며 [14][15] mm/√yr 오더이므로, 32–64 µm RVE는 노출 수 시간~수일 안에 관통된다 — 현행 런타임(28d 런 15분~5시간)으로 충분히 도달 가능한 검증 창이다. 다만 자기유사(√t) 구간이 짧아 계수 추출 시 초기 천이의 영향에 주의해야 한다.

### 3.3 RT-W4 — 검증 응용 후보

후보 간 우선순위까지 포함해 평가하면 다음과 같다.

**② Ca 용출 프로파일 — 최적 후보.** 현행 물리(포화·용질 확산·국소평형)와 정합하고, 문헌 앵커가 가장 풍부하다: CH 용해 전선의 √t 전진과 확산 지배 [14], 순수 페이스트 용출의 정량 모델 [15]. 단일 D₀ 폐포도 순수수 용출에서는 NP 대비 오차가 무시할 수준 [4]. 단, 질산암모늄 가속 용출 재현은 NP 없이는 2–4배 괴리가 나므로 [4] 순수수/저이온 배스로 한정할 것. 흥미로운 보너스: 평형 전용 pore-scale 모델이 후기 용출에서 실험과 갈라지는 원인으로 C-S-H 용해 동역학 부재가 지목되는데 [16], TINN은 모드 B의 τ가 정확히 그 노브다 — 용출 응용에서 모드 B+C 합성의 차별성이 생길 수 있다.

**① Deschner FA 블렌드 조성 zoning — 타당, W2b 의존.** E2의 클러스터 Ca/Si 관측치·지도와 자연스럽게 이어지는 목표다. 다만 zoning의 공간 스케일이 타일 해상도보다 가늘면 평활화되므로 미세 분할이 필요하고, 그것이 곧 GEM 호출 폭증이므로 W2b(경제)가 선행 조건이라는 PRD의 순서 판단이 옳다. `scm_logistic_override` 포트 필요성도 정확히 식별되어 있다.

**③ 탄산화 전선 — 조건부 타당.** 용존 탄산 경계(수중/지하수형 탄산화)로 한정하면 현행 물리로 접근 가능하고 pore-scale 선례도 있다 [16]. 대기 탄산화(불포화, 기체 CO₂ 확산)는 건조/RH가 범위 밖인 이상 불가능하며, PRD가 "수송 전 gas 분리 선행"을 명시한 것은 옳은 예지다. 추가로: 탄산화-용출 결합은 방해석 침전에 의한 공극 폐색(클로깅)으로 SNIA 계열에 가장 도전적인 시나리오이므로 [8][16], 세 후보 중 마지막에 두는 것이 합리적이다. 클로깅 국면에서는 전도장이 스텝 시작 스냅샷 고정(1스텝 지연)이라는 점도 dt 정책과 함께 재점검할 것.

### 3.4 문서화된 장기 경로

종별 Nernst–Planck 영전류 사영은 옳은 다음 단계다 [4][5][6]. 구현 시 "원소 원장 → 화학종 농도" 매핑이 필요해지므로(스페시에이션은 GEMS 응답에 이미 있음), 원소 상태를 유지한 채 교환 연산자만 종 공간에서 푸는 설계가 자연스러울 것이다 — v2 시점의 설계 리뷰 항목으로 남겨두면 된다.

## 4. 권고 (우선순위순)

첫째, **모드 B의 Δt–τ 관계에 검증 또는 경고를 추가하라.** 현행 f=min(1, Δt/τ)는 Δt ≥ τ인 스텝에서 f=1로 조용히 전평형에 회귀한다. τ가 dt_initial보다 작게 설정된 config는 "모드 B를 켰지만 아무것도 제한하지 않는" 상태가 될 수 있는데, 이는 이 프로젝트가 스스로 세운 침묵 무효과 금지 원칙(τ 없는 번들 거부와 동일 논리)과 긴장 관계다. config 검증에서 τ ≤ dt_initial이면 거부 또는 경고를 권한다.

둘째, **TPFA p-거리 근사의 한계를 PRD §4.6.2에 한 줄 명시하라.** "부분 타일 도메인(슬리버)에서 중심거리 p 가정은 국소 O(1) 오차 가능, 타일 크기가 해상도 노브"라는 취지면 충분하다. 앵커 2의 정확성 전제(타일별 상수장)와 실제 적용 영역의 차이를 문서가 알고 있음을 남기는 것이다.

셋째, **W2b에서 지연-해동 침전 버스트를 지표로 감시하라.** 계획된 n_deferred/max_r_deferred에 더해, 경제 활성 런에서 동결 이벤트(공간충전·물 재배치) 카운트가 dirty_rtol·eq_max_age와 어떻게 움직이는지 1회 실측해 기본값 선정 근거로 남길 것.

넷째, **W3 설계 리뷰에서 두 가지를 정량화하라.** (a) 고정 조성 배스 = 연속 갱신 극한이라는 경계조건의 의미(검증 비교 시 실험 배스 유형과의 대응), (b) morphology/dissolution 주기 랩 왜곡의 허용 기준(열화 전선 폭 vs RVE 깊이).

다섯째, **W4는 ② Ca 용출(순수수)을 첫 응용으로 하라.** 문헌 앵커가 가장 풍부하고 현행 폐포와의 정합이 가장 좋다. 이때 D₀ 선택(0.8~5.3×10⁻⁹ m²/s 범위)의 민감도를 해상도 사다리처럼 1회 실측·표기하면 단일 D₀ 폐포의 불확실성이 정직하게 드러난다.

여섯째(선택), **f=1−exp(−Δt/τ) 대안형은 채택하지 않아도 무방하다.** 현행 선형 f도 1차 정확하고 Δt≪τ 운용에서 차이가 없다. 다만 향후 τ를 실측에 보정할 때 dt 의존성이 문제가 되면, 지수형은 τ≤dt에서도 float에서 정확히 1.0이 되므로 비트 동일 게이트를 유지한 채 교체 가능하다는 점만 기록해 둘 것.

일곱째, **결정론·앵커·폐합 게이트 문화를 그대로 유지하라.** 해석해 앵커, 비트 동일 극한 게이트, 구성적 보존(플럭스 반대칭, "빼는 수치=먹인 수치"), dense_hash로 물리 불변을 입증하는 관행은 이 분야 연구 코드에서 드물게 높은 수준이며, 이 프로젝트의 가장 큰 공학적 자산이다.

## 5. 판정 요약표

| 항목 | 판정 | 근거 요약 |
|---|---|---|
| 모드 B 물리(τ 교환) | 타당 | C-S-H 재평형 수개월~수년 [1][2], 결정상 τ=0 기본 옳음 |
| 모드 B 수치(f=Δt/τ) | 타당(1차) | 교환 ODE의 일관 이산화; Δt≥τ 침묵 회귀만 주의 |
| 모드 C 구조(도메인+그래프) | 타당 | 혼합셀 RT + GEM OS 결합의 정석 [9][10] |
| 모드 C 수치(BE 플럭스형) | 타당 | 무조건 안정·M-행렬 양성·구성적 보존; 해석 앵커 검증 확인 |
| 단일 D₀ 폐포 | 조건부 타당 | 순수수 OK, 이온성 매질 2–4배 오차 [4]; NP는 문서화된 경로 |
| Lie T→R 분할 | 타당(1차) | SNIA 표준 [7]; 클로깅 국면만 주의 [8] |
| RT-W2b GEM 경제 | 타당 | GEMS3K 설계 의도 [9]·adaptive skip [12]과 정합, 서로게이트 대비 보수적 |
| RT-W3 경계 저수조 | 타당 | 표준 Dirichlet 반셀 결합; 배스 의미·주기 랩 기준만 정량화 필요 |
| RT-W4 ② 용출 | 타당·최우선 | √t 전선 문헌 앵커 [14][15], 규모 창 실행 가능 |
| RT-W4 ① FA zoning | 타당·W2b 의존 | E2 관측치와 연결; 타일 해상도가 관건 |
| RT-W4 ③ 탄산화 | 조건부 타당 | 용존 탄산 한정 [16]; 대기 탄산화는 범위 밖 유지 옳음 |

## 참고문헌

[1] [Kinetics of Al uptake in synthetic calcium silicate hydrate (C-S-H)](https://consensus.app/papers/details/822a7a784b1d576c829687cf80fb3bd0/?utm_source=claude_desktop) (Yan et al., 2023, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2023.107250)
[2] [The effect of equilibration time on Al uptake in C-S-H](https://consensus.app/papers/details/1b39113502765755bd86d35d07f82dc5/?utm_source=claude_desktop) (Barzgar et al., 2021, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2021.106438)
[3] [Alteration of nanocrystalline C-S-H at pH 9.2 and room temperature](https://consensus.app/papers/details/bf362982e5335c7b887ba780fcb3954b/?utm_source=claude_desktop) (Marty et al., 2015, Mineralogical Magazine, DOI: 10.1180/minmag.2015.079.2.20)
[4] [Influence of multi-species solute transport on modeling of hydrated Portland cement leaching in strong nitrate solutions](https://consensus.app/papers/details/74ed3bb4776454bc9d528949b38d91e6/?utm_source=claude_desktop) (Arnold et al., 2017, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2017.06.002)
[5] [Multi-species ionic diffusion in concrete with account to interaction between ions in the pore solution and the cement hydrates](https://consensus.app/papers/details/117799d5323d57bb8ad182854cdbf0e2/?utm_source=claude_desktop) (Johannesson et al., 2007, Materials and Structures, DOI: 10.1617/s11527-006-9176-y)
[6] [Solute transport solved with the Nernst-Planck equation for concrete pores with 'free' water and a double layer](https://consensus.app/papers/details/fe01b5945a545739a9eb9bf91ecb9fca/?utm_source=claude_desktop) (Appelo, 2017, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2017.08.030)
[7] [Operator-splitting procedures for reactive transport and comparison of mass balance errors](https://consensus.app/papers/details/4956024910cd5850b4f031b20c426294/?utm_source=claude_desktop) (Carrayrou et al., 2004, Journal of Contaminant Hydrology, DOI: 10.1016/s0169-7722(03)00141-4)
[8] [Operator-splitting-based reactive transport models in strong feedback of porosity change](https://consensus.app/papers/details/44e5328679385a37abb8a762c4380b4c/?utm_source=claude_desktop) (Lagneau & van der Lee, 2010, Journal of Contaminant Hydrology, DOI: 10.1016/j.jconhyd.2009.11.005)
[9] [GEM-Selektor geochemical modeling package: revised algorithm and GEMS3K numerical kernel for coupled simulation codes](https://consensus.app/papers/details/c93218a4fcf95fe7a4bdb5d0404565a4/?utm_source=claude_desktop) (Kulik et al., 2012, Computational Geosciences, DOI: 10.1007/s10596-012-9310-6)
[10] [OpenGeoSys-Gem: A numerical tool for calculating geochemical and porosity changes in saturated and partially saturated media](https://consensus.app/papers/details/f87213eef803599db6dd8bc92a69fb38/?utm_source=claude_desktop) (Kosakowski & Watanabe, 2013, Physics and Chemistry of the Earth, DOI: 10.1016/j.pce.2013.11.008)
[11] [Accelerating Reactive Transport Modeling: On-Demand Machine Learning Algorithm for Chemical Equilibrium Calculations](https://consensus.app/papers/details/e4e6d8f842b8500c85e01fe68eff0904/?utm_source=claude_desktop) (Leal et al., 2020, Transport in Porous Media, DOI: 10.1007/s11242-020-01412-1)
[12] [Enhancement of Simulation CPU Time of Reactive-Transport Flow in Porous Media: Adaptive Tolerance and Mixing Zone-Based Approach](https://consensus.app/papers/details/a03afbfd946458faa3f37ed09bbbab0e/?utm_source=claude_desktop) (Bordeaux-Rego et al., 2022, Transport in Porous Media, DOI: 10.1007/s11242-022-01789-1)
[13] [Speeding Up Reactive Transport Simulations in Cement Systems by Surrogate Geochemical Modeling](https://consensus.app/papers/details/a9a32d0b477653bba27cf1cfeff5fa91/?utm_source=claude_desktop) (Laloy & Jacques, 2021, Transport in Porous Media, DOI: 10.1007/s11242-022-01779-3)
[14] [Change in pore structure and composition of hardened cement paste during the process of dissolution](https://consensus.app/papers/details/064efedaaf335067bdee6b79afb50fa2/?utm_source=claude_desktop) (Haga et al., 2005, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2004.06.001) 및 [Effects of porosity on leaching of Ca from hardened OPC paste](https://consensus.app/papers/details/ea72254803c15c59a7930001429f97bb/?utm_source=claude_desktop) (DOI: 10.1016/j.cemconres.2004.06.034)
[15] [Modelling of leaching in pure cement paste and mortar](https://consensus.app/papers/details/f06df2b672ad5b7c80e0c5c668a8c379/?utm_source=claude_desktop) (Mainguy et al., 2000, Cement and Concrete Research, DOI: 10.1016/s0008-8846(99)00208-2)
[16] [A multi-level pore scale reactive transport model for the investigation of combined leaching and carbonation of cement paste](https://consensus.app/papers/details/ab6167feac2254d196fd72e33587bcd2/?utm_source=claude_desktop) (Patel et al., 2021, Cement and Concrete Composites, DOI: 10.1016/j.cemconcomp.2020.103831)
[17] [Computer simulation of the diffusivity of cement-based materials](https://link.springer.com/article/10.1007/BF01117921) (Garboczi & Bentz, 1992, Journal of Materials Science — C-S-H 상대확산 1/400의 출처 계보)
[18] [DecTree v1.0 – chemistry speedup in reactive transport simulations: purely data-driven and physics-based surrogates](https://consensus.app/papers/details/7be8b1e4e1585331921a75716ab28dd7/?utm_source=claude_desktop) (De Lucia et al., 2021, Geoscientific Model Development, DOI: 10.5194/gmd-14-4713-2021)
[19] [A new view on the kinetics of tricalcium silicate hydration](https://consensus.app/papers/details/2bb9328fe0ef5e3494ceba0612bb55c1/?utm_source=claude_desktop) (Nicoleau & Nonat, 2016, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2016.04.009)
[20] [A thermodynamic model of dissolution and precipitation of calcium silicate hydrates](https://consensus.app/papers/details/465783fb36c4596295476c5732e4df48/?utm_source=claude_desktop) (Sugiyama & Fujita, 2006, Cement and Concrete Research, DOI: 10.1016/j.cemconres.2005.09.002)

---

*이 문서는 외부 검토 산출물이며 저장소 문서 정책(PRD 원칙 5: 문서 2개)의 대상이 아니다 — 저장소 밖으로 옮기거나 삭제해도 무방하다.*
