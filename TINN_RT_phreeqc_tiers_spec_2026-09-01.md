# TINN v4.0/RT — PHREEQC 보조 커널 통합 설계 (Tier 0 / Tier 1)

작성 2026-09-01. PRD.md의 부속 설계문서(구현 시 PRD 해당 절에 병합). 이 문서는
구현 세션에 단독 공급 가능하도록 배경·계약·수식·스키마·게이트를 자립적으로 담는다.
결정 경위 요약: 주 화학 엔진은 **xGEMS 유지**(고용체·Cemdata 정합성이 방어선),
PHREEQC는 엔진 교체가 아니라 **역할이 한정된 보조 커널**로만 들어온다 —
Tier 0은 PHREEQC를 런타임이 아닌 **데이터 소스**(종별 확산계수)로, Tier 1은
**수착 전용 연산자**(IPhreeqc in-process)로 쓴다. PhreeqcRM 위임(Tier 2)은
비채택(재평가 트리거: 엔진 전환 결정 / 화학 병렬화가 지배 비용 / BMI 결합 필요).

## 0. 지위·전제·불변 원칙

- **전제 게이트 G0**: `scripts/spike_phreeqc_crosscheck.py`의 0D 교차검증
  (GEMS 번들 vs 공식 Cemdata18 PHREEQC 배포본
  `gems_bundles/PHREEQC-cemdata18/cemdata18.dat`, PROVENANCE.md 참조)이
  판정 기준(README_phreeqc_crosscheck.md)을 통과해야 **Tier 1** 구현을
  시작한다. **Tier 0은 G0와 무관**(PHREEQC 열역학을 쓰지 않으므로) —
  즉시 착수 가능.
- **침묵 무효과 금지·비트 동일 게이트**(프로젝트 헌법, transport 섹션 선례):
  신설 config 섹션이 **부재하면 현행 엔진과 비트 동일**. 활성 시에도 기존
  앵커·체크포인트 재시작 비트동일·§6.1 불변식은 전부 유지된다.
- **원소 원장 유일 권위**: 두 Tier 모두 원소 mol 원장(`state.py`,
  `ELEMENT_IDS = (Ca,Si,Al,Fe,S,Na,K,Mg,C,H,O)`, 전하 `Zz`)을 소유하지
  않는다. 종 공간 계산은 연산자 내부에서만 존재하고, 원장에는 원소 델타
  (플럭스/수착량)로만 반영된다.
- **모듈 상한 16 (현재 16, 도달)**: 이 작업은 **신규 모듈을 만들지 않는다**.
  Tier 0은 `transport.py` 내부 확장, Tier 1은 `backend.py` 동거
  (`SorptionOperator` 클래스). 도우미 데이터 생성기는 `scripts/`(모듈 아님).

## 1. 메커니즘 담당표 (이중계상 방지 — 규칙 1)

메커니즘당 권위 엔진은 정확히 하나. 구현은 이 표를 config 검증에 코드로 박는다
(위반 조합은 **거부**, 경고가 아니라 에러 — τ 없는 번들 거부와 같은 논리).

| 메커니즘 | 권위 | 비고 |
|---|---|---|
| 상 조합·용해/침전·고용체 (C-S-H 포함) | GEMS (현행) | 변경 없음 |
| Na/K의 C-S-H **구조적** 흡수 | GEMS(CNASH 번들) | CNASH 번들 활성 시 Tier 1의 알칼리 EXCHANGE **금지** |
| Cl의 AFm 결합 (Friedel염) | GEMS(번들에 있을 때) | 열역학적 결합 |
| Cl·SO4의 C-S-H **표면** 수착 | Tier 1 (PHREEQC SURFACE) | 광물 침전 전면 비활성 상태로만 호출 |
| 종별 확산 플럭스 가중 (전기적 결합) | Tier 0 (자체 구현) | PHREEQC는 D_w 데이터 출처일 뿐 |
| 이온강도 보정·활동도 | 각 엔진 내부 | 교차 주입 금지 |

주의: 현행 기본 번들이 CNASH이므로 기본 조합에서 Tier 1이 다룰 수 있는 것은
Cl/SO4 표면 수착뿐이다. 알칼리 수착을 Tier 1로 실험하려면 CSHQ(PC) 번들 +
`sorption.alkali_exchange=true`의 명시 조합만 허용(검증기에서 강제).

## 2. Tier 0 — 종별 Nernst–Planck 영전류 사영 (RT-P0)

### 2.1 목표와 위치

모드 C 도메인 그래프 확산(`transport.py` + `engine.py`의 BE 플럭스형 교환
연산자)의 **스칼라 공통 `d0_m2_s`**를, 수용액 스페시에이션과 종별 자기확산계수
D_i에 기반한 **원소별 유효 확산계수**로 대체하는 옵션. 검토서(§2.4, [4])가
지적한 "이온성 매질 단일 D₀ 2–4배 오차"를 해소하고, RT-W4 ② Ca 용출 검증의
D₀ 민감도 문제를 원리적으로 닫는다. PHREEQC 코드는 실행되지 않는다.

### 2.2 데이터: 종별 D_w 테이블

- 생성기 `scripts/make_species_dw_table.py`(신규, 모듈 아님): USGS 배포
  `phreeqc.dat`의 각 SOLUTION_SPECIES `-dw` 값(25 °C 자기확산계수, m²/s)과
  전하를 파싱해 `gems_bundles/species_dw/species_dw.json`으로 벤더링.
  필드: `{species, z, dw_25C_m2_s, source_line}` + 파일 전체 sha256 +
  출처(phreeqc.dat 버전) 헤더. PROVENANCE.md 동봉.
- **명명 매핑**: GEMS 수용액 종 이름(DCH) ↔ phreeqc.dat 종 이름은 다르다
  (예: `AlO2-`↔`Al(OH)4-` 표기, `CaSiO3`류). 매핑은 `species_dw.json`의
  `aliases` 필드로 명시하고, **매핑 없는 종은 기본값으로 조용히 대체하지
  않는다** — config `transport.species.default_dw_m2_s`가 명시된 경우에만
  그 값을 쓰고 리포트에 목록을 남긴다(없으면 에러). 침묵 무효과 금지.
- 온도 의존: 1차 구현은 Stokes–Einstein 보정
  `D(T) = D(298.15) · (T/298.15) · (η(298.15)/η(T))` (물 점도표 내장,
  0–100 °C). 리포트에 보정 계수 기록.

### 2.3 GEMS 응답 확장 (전제 작업)

`gems.py` 워커 응답에 수용액 종별 mol을 추가한다:
`aqueous_species_mol: Dict[str, float]`(aq 상의 DC별 mol; xGEMS
ChemicalEngine의 종 벡터에서 aq 상 슬라이스) + `species_charge: Dict[str,
float]`(DCH의 Zz 행). `GemsResult`에 필드 추가, 프로토콜 버전 키 증가,
구버전 응답 수신 시 Tier 0 활성 상태면 에러(비활성이면 무시 — 비트 동일 유지).
0D 프로브에도 노출해 스파이크 비교표에서 스페시에이션 차이를 직접 검증할 수
있게 한다.

### 2.4 수식과 이산화

도메인 d의 종 농도 c_i = n_i / V_w,d (GEMS 응답의 aq 종 mol / 수용액 부피).
Nernst–Planck 영전류(무전류) 조건에서 에지 (a,b)의 종 플럭스:

```
J_i = -D_i ∇c_i + z_i D_i c̄_i · Φ,
Φ   = ( Σ_j z_j D_j ∇c_j ) / ( Σ_j z_j² D_j c̄_j )     # Σ_i z_i J_i = 0 보장
```

∇c_i는 현행 TPFA p-거리 근사(§4.6.2와 동일 한계·동일 문서화), c̄_i는 에지
조화/상류 평균(현행 면 혼합 규칙 β 재사용). 원소 플럭스는
`F_el = Σ_i ν_{el,i} J_i` (ν는 DCH 화학량행렬 — 하드코딩 금지, 워커의
`species_elements`에서).

**이산화 전략(단계형)**:

- **RT-P0a (진단 전용)**: 물리 변경 없음. 매 평형 후 위 식으로 에지별
  원소 유효 확산계수 `D_eff[el] = |F_el| / |∇(원소농도)|`를 계산해
  리포트에만 기록(현행 d0와의 비율 히스토그램). 비트 동일 게이트 자명.
- **RT-P0b (본 구현)**: 스텝 시작 시점 스페시에이션을 **동결**(Lie 분할과
  같은 1차 정합)하고, 동결된 c_i, Φ 가중으로 에지·원소별 유효 전도도
  `g_edge[el]`를 만들어 **기존 BE 플럭스형 솔버를 원소별로 그대로 재사용**
  한다. 얻는 성질: 무조건 안정·원소별 M-행렬 양성·플럭스 반대칭(구성적
  보존) 전부 유지. 전기적 결합은 g_edge[el] 안의 Φ 항으로 1차 반영되고,
  스텝 내 비선형 재결합은 비범위(문서화).
  - 주의(양수성): Φ 항이 g_edge[el]<0을 만들 수 있는 조성에서는 해당
    에지·원소의 Φ 기여를 0으로 클램프하고 카운터를 리포트에 남긴다
    (클램프 = 순수 Fick 후퇴, 보존 훼손 없음).
- **RT-P0c (설계 리뷰 후 선택)**: 종 공간 완전 결합(교차 원소 커플링을
  행렬로) — 검토서 §3.4의 원안. P0b 실측에서 클램프율·오차가 유의할 때만.

### 2.5 config (스키마 초안)

```jsonc
"transport": {
  "domains": {
    "tile_vox": 8,
    // 기존: "d0_m2_s": 1.0e-9  ← species 블록과 상호 배타 (동시 지정 = 에러)
    "species": {
      "dw_table": "gems_bundles/species_dw/species_dw.json",
      "default_dw_m2_s": null,        // null이면 매핑 누락 = 에러
      "geometry_factor": 1.0,         // 다공성 매질 축소인자(공통), 필수 명시
      "phi_clamp_report": true
    },
    "dirty_rtol": 3.0e-3, "eq_max_age_steps": 16, "max_gem_calls_per_step": 128
  }
}
```

### 2.6 앵커·게이트 (§6.2 스타일)

1. **극한 게이트(비트 아님, 수치 항등)**: 모든 종 D_i = D₀ 동일 입력 ⇒
   Φ ≡ 0이고 원소 유효 전도도가 스칼라 경로와 상대 1e-12 이내 일치.
2. **Nernst–Hartley 해석 앵커**: 2종 전해질(예: Na⁺/Cl⁻)만 있는 합성
   스페시에이션에서 유효 염 확산계수가
   `D_salt = (z₊−z₋) D₊D₋ / (z₊D₊ − z₋D₋)` 와 일치(상대 1e-10).
   NaCl≈1.61e-9, KCl≈1.99e-9 m²/s 부근 값으로 교차 확인.
3. **전하 보존**: 매 에지 `Σ z_i J_i = 0` 기계 검증(클램프 에지는 별도 계상).
4. **보존 불변식**: 원소별 플럭스 반대칭·"빼는 수치=먹인 수치" 게이트를
   원소×에지로 확장 적용, 폐합 ≤ 현행 기준(1e-12).
5. **비트 동일**: `species` 블록 부재 시 전 테스트 스위트 비트 동일.
6. **응용 실측(RT-W4 ② 연계)**: Ca 용출 순수수 케이스에서 스칼라 D₀ 대비
   NP 전선 속도 차이를 1회 실측·표기(검토서 권고 5의 민감도 표를 대체).

## 3. Tier 1 — PHREEQC 수착 연산자 (RT-S1)

### 3.1 목표와 위치

Lie 분할을 T→R에서 **T→S→R**로 확장. S는 도메인별 순수 함수 연산자:
공극수 원소 벡터 + 물 + 흡착제 재고 → 수착/탈착 원소 델타. 구현은
IPhreeqc in-process(`phreeqpython`, 스파이크와 동일 스택), DB는 벤더링된
`cemdata18.dat`. **EQUILIBRIUM_PHASES·SOLID_SOLUTIONS 사용 금지** — 상
조합 권위는 GEMS(§1). SURFACE(비정전 또는 DDL, config로)만 사용.

### 3.2 계약 (backend.py 동거)

```python
class SorptionOperator:            # backend.py에 추가 (신규 모듈 금지)
    operator_id = "phreeqc_surface"
    def sorb(self, aqueous_elements: np.ndarray,   # (E,) 도메인 공극수
             water_mol: float,
             sorbent_sites_mol: float,             # §3.3에서 산출
             temperature_k: float) -> SorptionResult: ...

@dataclass
class SorptionResult:
    status: str                    # ok | nonconvergence(트라이얼 리젝트)
    sorbed_delta: np.ndarray       # (E,) 수착(+)/탈착(−) 원소 mol
    site_occupancy: Dict[str, float]  # 진단 전용
```

- 워커 격리는 불필요(IPhreeqc는 in-process, 로그 오염 없음)하나 DB 파일은
  GEMS 번들과 동일하게 **호출 전후 sha256 감사**(스파이크 PROVENANCE 규칙).
- 결정론: IPhreeqc 단일 스레드 호출 + 입력해시 메모이제이션(현행 GEMS 메모
  패턴 재사용).

### 3.3 흡착제 재고와 원장

- 사이트 총량: `sites_mol = Σ(C-S-H endmember mol × site_density_mol_per_mol)`
  — endmember mol은 **GEMS parcel의 E1 원장**(`Parcel.endmember_mol`)에서.
  `site_density_mol_per_mol`(endmember별)과 표면적/용량 파라미터는 config
  필수 명시, 기본값 없음(문헌값을 쓰더라도 config에 적는다 — d0 전례).
- 원장 확장: 도메인별 **수착 저장소**(sorbed inventory, (E,) 벡터)를
  `state.py`에 신설. §6.1 불변식에 "수용액+고체+수착 = 전체" 폐합 추가.
  체크포인트 스키마 버전 증가; `sorption` 부재 런의 체크포인트는 무변경.
- R(GEMS) 단계 입력에서 수착 저장소는 **제외**(수착분은 평형에 안 보임).
  C-S-H가 재용해로 줄어 사이트가 감소하면 다음 S 단계에서 초과 점유분이
  탈착 델타로 반환된다(사이트 감소 → 강제 탈착, 명시적 처리).

### 3.4 config (스키마 초안)

```jsonc
"sorption": {
  "operator": "phreeqc_surface",
  "phreeqc_dat": "gems_bundles/PHREEQC-cemdata18/cemdata18.dat",
  "surface_model": "no_edl",            // "no_edl" | "ddl"
  "site_density_mol_per_mol": {"CSHQ-TobH": 0.0, "...": 0.0},  // 필수 명시
  "elements": ["Cl"],                   // 수착 허용 원소 화이트리스트
  "alkali_exchange": false              // true는 CSHQ 번들에서만 허용(§1)
}
```

검증 규칙: `elements`에 Na/K 포함 + CNASH 번들 ⇒ 에러. `sorption` 부재 ⇒
S 단계 자체가 생성되지 않음(비트 동일).

### 3.5 앵커·게이트

1. **비트 동일**: `sorption` 부재 시 전 스위트 비트 동일.
2. **널 연산자 게이트**: `site_density…` 전부 0 ⇒ 활성 상태에서도 상태
   궤적이 비활성과 수치 항등(플로트 연산 경로 차이만 허용, 원장 값 동일).
3. **등온선 앵커**: 단일 도메인·고정 pH에서 S 연산자 단독 반복 호출이
   같은 입력에 같은 출력(순수 함수) + PHREEQC 단독 배치 계산과 mol 단위
   일치(스파이크 하니스 재사용).
4. **폐합**: 수착 포함 원소 수지 ≤ 1e-12(복셀 항등식 기준 유지),
   "빼는 수치=먹인 수치"를 S 단계에도 적용.
5. **G0 재확인**: Tier 1 첫 활성 런 전, 사용 DB(cemdata18.dat sha256)로
   0D 교차검증 결과를 리포트에 링크(감사 추적).

## 4. 구현 순서 (마일스톤)

| 단계 | 내용 | 게이트 |
|---|---|---|
| RT-P0a | dw 테이블 생성기 + 워커 스페시에이션 노출 + 진단 리포트 | 비트 동일 자명, 스파이크에 스페시에이션 비교 추가 |
| RT-P0b | 원소별 유효 전도도 BE 통합 (species 블록) | §2.6 앵커 1–5 |
| RT-W4② 재실측 | Ca 용출 스칼라 vs NP 비교 | §2.6 앵커 6 |
| RT-S1a | SorptionOperator + 원장/체크포인트 확장 (S 단계 미접속) | §3.5 앵커 1–3 |
| RT-S1b | 엔진 T→S→R 접속 + 담당표 검증기 | §3.5 앵커 4–5 |
| RT-P0c | (선택) 완전 결합 NP — P0b 실측 후 설계 리뷰 | 별도 문서 |

각 단계는 독립 커밋·독립 게이트. P0와 S1은 서로 의존 없음(병행 가능).

## 5. 비범위 (명시)

PhreeqcRM/BMI 결합, 주 엔진 교체, PHREEQC 열역학의 R 단계 사용, 대기
탄산화, 온도 의존 dw의 정밀 모델(Stokes–Einstein 초과), C-S-H 이외
흡착제(AFm 표면 등), 전기영동/이류. 재평가 트리거는 서두에 기록된 세 가지.

## 6. 참조 파일 앵커 (구현 세션용)

- `src/tinn/transport.py` — 도메인 그래프·클러스터 라벨링 (P0 수정 지점)
- `src/tinn/engine.py` — Lie 분할 오케스트레이션 (S 단계 접속 지점)
- `src/tinn/backend.py` — ReactionBackend/Parcel 계약 (SorptionOperator 동거)
- `src/tinn/gems.py` — 워커 프로토콜 (`equilibrate_elements`, `GemsResult`;
  §2.3 확장 지점), 번들 sha256 감사 패턴
- `src/tinn/state.py`, `src/tinn/ledger.py` — 원장·§6.1 불변식 (수착 저장소)
- `src/tinn/config.py` — 스키마+해시 (`transport.species`, `sorption` 추가)
- `scripts/spike_phreeqc_crosscheck.py` — G0 하니스·PHREEQC 호출 예제
- `gems_bundles/PHREEQC-cemdata18/` — 벤더링 DB + PROVENANCE 규칙
- `TINN_RT_scientific_review_2026-08-20.md` §2.4·§3.4 — 단일 D₀ 한계와
  NP 경로 근거, [4][5][6] 문헌 앵커
