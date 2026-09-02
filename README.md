# MicroGEM3D — reactive-transport 선 (`rt`, v4.0/RT)

배합(C3S, OPC 4상, SCM 블렌드), PSD, w/c, 온도, 시간 스케줄을 입력받아 질량 보존되는
시간 의존 3D 미세구조(상 분포·물 분배·공극·수송 지표)를 결정론적·재시작 가능하게
출력하는 시뮬레이션 플랫폼. 설계와 범위는 [PRD.md](PRD.md)가 유일한 기준 문서다.

**버전 계보**: MicroGEM3D `main` = v3(transport 도입 전, 논문판, 태그 `v3.0-paper`),
이 브랜치 `rt` = v4(reactive transport, 태그 `v4.0-rt-tier1` …). 개발 중 이름은
TINN이었고 패키지 이름공간(`src/tinn/`, `TINN_GEMS_PYTHON`)은 재현성 계약상 유지한다.

이 브랜치는 플랫폼 v2(f7be88b)에서 분기한 **v4.0/RT 개발 트리**다 (PRD §1.4 v4.0/RT,
§4.6): config `transport` 섹션으로 ① 속도제한 재평형(모드 B, `exchange_tau_h`) ②
서브클러스터 평형 도메인 + 도메인 그래프 확산(모드 C, `domains`) ③ 경계 저수조
(RT-W3, `boundary`)를 선택한다. `transport` 부재 시 현행 엔진과 비트 동일하게 동작한다.

```jsonc
// 예: 모드 B+C 합성 (RT-W2부터 유효 — 필드는 PRD §4.6)
"transport": {
  "exchange_tau_h": 2160.0,          // C-S-H 교환시간 90일; 생략 = 전평형
  "domains": {
    "tile_vox": 8,                   // 격자 크기의 약수; = 격자 크기 ⇒ 현행과 동일
    "d0_m2_s": 1.0e-9,               // 공통 유효 확산계수 (필수 명시, 기본값 없음)
    "dirty_rtol": 3.0e-3,            // 0 = 매 스텝 전 도메인 평형 (순수 C 기준)
    "eq_max_age_steps": 16,
    "max_gem_calls_per_step": 128
  }
}

// Tier 0 (RT-P0, PRD 4.6.4): d0 대신 종별 NP 원소별 유효 전도도.
// d0_m2_s와 상호 배타 (diagnostics_only: true면 d0 유지 + 리포트만).
"transport": {
  "domains": {
    "tile_vox": 8,
    "species": {
      "dw_table": "gems_bundles/species_dw/species_dw.json",
      "default_dw_m2_s": 1.0e-9,     // null = 미매핑 종 존재 시 거부
      "geometry_factor": 1.0,        // 필수 명시 (d0 전례)
      "phi_clamp_report": true
    }
  }
}
```

## 요구 사항

- Python ≥ 3.11, `numpy`, `pydantic` (테스트는 `pytest`)
- GEMS 백엔드(`gems3k`)는 xgems(pybind11)가 설치된 별도 인터프리터가 필요
  (config `gems_worker_python` 또는 환경변수 `TINN_GEMS_PYTHON`; 예:
  `miniforge3/envs/py313-xgems`). 미설치 시 합성 백엔드의 전 기능은 정상 동작.

## 빠른 시작

```powershell
cd "TINN platform v2"
$env:PYTHONPATH = "src"

# config 검증
py -3 -m tinn.cli validate-config examples\c3s_32.json

# C3S 32³ 런 (alpha 0→0.35, 체크포인트 + summary.json 생성, 약 1분)
py -3 -m tinn.cli run examples\c3s_32.json --out runs\c3s

# 체크포인트에서 재시작 (무중단 실행과 비트단위 동일)
py -3 -m tinn.cli restart runs\c3s\ckpt_001 --out runs\c3s_restart

# 리포트 (요약 JSON + 슬라이스 PNG + 공극/수송 분석 + §6.3 판정)
py -3 -m tinn.cli report runs\c3s

# GEMS 백엔드 런 — 기본 열역학 번들은 CNASH(gems_bundles/CNASH/Test-dat.lst,
# InverseGems Test 번들 벤더링). config에서 gems_bundle_lst 생략 시 자동 사용.
py -3 -m tinn.cli run examples\opc_cnash_32.json --out runs\opc_cnash

# CSHQ(PC/Cemdata) 번들로 교체하려면 config에 명시:
#   "gems_bundle_lst": "gems_bundles/PC/PC-dat.lst"
py -3 -m tinn.cli run examples\opc_gems_32.json --out runs\opc_gems

# 테스트
py -3 -m pytest -q
```

## 리포트 옵션

```powershell
py -3 -m tinn.cli report RUN_DIR `
  --kc-constant-m2 1e-12 `        # Kozeny-Carman C 오버라이드 (기본 d_mean²/180)
  --gel-rel-diffusivity 0.0025 `  # 전도도 네트워크의 C-S-H 상대확산계수 (Garboczi–Bentz)
  --face-mixing-beta 0.0          # 목 보정 노브 (0=조화[기본]…1=부분 복셀 면에서 산술)
```

리포트 산출: 연결/고립 모세관 공극률, 부피 가중 공극 크기분포(서브복셀 슬래브 꼬리
포함), 3축 유효 상대확산계수 D_eff/D₀(전도도 네트워크, Jacobi-CG), KC 투수성,
percolation, 상 분율, 슬라이스 PNG, §6.3 sanity band.

## 해상도 지침 (FA30 28d 사다리 실측, PRD §1.3)

| 목적 | 권장 해상도 | 28일 런타임 |
|---|---|---|
| 공극률·반응도·상 조성 | 1.0 µm / 32³ | ~15분 (GEMS) |
| 연결성·크기분포·KC | 0.5 µm / 64³ | ~20분 |
| D_rel 정밀·수렴 확인 | 0.25 µm / 128³ | ~5시간 |

부피·화학 지표는 해상도 무관(<1%), 크기·수송 지표는 격자 세분으로 수렴
(D_rel의 후기 재령 격자 의존은 위상 결손 — PRD §1.3 실측 기록 참조).

## 주요 config 필드 (전체는 PRD §1.2)

- `binder.mass_fractions` — C3S/C2S/C3A/C4AF + SCM(slag/fly_ash/metakaolin/silica_fume)
- `material_psd` — 재료별 PSD 맵(키: "clinker"/SCM id; 생략 시 공용 `psd`)
- `material_shape` — 재료별 타원체 형상(반축비, 부피 정규화; 생략 시 구형)
- `rve` — grid_size 32/64/128, voxel_size_um 0.25–1.0, seed
- `chemistry` — backend `stoichiometric`|`gems3k`; `gems_bundle_lst`(기본 CNASH),
  `gems_gel_porosity`(기본 CSHQ/CNASH 0.28), `gems_worker_python`
- 지속 GEMS 워커는 기본 활성(`TINN_GEMS_PERSISTENT=0`로 비활성) + 입력해시 메모이제이션

## 진행 상태 (PRD §3 마일스톤)

- [x] **M0–M5** — 골격/보존 코어/P&K 동역학/GEMS 0D/GEMS→3D 결합/분석 리포트
      (원소 수지 ~1e-24 mol, 복셀 항등식 ≤1e-12, 체크포인트 재시작 비트동일).
- [x] **v2.1** — SCM 4종을 kinetic 상으로: InverseGems의 조성·밀도·로지스틱
      반응도·CH 가용성 보정 이식 (α_FA(360d)=0.5522 정확 일치).
- [x] **v2.2** — 전평형 재용해(gems3k snapshot: CH 소모·상 재배열이 평형에서
      발생), 재료별 입자군+개별 PSD, 구조 기반 공극 분석, 지속 GEMS 워커.
- [x] **rev.2 (2026-07-16~17)** — 겔 전도 용해(클링커 α 정체 해소: 실현 α가 P&K
      목표의 93~96%), 공간 충전 동결·오버플로 배치(장기 런 안정화, FA30 360일 완주),
      기본 번들 CNASH 채택(InverseGems와 28d 상 조성 0.5~14% 일치 교차검증),
      2-스케일 공극 분석(네트워크 D_eff + 서브복셀 크기분포), 해상도 수렴 사다리
      (32/64/128³), 재료별 타원체 형상(효과 실측: D_rel −4.4% — 구형 유지 판정),
      목 보정 노브(1d 사다리 평탄화, 28d는 위상 한계 — 기본 조화 유지).
- [x] **RT-S1a/S1b (2026-09-02)** — Tier 1 PHREEQC 수착 연산자(PRD 4.6.5):
      T→S→R 분할, config 소유 표면 화학(SO4 첫 대상), endmember 보존 사이트
      부기, §6.1 폐합에 수착 저장소 편입 + `balance_sorption`. 포맷 무단절
      (v5 예약분 활성).
- [x] **RT-S1c (2026-09-02)** — SO4 문헌 보정(PRD 4.6.5 실측 블록): Labbez
      실란올 4.8/nm² → 사이트 0.4766 mol/mol-Si(endmember별 ×Si), Divet
      1998 0.1 M NaOH 등온선을 연산자 자체로 재현해 log_K +0.50 피팅
      (`scripts/fit_so4_logk.py`), Ochs Se(VI) 밴드 sanity. 정준 스케일
      앵커를 용액×사이트 기하평균으로 보강. 28d calibrated 실측
      `runs/sorption_so4_results_calibrated.json`. 후속 실측 수정(같은 날):
      재제공 O/H 프레임의 부호를 명시적 `H2O`/`O2` 반응 항으로 원소-정확
      처리(음수 H 침묵 클립 제거, `_frame_*` 진단), 수착 저장소 리매핑을
      물 가중→흡착제(CSHQ 부피) 가중으로 교체(황산염 노출 스위치 H overdraw).
- [x] **RT-S1d (2026-09-02)** — S1-OPEN-2 황산염 트랩 아티팩트 수정: S 단계
      계에 반응기 소유 CH를 완충 평형상으로 포함(`sorption.buffer_phase`,
      opt-in), 델타를 용액↔CH 풀에 기입해 R이 재결정(PRD 4.6.5). 원인 실측:
      재제공 부분계 pH 0.72(calibrated)/−0.26(알칼리) → 연산자가 일방향 싱크;
      S1c 28d·알칼리 28d 이전 수치는 싱크 누적량으로 재해석.
- [x] **RT-S2a (2026-09-02)** — ddl 표면 모형 능력(PRD 4.6.5): config 소유
      면적·하전 반응(탈양성자화/Ca 착화, 수착과 동일 원장 기계), NaCl 담체
      분해(E3 선행), 하전 종명 파서 수정. 공동 보정 실측(`fit_so4_ddl.py`):
      GC가 NaCl 경향 부호는 재현하나 pH 사다리 정량 실패 → SO4는 no_edl
      국소 보정 유지, ddl은 정전 지배 이온용으로 보류.
- [x] **RT-P0d (2026-09-02)** — 종별 수송 + 용질 배스(PRD 4.6.4): 저수조를
      초기화 시 1회 GEMS 스페시에이션(동결, 억제-증인 보호), 배스 면 조화/
      one-sided 규칙, O/H 배스는 비트 동일. 외부 침식 케이스에 Tier 0 개방.
- [x] **RT-P0a/P0b (2026-09-02)** — Tier 0 종별 NP 영전류 사영(PRD 4.6.4):
      dw 테이블 벤더링 + 워커 프로토콜 v2(스페시에이션 노출) + 원소별 BE
      전도도(FORMAT_VERSION 5, Tier 1 수착 배열 예약 동승). 실측: OPC 공극수
      D_eff/D₀ 0.008–8.1×(p50 1.57×). 전제 G0(PHREEQC-Cemdata18 0D
      교차검증)은 `scripts/README_phreeqc_crosscheck.md`에 기록.

## 예제 config

- `examples/c3s_32.json` — 순수 C3S, tabulated alpha (§6.2 C3S 픽스처).
- `examples/opc_srm114q_32.json` — NIST SRM 114q 4상, P&K `pk_elakneswaran_2018`.
- `examples/opc_srm114q_measured_psd_64.json` — **실측 PSD**: SRM 114q의 레이저 회절
  누적곡선(NIST SP 260-166 Table 8) 원표 입력 + 격자 절단(`truncate_to_grid`).
  측정창 밖 5.2%·절단 1.6%가 geometry 리포트에 명시되고 구 가정 비표면(`ssa_est_m2_kg`)으로
  실측 Blaine(381.8)과 교차확인 가능. `cumulative` 대신 `rosin_rammler`
  {d_prime_um, n, d_min_um, d_max_um} 입력도 지원.
- `examples/opc_cnash_32.json` — OPC 4상 + **기본 CNASH 번들**(gems_bundle_lst 생략).
- `examples/opc_gems_32.json` — OPC 4상 + PC(CSHQ) 번들 명시.
- `examples/opc_slag_populations_32.json` — 슬래그 블렌드 + 재료별 PSD.
- `examples/opc_gypsum_cnash_32.json` — **황산염·알칼리 채널(v3.0/E3)**: OPC 4상 +
  석고 5.0 + 아르카나이트(K2SO4) 0.6 + 테나르다이트(Na2SO4) 0.4 wt%. 미배정 잔여
  2.0%는 inert 유지. 이 담체 분율은 **시연 배합**으로, SO3 ≈ 2.83 %·Na2O당량 ≈ 0.39 %
  (리포트 `binder_oxides`에 파생 표기)라는 일반 OPC 수준을 재현하도록 고른 값이다 —
  밀 성적서 실측치가 있으면 그대로 교체하면 된다. **담체 용해에는 속도식이 없다**:
  매 스텝 물이 닿는 전량을 GEMS 평형에 내놓고 얼마가 고체로 남을지는 용해도가 정한다
  (PRD §1.2/§4.2). 따라서 맞춰야 할 시간상수도, tabulated 표의 담체 열도 없다.
  리포트의 `binder_oxides`는 레시피 값과 `*_as_built`(래스터화된 RVE가 실제로 담은 양)를
  함께 싣는다 — 소량 담체는 입자 샘플러가 목표를 근사로만 맞추므로 격차가 크면 더 미세한
  `material_psd`나 더 큰 격자가 필요하다는 신호다. 재침전 경로는 번들에 달렸다:
  CNASH 번들에는 K2SO4가 없어 arcanite는 무수석고처럼 용해 전용이다(PC 번들에는 있다).

## 모듈 (상한 16, 현재 16 — 상한 도달, 추가 시 통합·삭제 선행)

`src/tinn/`: `config.py`(스키마+해시), `registry.py`(상/성분 데이터),
`geometry.py`(주기 RVE 초기화, 3계층 입자, 타원체 형상), `kinetics.py`(Tabulated+
ParrotKilloh+SCM 로지스틱), `state.py`(SimulationState, mol 권위 원장),
`ledger.py`(§6.1 불변식), `dissolution.py`(전도 면수 가중 배분 — 겔 전도 포함),
`transport.py`(클러스터 라벨링+리매핑 — v4.0/RT 도메인 그래프·BE 교환 + Tier 0 NP 유효 전도도), `backend.py`(ReactionBackend+합성),
`morphology.py`(배치+제거+오버플로), `engine.py`(트랜잭션 오케스트레이터),
`storage.py`(Zarr-v2 체크포인트), `gems.py`(격리 xGEMS 워커+번들 감사+0D 프로브+
GemsBackend snapshot), `analysis.py`(공극/수송 분석+리포트+§6.3 판정+PNG),
`cli.py`(validate-config/run/restart/report), `__init__.py`.
