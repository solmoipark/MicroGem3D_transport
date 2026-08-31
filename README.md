# MicroGEM3D — 시멘트 수화 4D 플랫폼 v2

배합(C3S, OPC 4상, SCM 블렌드), PSD, w/c, 온도, 시간 스케줄을 입력받아 질량 보존되는
시간 의존 3D 미세구조(상 분포·물 분배·공극·수송 지표)를 결정론적·재시작 가능하게
출력하는 시뮬레이션 플랫폼. 설계와 범위는 [PRD.md](PRD.md)가 유일한 기준 문서다.

> **이름 안내.** 플랫폼 이름은 **MicroGEM3D**이고, 개발 중 쓰던 이름은 TINN이었다.
> 파이썬 패키지·모듈 이름공간(`src/tinn/`, `python -m tinn.cli`)과 환경변수
> (`TINN_GEMS_PYTHON`, `TINN_GEMS_PERSISTENT`)는 **아직 `tinn`**이다. 재현성 계약
> (config 해시·체크포인트)에 얽혀 있어 리네임은 별도 작업으로 남겨 둔다. 아래
> 명령의 `tinn`은 오타가 아니다.

## 요구 사항

- Python ≥ 3.11, `numpy`, `pydantic` (테스트는 `pytest`)
- GEMS 백엔드(`gems3k`)는 xgems(pybind11)가 설치된 별도 인터프리터가 필요
  (config `gems_worker_python` 또는 환경변수 `TINN_GEMS_PYTHON`; 예:
  `miniforge3/envs/py313-xgems`). 미설치 시 합성 백엔드의 전 기능은 정상 동작.

## 빠른 시작

```powershell
cd MicroGEM3D
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
`transport.py`(클러스터 라벨링+리매핑), `backend.py`(ReactionBackend+합성),
`morphology.py`(배치+제거+오버플로), `engine.py`(트랜잭션 오케스트레이터),
`storage.py`(Zarr-v2 체크포인트), `gems.py`(격리 xGEMS 워커+번들 감사+0D 프로브+
GemsBackend snapshot), `analysis.py`(공극/수송 분석+리포트+§6.3 판정+PNG),
`cli.py`(validate-config/run/restart/report), `__init__.py`.
