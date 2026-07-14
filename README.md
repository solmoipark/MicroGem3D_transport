# TINN 시멘트 수화 4D 플랫폼 v2

배합(C3S 또는 OPC 4상), PSD, w/c, 온도, 시간 스케줄을 입력받아 질량 보존되는
시간 의존 3D 미세구조를 결정론적·재시작 가능하게 출력하는 시뮬레이션 플랫폼.
설계와 범위는 [PRD.md](PRD.md)가 유일한 기준 문서다.

## 요구 사항

- Python ≥ 3.11, `numpy`, `pydantic` (테스트는 `pytest`)

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

# 테스트
py -3 -m pytest -q
```

## 진행 상태 (PRD §3 마일스톤)

- [x] **M0** — 골격과 초기화: config/registry/geometry/cli, 32³ 다상 RVE 초기화,
      동일 시드 → 동일 해시, 고체분율·w/c 오차 리포트 (~1e-5 수준).
- [x] **M1** — 보존 코어: 트랜잭션 스테핑(trial→검사→commit/rollback), 원소 수지
      ~4e-25 mol, 물 수지 0, 복셀 항등식 ≤1e-12, Zarr-v2 체크포인트/재시작
      비트단위 동등, 32³ C3S 런 40초.
- [ ] M2 — P&K 동역학과 4상 구동
- [ ] M3 — GEMS 0D 프로브
- [ ] M4 — GEMS→3D 결합
- [ ] M5 — 분석과 리포트

## 예제 config

- `examples/c3s_32.json` — 순수 C3S, w/C3S 0.50, 293.15 K, tabulated alpha (§6.2 C3S 픽스처).
- `examples/opc_srm114q_32.json` — NIST SRM 114q 4상 레시피(60/14/7/10, 미배정 9%),
  P&K 프리셋 `pk_elakneswaran_2018` (동역학 구현은 M2).

## 모듈 (상한 16, 현재 14)

`src/tinn/`: `config.py`(스키마+해시), `registry.py`(상/성분 데이터),
`geometry.py`(주기 RVE 초기화, 3계층 입자), `kinetics.py`(TabulatedKinetics),
`state.py`(SimulationState, mol 권위 원장), `ledger.py`(§6.1 불변식),
`dissolution.py`(액체 접촉 가중 배분), `transport.py`(클러스터 라벨링+리매핑),
`backend.py`(ReactionBackend+합성), `morphology.py`(내부/외부 배치),
`engine.py`(트랜잭션 오케스트레이터), `storage.py`(Zarr-v2 체크포인트),
`cli.py`(validate-config/run/restart), `__init__.py`.
남은 슬롯: `gems.py`(M3), `analysis.py`(M5).
