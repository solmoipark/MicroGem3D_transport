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

# 32³ RVE 초기화 스모크 (M0)
py -3 -c "from tinn.config import TinnConfig; from tinn.registry import default_registry; from tinn.geometry import initialize_rve; import json; r = initialize_rve(TinnConfig.from_json_file('examples/c3s_32.json'), default_registry()); print(json.dumps(r.report, indent=2))"

# 테스트
py -3 -m pytest -q
```

## 진행 상태 (PRD §3 마일스톤)

- [x] **M0** — 골격과 초기화: config/registry/geometry/cli, 32³ 다상 RVE 초기화,
      동일 시드 → 동일 해시, 고체분율·w/c 오차 리포트 (~1e-5 수준).
- [ ] M1 — 보존 코어 (합성 백엔드 전체 루프)
- [ ] M2 — P&K 동역학과 4상 구동
- [ ] M3 — GEMS 0D 프로브
- [ ] M4 — GEMS→3D 결합
- [ ] M5 — 분석과 리포트

## 예제 config

- `examples/c3s_32.json` — 순수 C3S, w/C3S 0.50, 293.15 K, tabulated alpha (§6.2 C3S 픽스처).
- `examples/opc_srm114q_32.json` — NIST SRM 114q 4상 레시피(60/14/7/10, 미배정 9%),
  P&K 프리셋 `pk_elakneswaran_2018` (동역학 구현은 M2).

## 모듈 (상한 16, 현재 5)

`src/tinn/`: `config.py`(스키마+해시), `registry.py`(상/성분 데이터),
`geometry.py`(주기 RVE 초기화, 3계층 입자), `cli.py`(validate-config), `__init__.py`.
