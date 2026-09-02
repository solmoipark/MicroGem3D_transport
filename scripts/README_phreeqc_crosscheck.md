# 0D cross-check spike: PHREEQC-Cemdata18 vs xGEMS

목적: 공식 Cemdata18 PHREEQC 배포본이 **이 프로젝트의 조성 창** 안에서 GEMS
번들의 평형 화학을 얼마나 재현하는지 데이터로 확정한다. 이 결과가
PHREEQC를 (a) 교차검증용 제2엔진으로 둘지, (b) 주 엔진 후보로 승격할지,
(c) Tier 0/1(종별 NP 수송·수착 연산자)에만 쓸지를 결정한다.

스파이크다: 모듈이 아니고(상한 16 비침범), engine/ledger를 건드리지 않으며,
양 엔진에 **같은 원소 mol 벡터**(원장 통화)를 넣는다.

## 설치 (사용자 머신, 한 번)

```powershell
py -3 -m pip install phreeqpython
```

xGEMS 쪽은 기존 `py313-xgems` 인터프리터 그대로 (`TINN_GEMS_PYTHON` 또는
`--gems-python`).

## 실행

```powershell
cd "TINN_ reactive transport"

# 1) PHREEQC 측 (아무 파이썬에서나 돌아감)
py -3 scripts\spike_phreeqc_crosscheck.py --engines phreeqc --out runs\spike_crosscheck

# 2) GEMS 측 (CNASH 번들 기본; xgems 필요)
py -3 scripts\spike_phreeqc_crosscheck.py --engines gems --out runs\spike_crosscheck

# 3) 비교표 생성 (comparison.md / comparison.json)
py -3 scripts\spike_phreeqc_crosscheck.py --compare runs\spike_crosscheck
```

한 번에: `--engines phreeqc,gems`. PC(CSHQ) 번들 비교는
`--bundle gems_bundles/PC/PC-dat.lst --csh-model cshq`,
CNASH 번들 비교는 `--csh-model cnash`(기본은 cshq이므로 CNASH 번들에
맞출 때 명시).

## 케이스 (100 g 결합재 기준, 원소 벡터는 registry 화학식으로 생성)

| case | 내용 | 노리는 영역 |
|---|---|---|
| ch_buffered_opc | OPC α=0.5 + 석고, w/b 0.5 | CH 완충 일상 영역 |
| post_ch_sf_blend | 70 g OPC(α=0.7) + 30 g SF | CH 소진, 저 Ca/Si C-S-H |
| e3_alkali_sulfate | A + 아르카나이트 0.6 g + 테나르다이트 0.4 g | E3 채널 |
| high_alkali_stress | A + 0.5 mol/kgw NaOH | 활동도 모델 발산 예상 지대 (진단용) |

## 판정 기준 제안

- pH 차 ≤ 0.1 (스트레스 케이스 제외), 주요 원소 수용액 농도 상대차 수십 %
  이내(절대 농도가 1e-7 mol 미만인 원소는 제외), 상 조합 정성 일치(같은
  상이 존재/부재), C-S-H Ca/Si 차 ≤ 0.1 — 통과 시 "교차검증용 제2엔진"
  적격. 주 엔진 승격 논의는 InverseGems 28d 앵커 재현까지 요구할 것.
- high_alkali_stress 는 벌어져도 실패가 아님 — 어디부터 벌어지는지 좌표를
  남기는 케이스다.

## 구현 노트 (검토 포인트)

- 원소 벡터 → 중성 산화물/H2O/O2 분해 후 REACTION 으로 투입. 분해는 정확
  폐합 assert (PHREEQC REACTION 은 계수 × 총량 의미라 총량 1.0 사용).
- 후보 상 리스트는 **opt-in** (클링커 억제의 거울상): Si-hydrogarnet,
  석영, 제올라이트는 25 °C 관행대로 제외 — `--extra-phases` 로 추가 가능.
- 고용체는 입력에서 조립: CSHQ 4-endmember 또는 CNASH 8-endmember 이상
  고용체 (`--csh-model`). endmember 몰이 selected output `s_` 칼럼으로
  나와 E1 원장과 같은 수준의 비교가 가능.
- 원소 폐합은 .dat 의 PHASES 반응식을 파싱해 상별 화학식으로 재합산
  (손 복사 없음). 컨테이너 실측: 4 케이스 모두 ≤ 5e-11 (상대).
- 알려진 모델 차이(비교표 해석 시 참고): CNASH_ss 에는 K endmember 가
  없어 K 는 수용액에 남는다(ECSH 계열은 별도). **정정(2026-09-02 실측)**:
  CSHQ "알칼리 무흡수"는 사실이 아니다 — PC 번들의 CSHQ 는 KSiOH/NaSiOH
  를 포함한 6-endmember 이고, cemdata18.dat 에도 같은 이름·같은 화학식
  (((KOH)2.5SiO2H2O)0.2 등)으로 존재한다. cshq 모델 리스트에 반영됨.
  post-CH 케이스에서 CSHQ(pH 11.7)와 CNASH(pH 10.6)가 크게 갈리는 것은
  C-S-H 모델 자체의 차이로, 번들과 같은 모델을 골라 비교할 것.

## 확정 구성 (사용자 결정, 2026-09-02)

**CSHQ(PC 번들)를 사용하고 Si-hydrogarnet 은 억제하지 않는다** — 논문 런과
동일한 구성. 억제(B안)는 교차검증의 대칭 조건으로만 쓰였고 프로덕션에는
적용하지 않는다. 확정 구성 재실행(`runs/spike_crosscheck_pc_final`,
하이드로가넷 양쪽 허용: `--extra-phases "C3AFS0.84H4.32,C3FS0.84H4.32"
--gems-unsuppress "C3(AF)S0.84H"`)에서도 일치 품질은 동일하다:
ΔpH −0.001/+0.104/+0.034/+0.042, 하이드로가넷 몰수 0.01646/0.01642 일치,
post-CH 케이스는 양 엔진 모두 형성하지 않음. **판정: PHREEQC-Cemdata18 은
이 조성 창에서 교차검증용 제2엔진 적격.**

## 실측 결과 (PC 번들, 사용자 머신, 2026-09-02)

전제 조건 두 가지를 맞춘 뒤의 결과다: (i) GEMS 콜드스타트에 상대 O2 시드
(O2_SEED_REL, 없으면 4 케이스 전부 LPP AIA 비수렴), (ii) 양 엔진 후보 상
정렬 — Si-hydrogarnet C3(AF)S0.84H 와 제올라이트 5종(Chabazite/Natrolite/
ZeoliteP/X/Y)을 GEMS 측에서도 억제(25 °C 관행, B안 결정), PHREEQC 측
CSHQ 를 6-endmember(KSiOH/NaSiOH 포함)로.

| case | pH (PHQ/GEMS) | ΔpH | Ca/Si (PHQ/GEMS) | ΔCa/Si | 판정 |
|---|---|---|---|---|---|
| ch_buffered_opc | 12.475 / 12.477 | −0.001 | 1.6267 / 1.6270 | −0.0004 | 통과 |
| post_ch_sf_blend | 11.229 / 11.125 | +0.104 | 0.8313 / 0.8007 | +0.031 | 경계 (기준 0.1) |
| e3_alkali_sulfate | 13.131 / 13.097 | +0.034 | 1.6031 / 1.6039 | −0.001 | 통과 |
| high_alkali_stress | 13.452 / 13.410 | +0.043 | 1.5853 / 1.5866 | −0.001 | 통과 (스트레스 케이스인데도) |

수용액 원소는 Na ±4 % / K ±2 % / Si ±10 % / Ca ±20 % / S ±15 % 수준.
Fe 는 억제 정렬 후 ferrihydrite 로 양 엔진 일치(0.016462 mol, 8자리).
결론: **교차검증용 제2엔진 적격** (README 판정 기준). 억제 정렬 전에는
ΔpH 0.3~0.6 이 났으며 전부 후보 상·endmember 비대칭이 원인이었다 —
데이터베이스 자체의 열역학은 이 조성 창에서 잘 맞는다.

## 실측 결과 (CNASH Test 번들, 사용자 머신, 2026-09-02)

**번들 조건부 주의(실측)**: CNASH Test 번들의 유일한 Fe 수화물은
C3(AF)S0.84H 다 — ferrihydrite 가 없다. 여기에 B안 억제를 적용하면 투입
Fe 0.0165 mol 전량이 수용액으로 밀려나 pH 비교가 0.4~1.0 오염된다(실측).
따라서 CNASH 비교는 하이드로가넷을 **양쪽 다 허용**으로 돌린다:
`--extra-phases "C3AFS0.84H4.32,C3FS0.84H4.32" --gems-unsuppress "C3(AF)S0.84H"`.
같은 이유로, B안(Si-하이드로가넷 억제)의 프로덕션 채택은 PC 번들에만
가능하고 CNASH 번들에는 대체 Fe 싱크 없이는 불가하다.

| case | pH (PHQ/GEMS) | ΔpH | Ca/Si (PHQ/GEMS) | 판정 |
|---|---|---|---|---|
| ch_buffered_opc | 12.475 / 12.477 | −0.001 | 1.3571 / 1.3576 | 통과 |
| post_ch_sf_blend | 10.555 / 10.386 | +0.169 | 0.8561 / 0.8287 | 경계 초과 |
| e3_alkali_sulfate | 13.343 / 13.304 | +0.039 | 1.3495 / 1.3467 | 통과 |
| high_alkali_stress | 13.472 / 13.451 | +0.021 | 1.2883 / 1.2917 | 통과 |

Fe 는 하이드로가넷으로 양 엔진 일치(0.01646/0.01641 mol), CH 도 3자리
일치(0.2846/0.2842). post-CH 케이스는 PC 모드(+0.104)와 같은 방향으로
CNASH 모드가 조금 더 벌어진다(+0.169) — 저 Ca/Si·저알칼리 완충 영역이
두 코드의 활동도 모델 차이에 가장 민감하다는 뜻이고, 좌표로 기록해 둔다.

## 컨테이너 실측 스냅샷 (PHREEQC 측, 2026-09-01, 4-endmember 당시)

CSHQ 모델: A) pH 12.475, Ca/Si 1.63, CH 0.219 mol, AFm/AFt 형성;
B) pH 11.71, Ca/Si 0.83, CH 소멸, gibbsite 형성; C) pH 13.50;
D) pH 13.74, I=0.64 (ettringite→monosulphate 전환). 모두 시멘트 화학
상식과 정합 — GEMS 측 수치는 사용자 머신 실행으로 채울 것.

## G0-확장: 0D 활동도/스페시에이션 비교 (2026-09-02, PHREEQC-main 검토 입력)

`spike_activity_crosscheck.py` — 실측 체크포인트 6점(밀봉 OPC 24/168/672 h
+ NP 용출 체인 168/168.1/168.2 h)의 실제 공극수를 **수용액-전용**으로 양쪽
엔진에서 스페시에이션(고체 억제, PHREEQC는 상 블록 없음), 공동 정준 스케일.

| 지표 | 결과 (6케이스) |
|---|---|
| ΔpH (P−G) | **+0.037** (pH 13 균일) → +0.017 (12.4) → +0.006 (11.7) — 계통 오프셋, 희석과 함께 소멸 |
| Δ이온강도 | **≤ 0.7%** 전 케이스 |
| 자유 SO4²⁻ 분율 (수용액 기준) | 1.5–3% 이내 일치 (G 0.65–0.80 vs P 0.63–0.79) |
| 자유 Ca²⁺ 분율 | 1–9% 이내 (pH 13에서 최대 — CaOH⁺/CaSiO3⁰ 쌍형성 디테일 차이) |

**판정**: 우리 운영 창(pH 11.7–13.1, I 9–126 mM)에서 두 코드의 수용액
모형은 한 자릿수 % 이내로 호환 — "activity 개선"을 이유로 한 PHREEQC-main
전환은 실측으로 지지되지 않음. ΔpH +0.037은 수착 log_K 환산 ~9%로 S1c
불확도 밴드(±0.1 dex) 내부.

**부수 발견 (버그)**: `suppress_multiple_phases`가 0D 경로에서
**CO3_SO4_AFt 고용체를 억제하지 못함**(입력 S의 최대 45%가 석출; 자매상
SO4_CO3_AFt와 모든 단일-DC 상은 정상 — xgems 인덱싱 의심). 엔진 경로는
비노출(단일-DC 클링커 억제 + 자체 증인). 조치: `equilibrate_elements`에
엔진과 동일한 억제-증인 이식(누출 = 하드 에러, 침묵 오염 차단), 스파이크는
해당 상을 의도적 비억제로 처리(수용액-분모 비교라 유효). 과거 G0 비교의
억제 상(하이드로가넷/제올라이트)은 단일-DC 위주로 무발화였을 가능성이
크나, 다음 스파이크 재실행 시 증인이 자동 검증한다.
