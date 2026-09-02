# literature/ — 사용자 공급 문헌 색인 (PDF·db는 gitignore, 색인만 추적)

원칙: 보정 상수는 문헌이 소유(PRD d0 전례). 각 항목은 어느 실측 기록에 쓰였는지 적는다.

## sulfate/ — RT-S1c SO4 표면 수착 보정 (PRD 4.6.5)
| 파일 | 서지 | 역할 |
|---|---|---|
| 1998_Divet_CCR28 | Divet L., Randriambololona R., *Cem. Concr. Res.* 28(3) (1998) 357–363 | 등온선(Fig. 2–4) → log_K 피팅, SSA 350 m²/g |
| 2006_Labbez_JPCB110 | Labbez C., Jönsson B., Pochard I., Nonat A., Cabane B., *J. Phys. Chem. B* 110 (2006) 9219–9230 | 실란올 4.8/nm² → 사이트 밀도 |
| 2015_Haas_Nonat_CCR68 | Haas J., Nonat A., *Cem. Concr. Res.* 68 (2015) 124–138 | 몰당 titratable 실란올(교차), pK_H/K_SiOCa(ddl 시도) |
| 2007_Barbarulo_CCR37 | Barbarulo R., Peycelon H., Leclercq S., *Cem. Concr. Res.* 37 (2007) 1176–1181 | 20 °C AFt 평형 [SO4] ≈ 0.4 mM sanity |
| 2016_Ochs_book | Ochs M., Mallants D., Wang L., *Radionuclide and Metal Sorption on Cement and Concrete*, Springer (2016) | 약한 음이온(Se(VI)) R_d 밴드 sanity |

## chloride/ — RT-Cl-2 Cl 표면 수착 보정 (예정)
| 파일 | 서지 | 역할 |
|---|---|---|
| 2016_Plusquellec_Nonat_CCR90 | Plusquellec G., Nonat A., *Cem. Concr. Res.* 90 (2016) 89–96 | 합성 C-S-H Cl⁻ 흡착 등온선 — 1차 보정 |
| 2009_Elakneswaran_CCR39 | Elakneswaran Y., Nawa T., Kurumisawa K., *Cem. Concr. Res.* 39 (2009) 340–344 | PHREEQC 표면착화 반응·log K — 직접 이식 후보 |
| 2005_Hirao_JACT3 | Hirao H., Yamada K., Takahashi H., Zibara H., *J. Adv. Concr. Technol.* 3(1) (2005) 77–84 | 수화물별(C-S-H/AFm) 결합 등온선 — C-S-H 몫 분리 |
| 2012_Florea_Brouwers_CCR42 | Florea M.V.A., Brouwers H.J.H., *Cem. Concr. Res.* 42 (2012) 282–290 | C-S-H(물리) vs AFm(화학) 배분 sanity |
| 1993_Tang_Nilsson_CCR23 | Tang L., Nilsson L.-O., *Cem. Concr. Res.* 23 (1993) 247–253 | 페이스트 총 결합 등온선 sanity |
| 1990_Beaudoin_CCR20 | Beaudoin J.J., Ramachandran V.S., Feldman R.F., *Cem. Concr. Res.* 20 (1990) 875–883 | 화학흡착 vs 물리흡착 — 반응형 근거 |
| 2015_DeWeerdt_CCR68 | De Weerdt K., Colombo A., Coppola L., Justnes H., Geiker M.R., *Cem. Concr. Res.* 68 (2015) 196–202 | NaCl vs CaCl₂ 양이온 보정 |
| 2001_Zibara_PhD | Zibara H., *Binding of external chlorides by cement pastes*, PhD thesis, Univ. Toronto (2001) | 광범위 결합 등온선(보조) |

## tdm/ — 사용자 TDM 데이터베이스 (Elsevier 문헌 텍스트-데이터마이닝)
`chloride_tdm.db` (SQLite, 200 MB): papers 2,179 · mixes 20,360 · binding_isotherms 3,403(피팅형·계수)
· binding_points 318 · profiles 15,386 · profile_points 3,808 · aux_observations 341,682.
용도: 페이스트/콘크리트 수준 결합 등온선·침투 프로파일의 sanity 밴드(표면 상수 원천은 아님).
