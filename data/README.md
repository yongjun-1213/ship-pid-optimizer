# 관측 데이터 출처

## R/V Gunnerus

- `gunnerus_seapath_2hz.csv.gz`는 shiplab 공개 저장소의 `operational/wave-shielding-2023/data/seapath_2hz.csv.gz` 원본(15,597행)입니다. 파일 내용을 변경하지 않았습니다. [데이터 파일](https://github.com/shiplab/gunnerus/blob/main/operational/wave-shielding-2023/data/seapath_2hz.csv.gz)
- Jisang Ha와 Henrique M. Gaspar(NTNU)가 데이터를 정리했습니다. Roll·Pitch 부호와 단위는 공개 데이터 사전을 따릅니다. [데이터 사전](https://github.com/shiplab/gunnerus/blob/main/operational/wave-shielding-2023/channels.csv) · [프로젝트](https://shiplab.github.io/gunnerus/)
- Wang, T. et al. (2025). *Full-scale experimental research on wave shielding effect of RV Gunnerus for offshore operations*. Ocean Engineering 320, 120189. [실험 논문](https://doi.org/10.1016/j.oceaneng.2024.120189)
- 공개 자료 이용 조건은 CC BY-NC 4.0입니다. [출처의 이용 조건](https://github.com/shiplab/gunnerus/tree/main/operational/wave-shielding-2023) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- SHA-256 `ba220848db4ebf375c1a2d802c2dfee6eb12551b73c8b18f82b7e8da684c4df8`

## R/V Akademik Tryoshnikov

- `ace_hydrins_20170126_excerpt.csv.gz`는 `ace-inertial-navigation-csv.zip`의 `ace_hydrins_20170126.csv`에서 PC UTC 13:20:00–13:35:00 구간을 발췌한 900행입니다. 헤더, 원본 열, 각 행의 값을 유지하고 gzip으로 압축했습니다.
- 원본 헤더 문서의 PC 시각 열은 `pc_time_hhmmssss`로 적혀 있지만 검증한 아카이브의 실제 파일은 `pc_time_hhmmsssss`를 사용합니다. 두 이름을 업로드 매핑 후보에 포함합니다. 각도 단위는 도이며, Roll·Pitch 부호는 별도 설정이 필요합니다. [데이터 헤더](https://zenodo.org/records/3733228/files/data_file_header.txt) · [자료 설명](https://zenodo.org/records/3733228/files/README.txt)
- Thomas, J., & Pina Estany, C. (2020). *Uncorrected inertial navigation dataset collected on board the R/V Akademik Tryoshnikov during the austral summer of 2016/2017 as part of the Antarctic Circumnavigation Expedition (ACE)*. Zenodo, v1.0. [DOI 10.5281/zenodo.3733228](https://doi.org/10.5281/zenodo.3733228)
- 공개 자료 이용 조건은 CC BY 4.0입니다. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- SHA-256 `ff33a637205a0e8bfe624e43d6b9f1d11891981f21cae2b7b3b635daf5750adb`
