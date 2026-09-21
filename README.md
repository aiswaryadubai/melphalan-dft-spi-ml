# Dual-Index DFT–Machine-Learning Framework for Melphalan Sensing

This repository contains the 76-system dataset and analysis code supporting the manuscript **“Beyond Adsorption Energy: A Dual-Index DFT–Machine Learning Framework for Prioritizing Nanostructures for Melphalan Sensing.”**

## Important implementation note

This release preserves the numerical ranking formulation used to generate the manuscript's current ranking tables. The implemented component scores are:

- `Eads_score = MinMax(E_ads_DFT_eV)`
- `Q_score = MinMax(Charge_Transfer_e)`
- `Recovery_score = 1 - MinMax(Recoverytime_sec)`
- `Gap_score = 1 - MinMax(Bandgap_eV)`

The adsorption-energy and charge-transfer transformations therefore use the **signed values**, not their absolute magnitudes. In this implementation, a larger adsorption-energy score corresponds to a less-negative adsorption energy, while a larger charge-transfer score corresponds to a more-positive signed charge-transfer value. These definitions must be reported exactly when citing the archived results.

The indices are:

```text
SPI_sensor = 0.40*Eads_score + 0.40*Q_score + 0.20*Recovery_score

SPI_device = 0.30*Eads_score + 0.30*Q_score
             + 0.20*Recovery_score + 0.20*Gap_score
```

The machine-learning target is the same `SPI_sensor` calculated above. Because the target is constructed from descriptors that are also supplied to the models, the ML analysis is a nonlinear surrogate-modelling exercise and not independent discovery.

## Repository contents

```text
data/
  melphalan_76_systems.xlsx   Original 76-system input table
  melphalan_76_systems.csv    Machine-readable copy
src/
  run_analysis.py             SPI, rankings, sensitivity and ML validation
  shap_analysis.py            SHAP analysis for Gradient Boosting
results/                      Generated after running the analysis
requirements.txt              Python dependencies
CITATION.cff                  Citation metadata template
LICENSE                       MIT licence for the code
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce the analysis

Run from the repository root:

```bash
python src/run_analysis.py
python src/shap_analysis.py
```

The first command creates the complete rankings, weight-sensitivity summary, fold-level cross-validation results, model-performance table, and cross-validation figure in `results/`. The second creates the SHAP feature-importance table and summary plot.

## Expected ranking check

With the preserved signed-value formulation, aqueous BNNS–Ni (`BN_Ni1`) should be Rank 1 under both baseline indices.

## Creating a DOI

1. Create a public GitHub repository named `melphalan-dft-spi-ml`.
2. Upload the contents of this folder, keeping the folder structure unchanged.
3. Run the analysis once from a clean environment and commit the generated results.
4. Connect the GitHub repository to Zenodo.
5. On GitHub, create a release tagged `v1.0.0`.
6. Zenodo will archive that release and issue a version-specific DOI.
7. Replace the placeholder DOI and repository URL in `CITATION.cff`.

## Data and code availability statement

> The complete 76-system dataset, Sensor Performance Index calculations, weight-sensitivity analysis, machine-learning scripts, model settings, cross-validation results, and ranking tables are available in Zenodo at https://doi.org/10.5281/zenodo.XXXXXXX. The development version is available at https://github.com/USERNAME/melphalan-dft-spi-ml.

## Authors

Aiswarya T and K. K. Singh  
Department of Physics  
Birla Institute of Technology and Science, Pilani, Dubai Campus, United Arab Emirates

