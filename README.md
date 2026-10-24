# Forecasting Carbon Emissions with Deep Learning

Univariate forecasting of sector-level CO2 emissions to 2050 for the seven largest emitters, comparing linear regression, ARIMA, and shallow and deep neural networks implemented in PyTorch.

## Overview

Carbon dioxide accounts for roughly four-fifths of global greenhouse-gas emissions, and credible forecasts of where those emissions are heading are a prerequisite for climate policy and investment planning. This project asks which of four forecasting approaches can be trusted to project each emission series to 2050, using only the series' own history.

**Data.** World Bank World Development Indicators, 1970 to 2023, in million tonnes of CO2.

**Regions (7).** Brazil, China, European Union, India, Indonesia, Russia, United States.

**Sectors (6).** Agriculture, fugitive emissions, industrial combustion, industrial processes, power, transportation.

**Models (4).**

| Model | Description |
|---|---|
| Linear Regression | Ordinary least squares on a cubic polynomial of time |
| ARIMA(1,0,0) | Autoregressive model with constant and the cubic time polynomial as exogenous regressors |
| Shallow Neural Network | 1 hidden layer of 100 ReLU neurons, PyTorch |
| Deep Neural Network | 3 hidden layers of 100 ReLU neurons, PyTorch |

**Evaluation.** Each series is split chronologically into 43 training years (1970 to 2012) and 11 test years (2013 to 2023). A model is accepted for a series only if it passes two tests: its test MAPE exceeds its training MAPE by less than 10 percentage points (no overfitting), and its test MAPE is below that of a naive forecast that carries last year's value forward. Accepted models are refit on the full history and used to forecast 2024 to 2050.

## Results

Of the 42 sector-region series, the deep neural network passed both tests on 7, while linear regression, ARIMA and the shallow network each passed on 2. The deep network also had the lowest mean test error (14.1%, against 15.3% for the shallow network and about 31% for linear regression and ARIMA) and was the only model to pass in five of the six sectors. The seven accepted DNN forecasts cover India's industrial combustion, industrial processes, power and transportation sectors, China's power sector, and fugitive emissions in Indonesia and the European Union.

Full tables, forecasts and plots are in the notebook and in `outputs/`.

## Repository contents

```
Forecasting_Carbon_Emissions_with_Deep_Learning.ipynb   Executed notebook with all outputs
outputs/
  co2_sector_data_1970_2023.xlsx       Raw World Bank data, one sheet per sector
  results_<MODEL>.xlsx                 Train, test and naive MAPE and pass/fail for all 42 series
  results_all_models.xlsx              The four results tables combined
  success_matrix.xlsx                  Pass/fail grid, series by model
  forecasts_all_models.xlsx            2024, 2025, 2030, 2040 and 2050 forecasts for every model and series
  <MODEL>_Fugitive_Emissions_forecasts_2024_2050.xlsx   Year-by-year forecasts for the walkthrough sector
  validation_2024.xlsx                 Forecasts compared with actual 2024 values
  *.png                                Forecast plots for the walkthrough sector
requirements.txt
```

## Running the notebook

Open the notebook in Google Colab or Jupyter and run the cells in order. The first cell installs the dependencies; the data is downloaded from the World Bank API at run time, so an internet connection is required. A full run takes a few minutes on CPU.

Results are seeded and reproduce exactly in the pinned environment in `requirements.txt`. Linear regression and ARIMA use a centered time variable so that their fits are exact on any machine; the neural networks use standardised inputs and a fixed seed, and their results have been confirmed identical across different machines and PyTorch versions.

## Notes

The naive benchmark is deliberately hard to beat on annual data, and most series are not forecast reliably by any of the four models. The results should be read as a comparison of model families rather than as a claim that any of them forecasts emissions accurately in general. On the one year of truly unseen data now available (2024), the accepted DNN forecasts averaged 6.0% error.

## Author

Vedant Jagtap  
vsj7589@nyu.edu
