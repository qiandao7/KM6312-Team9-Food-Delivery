# Food Delivery Operations and Delivery Efficiency

KM6312 Group Project — Team 9, Nanyang Technological University.

## Research questions

1. What operational, environmental, and delivery-partner factors are associated with delivery time and delivery efficiency?
2. To what extent can actual delivery time be predicted using information available before delivery completion?
3. Can machine-learning models identify low-efficiency deliveries in advance, and which factors contribute most to the prediction?

## Files

| File | Purpose |
| --- | --- |
| `food_delivery_analytics_cleaned.csv` | Original dataset, kept unchanged. |
| `KM6312_Data_Preparation.ipynb` | Complete analysis, saved outputs, sensitivity checks, and illustrative cases. |
| `make_report_figures.py` | Recreates the 13 figures used in the report, with its publication layout and colours. |
| `requirements.txt` | Tested Python package versions. |

## Data source

Deepesh Kansotia, *Food Delivery Operations & Customer Analytics*, Kaggle:
https://www.kaggle.com/datasets/deepeshkansotia/food-delivery-operations-and-customer-analytics

The dataset contains 15,000 synthetic records and 30 variables. The primary analysis uses 12,997 non-cancelled orders. It uses an 80/20 split with `random_state=42`, giving 10,397 training records and 2,600 test records.

## Run the analysis

Tested with Python 3.14.6 and the package versions in `requirements.txt`.

1. Download the repository using **Code → Download ZIP**, then extract it.
2. Open a terminal in the extracted project folder and install the packages:

```bash
python -m pip install -r requirements.txt
```

3. Start JupyterLab:

```bash
python -m jupyterlab
```

4. Open `KM6312_Data_Preparation.ipynb`. Keep the CSV in the same folder and choose **Restart Kernel and Run All Cells**. The notebook creates `outputs/step03` through `outputs/step06`, including 15 analysis figures and the CSV results needed for report figures.
5. To recreate the report figures after running the notebook:

```bash
python make_report_figures.py
```

The script saves 13 figures in `report_figures/`. The report uses different figure numbering and layouts from the notebook. Times New Roman should be installed to match the report typography.

## Main results and interpretation

Ridge regression achieves test MAE 6.309 minutes, compared with 6.289 for the existing ETA. Logistic regression achieves F1 0.921 and ROC-AUC 0.992. Low efficiency means a score at or below the training-set 25th percentile, 46.9.

The primary prediction scenario assumes actual preparation time is known before delivery completion. Removing preparation time increases MAE to 13.096 and reduces F1 to 0.806. Availability of the remaining inputs at prediction remains an assumption because collection timestamps are absent.

The efficiency score is exactly reproduced by a fixed expression involving delivery time, traffic, weather, and partner experience. Actual delivery time is excluded from classifier inputs. Results describe prediction within synthetic data and do not establish causal effects or validated improvements on a real delivery platform.
