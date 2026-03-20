# Danish Pharmacy Sales Analysis

Analysis of pharmaceutical sales trends for a pharmacy in Roskilde, Denmark, using 13 days of transaction data from April 2023.

## Structure

```
MEDICAL_SALES_ANALYSIS/
├── app/
│   ├── forecasting.py
│   └── main.py
├── data/
│   └── pharmacy_sales.csv
├── notebooks/
│   ├── 01_medicine_sales_analysis.ipynb
│   └── 02_top_selling_medicines_analysis.ipynb
├── Insights_Report.docx
├── Project_Scope.docx
└── requirements.txt
```

## Setup

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the notebooks in order using Jupyter with the project virtual environment as the kernel.

Run the forecasting dashboard from the `app/` directory:

```bash
streamlit run main.py
```
