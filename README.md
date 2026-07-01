# NFL Touchdown Predictor 🏈

A Machine Learning project that predicts whether an NFL play will result in a **Touchdown**, built on a dataset of 362,447 real NFL plays (2009–present).

---

## Project Overview

Given a game situation — down, field position, score, time remaining, play type — the model predicts the probability of the current play resulting in a touchdown.

**Target Variable:** `Touchdown` (0 = No Touchdown, 1 = Touchdown)  
**Problem Type:** Binary Classification  
**Dataset:** NFL Play-by-Play Data (362,447 rows, 102 columns)

---

## Project Structure

```
nfl-touchdown-predictor/
├── NFL_Touchdown_Prediction.ipynb   # Full EDA + Model Building notebook
├── app.py                           # Flask web application
├── nfl_touchdown_model.pkl          # Trained Random Forest model
├── scaler.pkl                       # StandardScaler for feature normalization
├── playtype_encoder.pkl             # LabelEncoder for PlayType column
└── templates/
    └── index.html                   # Frontend UI for the web app
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

**Basic EDA**
- Dataset shape, dtypes, missing value analysis
- Target variable distribution — identified heavy class imbalance (~4% TD rate)

**Intermediate EDA**
- Touchdown rate by PlayType → found TDs occur almost exclusively on Pass/Run plays
- Scoped dataset to Pass + Run plays only (249,314 rows) — data-driven decision, not arbitrary
- Box plots comparing field position and yards-to-go for TD vs non-TD plays
- Class imbalance visualization

**Advanced EDA**
- Correlation heatmap — confirmed `yrdline100` and `GoalToGo` as dominant predictors
- Feature engineering: `grid_vs_finish`, `points_per_grid` style derived features
- Missing value analysis on final feature set

### 2. Preprocessing

- Dropped rows with missing values in selected features (< 0.2% of data)
- Label encoded `PlayType` (Pass/Run → 0/1)
- Train-test split: 80/20 with `stratify=y` to preserve class balance
- StandardScaler applied to features (fit on train, transform on test only)
- Class imbalance handled via `class_weight='balanced'`

### 3. Features Used

| Feature | Description |
|---|---|
| `qtr` | Quarter number (1-4) |
| `down` | Down number (1-4) |
| `TimeSecs` | Time remaining in seconds |
| `yrdline100` | Yards from own goal line (1=opponent's end zone) |
| `ydstogo` | Yards needed for first down |
| `GoalToGo` | Whether it's a goal-to-go situation |
| `ScoreDiff` | Score difference (positive = winning) |
| `AbsScoreDiff` | Absolute score difference |
| `posteam_timeouts_pre` | Possession team timeouts remaining |
| `HomeTimeouts_Remaining_Pre` | Home team timeouts |
| `AwayTimeouts_Remaining_Pre` | Away team timeouts |
| `PlayType_enc` | Encoded play type (Pass/Run) |

### 4. Models Trained & Compared

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.868 |
| Decision Tree | 0.838 |
| **Random Forest** | **0.871** ✅ |
| XGBoost | 0.866 |

**Winner: Random Forest** (highest ROC-AUC)

### 5. Hyperparameter Tuning

GridSearchCV applied to Random Forest with 3-fold cross-validation:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10, 15],
    'min_samples_split': [2, 5]
}
```

Best parameters selected based on ROC-AUC scoring.

---

## Key Findings

- **Field position is everything** — `yrdline100` is the single most important predictor of a touchdown
- **Class imbalance** (4% TD rate) was handled explicitly — plain accuracy is meaningless on this dataset
- **Touchdowns only happen on Pass/Run plays** — filtering to these improved model focus and removed noise
- **Random Forest outperformed all other models** with ROC-AUC of 0.871

---

## Flask Web App

The trained model is deployed as a local Flask web application.

### Running the App

1. Clone this repository
2. Install dependencies:
```
pip install flask pandas numpy scikit-learn xgboost joblib
```
3. Run the app:
```
python app.py
```
4. Open browser → `http://127.0.0.1:5000`

### App Screenshot

Enter game situation details (quarter, down, field position, score, timeouts, play type) and get an instant TD prediction with probability.

---

## Dependencies

```
flask
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn
openpyxl
```

---

## Author

**Kavin Sanjay**  
ECE Student, VIT Chennai  
ML Internship Project — 2026
