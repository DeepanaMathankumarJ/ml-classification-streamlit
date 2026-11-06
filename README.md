# <Your Project Title>

## a. Problem statement
<2-4 sentences: what are you predicting, why does it matter, binary or multi-class?>

## b. Dataset description
- **Source:** <Kaggle / UCI link>
- **Instances:** <N>  |  **Features:** <N>  |  **Target:** `<column>` (<classes>)
- **Feature types:** <numeric / categorical, key columns>
- **Class balance:** <e.g. 63% / 37%>
- **Preprocessing:** median imputation + standard scaling for numeric features, one-hot encoding for categorical features; stratified 80/20 train-test split (random_state=42).

## c. GitHub repository link
<https://github.com/your-username/your-repo>

## d. Models used

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

(Paste the table printed by `python train_models.py`.)

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <e.g. strong baseline; features roughly linearly separable; high AUC> |
| Decision Tree | <e.g. lower AUC, prone to overfitting even with max_depth=6; easy to interpret> |
| KNN | <e.g. sensitive to scaling; high recall, slower at prediction time> |
| Naive Bayes | <e.g. independence assumption violated by correlated features; fast but weaker MCC> |
| Support Vector Machine | <e.g. RBF kernel captures non-linearity; top-tier AUC/MCC> |
| **Overall winner** | <model + 1-2 lines justifying using MCC/AUC, not accuracy alone> |

## How to run
```bash
pip install -r requirements.txt
python train_models.py        # trains + saves models, test_data.csv
streamlit run app.py
```

## Live app
<https://your-app-name.streamlit.app>
