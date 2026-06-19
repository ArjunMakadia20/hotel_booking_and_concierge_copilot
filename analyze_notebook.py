import re

# Read the notebook file
with open('notebooks/hotel_booking_and_concierge_copilot_eda.ipynb', 'r') as f:
    content = f.read()

# Count cells
python_cells = len(re.findall(r'language="python"', content))
markdown_cells = len(re.findall(r'language="markdown"', content))

print(f'Python cells: {python_cells}')
print(f'Markdown cells: {markdown_cells}')
print(f'Total cells: {python_cells + markdown_cells}')

# Extract section titles
sections = re.findall(r'### (.+)', content)
print(f'\nNotebook sections:')
for i, section in enumerate(sections, 1):
    print(f'  {i}. {section}')

# Check for specific keywords that indicate completed work
keywords = {
    'EDA': 'exploratory data analysis',
    'Visualization': 'matplotlib|seaborn|plot|scatter|hist|boxplot',
    'Cross-validation': 'cross_val|KFold|cross_validate',
    'Hyperparameter': 'GridSearchCV|RandomizedSearchCV|hyperparameter',
    'XGBoost': 'XGBRegressor|train_xgboost',
    'SHAP': 'shap.Explainer|shap.summary_plot|SHAP',
    'Confusion/Metrics': 'RMSE|MAE|R2|r2_score',
}

print('\nFeature detection:')
for feature, pattern in keywords.items():
    found = bool(re.search(pattern, content, re.IGNORECASE))
    status = '✓' if found else '✗'
    print(f'  {status} {feature}')
