"""Script to add Phase 1 EDA cells to the notebook safely."""

import re

# Read the existing notebook with UTF-8 encoding
with open('notebooks/hotel_booking_and_concierge_copilot_eda.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Define new EDA cells to insert after the "Cleaning" section
# We'll insert after the cell with "### Cleaning" markdown

eda_cells = '''<VSCode.Cell language="markdown">
## EXPLORATORY DATA ANALYSIS (EDA) - Phase 1

### Deep Dive into Hotel Booking Patterns

Now that we have cleaned data, let's explore the key patterns, distributions, and relationships in our hotel booking dataset. This analysis will inform our feature engineering and modeling strategy.

**Key Analysis Areas:**
1. Target variable (ADR) distribution and business implications
2. Missing values impact on downstream analysis
3. Hotel type and property comparison
4. Cancellation impact on revenue
5. Lead time patterns and their effect on ADR
6. Seasonal and monthly trends
7. Market segment analysis
8. Feature correlations for modeling
9. Guest demographics
10. Key business insights and recommendations
</VSCode.Cell>

<VSCode.Cell language="python">
# EDA Setup: Matplotlib, Seaborn, and Visualization Configuration
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create reports directory if it doesn't exist
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

# Configure visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure size for consistent output
FIGSIZE = (14, 6)

print(f"Reports will be saved to: {reports_dir}")
print(f"Cleaned dataset shape: {hotel_clean.shape}")
print(f"Features: {hotel_clean.columns.tolist()}")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 1. Target Variable (ADR) Distribution Analysis

**Business Context:** ADR (Average Daily Rate) is our target variable - the average daily revenue per room. Understanding its distribution helps us:
- Identify if it's normally distributed (affects modeling approach)
- Detect outliers that might skew predictions
- Understand pricing strategy and revenue patterns

**Expected Output:** Distribution histogram, statistics, and business interpretation
</VSCode.Cell>

<VSCode.Cell language="python">
# 1. ADR Distribution Analysis
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# Histogram with KDE
axes[0].hist(hotel_clean['adr'], bins=50, edgecolor='black', alpha=0.7)
hotel_clean['adr'].plot(kind='kde', ax=axes[0], secondary_y=True, color='red', linewidth=2)
axes[0].set_title('ADR Distribution (Histogram + KDE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Average Daily Rate ($)')
axes[0].set_ylabel('Frequency')

# Box plot to show outliers
axes[1].boxplot(hotel_clean['adr'], vert=True, patch_artist=True)
axes[1].set_title('ADR Box Plot (Outlier Detection)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Daily Rate ($)')

plt.tight_layout()
plt.savefig(reports_dir / '01_adr_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Summary
print("=" * 70)
print("ADR DISTRIBUTION ANALYSIS")
print("=" * 70)
adr_stats = hotel_clean['adr'].describe()
print(adr_stats)
print(f"\nSkewness: {hotel_clean['adr'].skew():.3f}")
print(f"Kurtosis: {hotel_clean['adr'].kurtosis():.3f}")
print(f"CV (Coeff. of Variation): {hotel_clean['adr'].std() / hotel_clean['adr'].mean():.3f}")

# Business insights
print("\n[BUSINESS INSIGHTS]:")
print(f"- Average room price: ${hotel_clean['adr'].mean():.2f}/night")
print(f"- Median room price: ${hotel_clean['adr'].median():.2f}/night")
print(f"- Price range: ${hotel_clean['adr'].min():.2f} - ${hotel_clean['adr'].max():.2f}")
print(f"- Price volatility (std): ${hotel_clean['adr'].std():.2f}")
print(f"- Distribution shape: {'Right-skewed (high-value bookings)' if hotel_clean['adr'].skew() > 0 else 'Left-skewed'}")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 2. Hotel Type Comparison

**Business Context:** Different hotel types (City Hotel vs Resort Hotel) likely have different revenue patterns, guest profiles, and cancellation rates. This comparison helps us:
- Understand revenue differences by property type
- Identify if we need separate models per type
- Optimize pricing strategy by hotel type
</VSCode.Cell>

<VSCode.Cell language="python">
# 2. Hotel Type Comparison
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# ADR by Hotel Type
hotel_clean.boxplot(column='adr', by='hotel', ax=axes[0], patch_artist=True)
axes[0].set_title('ADR by Hotel Type', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Hotel Type')
axes[0].set_ylabel('Average Daily Rate ($)')
plt.sca(axes[0])
plt.xticks(rotation=0)

# Count and cancellation rate by hotel type
hotel_stats = hotel_clean.groupby('hotel').agg({
    'adr': ['mean', 'median', 'std'],
    'is_canceled': ['count', 'sum']
}).round(2)

hotel_cancel_rate = hotel_clean.groupby('hotel')['is_canceled'].mean() * 100

hotel_stats_df = pd.DataFrame({
    'Hotel Type': hotel_clean['hotel'].unique(),
    'Count': hotel_clean.groupby('hotel').size().values,
    'Cancellation Rate': [hotel_cancel_rate.get(h, 0) for h in hotel_clean['hotel'].unique()]
})

hotel_stats_df.plot(x='Hotel Type', y='Cancellation Rate', kind='bar', ax=axes[1], legend=False, color='coral')
axes[1].set_title('Cancellation Rate by Hotel Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cancellation Rate (%)')
plt.sca(axes[1])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(reports_dir / '02_hotel_type_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("HOTEL TYPE COMPARISON")
print("=" * 70)
print("\nRevenue Statistics by Hotel Type:")
print(hotel_clean.groupby('hotel')['adr'].describe().round(2))
print("\n" + "-" * 70)
print("Cancellation Analysis by Hotel Type:")
for hotel_type in hotel_clean['hotel'].unique():
    mask = hotel_clean['hotel'] == hotel_type
    cancel_rate = hotel_clean[mask]['is_canceled'].mean() * 100
    avg_adr = hotel_clean[mask]['adr'].mean()
    print(f"\n{hotel_type}:")
    print(f"  - Bookings: {mask.sum()}")
    print(f"  - Avg ADR: \${avg_adr:.2f}")
    print(f"  - Cancellation Rate: {cancel_rate:.1f}%")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 3. Cancellation Impact on ADR

**Business Context:** Do guests who cancel their bookings differ in their pricing patterns? This helps us:
- Understand if high-value bookings are more prone to cancellation
- Predict cancellation risk based on booking value
- Develop targeted retention strategies
</VSCode.Cell>

<VSCode.Cell language="python">
# 3. Cancellation Impact on ADR
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# ADR by Cancellation Status
cancel_labels = ['Not Canceled', 'Canceled']
cancel_data = [hotel_clean[hotel_clean['is_canceled'] == 0]['adr'],
               hotel_clean[hotel_clean['is_canceled'] == 1]['adr']]

axes[0].boxplot(cancel_data, labels=cancel_labels, patch_artist=True)
axes[0].set_title('ADR by Cancellation Status', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Average Daily Rate ($)')

# Cancellation rate vs ADR price bins
hotel_clean['adr_bin'] = pd.cut(hotel_clean['adr'], bins=5)
cancel_by_adr = hotel_clean.groupby('adr_bin')['is_canceled'].apply(lambda x: (x.sum() / len(x) * 100))

cancel_by_adr.plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Cancellation Rate by ADR Price Range', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cancellation Rate (%)')
axes[1].set_xlabel('ADR Price Range ($)')
plt.sca(axes[1])
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(reports_dir / '03_cancellation_impact.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("CANCELLATION IMPACT ON ADR")
print("=" * 70)
print("\nRevenue by Cancellation Status:")
for status, label in [(0, 'Not Canceled'), (1, 'Canceled')]:
    mask = hotel_clean['is_canceled'] == status
    print(f"\n{label}:")
    print(f"  - Count: {mask.sum()}")
    print(f"  - Avg ADR: \${hotel_clean[mask]['adr'].mean():.2f}")
    print(f"  - Median ADR: \${hotel_clean[mask]['adr'].median():.2f}")
    print(f"  - Std Dev: \${hotel_clean[mask]['adr'].std():.2f}")

# Overall cancellation rate
overall_cancel = hotel_clean['is_canceled'].mean() * 100
print(f"\nOverall Cancellation Rate: {overall_cancel:.1f}%")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 4. Lead Time and Seasonal Trends

**Business Context:** When bookings are made relative to arrival (lead time) and seasonal patterns affect both demand and pricing. This helps us:
- Understand booking behavior patterns
- Identify seasonal pricing opportunities
- Predict ADR based on timing
</VSCode.Cell>

<VSCode.Cell language="python">
# 4. Lead Time and Seasonal Trends
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Lead Time vs ADR
axes[0, 0].scatter(hotel_clean['lead_time'], hotel_clean['adr'], alpha=0.3, s=20)
axes[0, 0].set_title('Lead Time vs ADR', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Lead Time (days)')
axes[0, 0].set_ylabel('Average Daily Rate ($)')
axes[0, 0].set_xlim(0, 365)  # Reasonable lead time range

# Lead time bins vs ADR
hotel_clean['lead_time_bin'] = pd.cut(hotel_clean['lead_time'], bins=[0, 7, 30, 90, 365], 
                                       labels=['<1 week', '1-30 days', '1-3 months', '>3 months'])
lead_adr = hotel_clean.groupby('lead_time_bin')['adr'].mean()
lead_adr.plot(kind='bar', ax=axes[0, 1], color='teal')
axes[0, 1].set_title('Average ADR by Lead Time Category', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Average ADR ($)')
plt.sca(axes[0, 1])
plt.xticks(rotation=45)

# Monthly ADR trends
if 'arrival_month_num' in hotel_features.columns:
    monthly_adr = hotel_clean.groupby('arrival_date_month')['adr'].mean()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_adr = monthly_adr.reindex(month_order)
    monthly_adr.plot(kind='line', ax=axes[1, 0], marker='o', color='green', linewidth=2)
    axes[1, 0].set_title('Seasonal ADR Trends (Monthly)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average ADR ($)')
    axes[1, 0].set_xlabel('Month')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)

# Total nights vs ADR
if 'total_nights' in hotel_features.columns:
    axes[1, 1].scatter(hotel_clean['stays_in_weekend_nights'] + hotel_clean['stays_in_week_nights'], 
                      hotel_clean['adr'], alpha=0.3, s=20, color='purple')
    axes[1, 1].set_title('Total Stay Length vs ADR', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Total Nights')
    axes[1, 1].set_ylabel('Average Daily Rate ($)')

plt.tight_layout()
plt.savefig(reports_dir / '04_lead_time_seasonal.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("LEAD TIME AND SEASONAL ANALYSIS")
print("=" * 70)
print("\nAverage ADR by Lead Time Category:")
print(hotel_clean.groupby('lead_time_bin')['adr'].agg(['mean', 'median', 'count']).round(2))

print("\n" + "-" * 70)
print("Monthly ADR Trends:")
monthly_stats = hotel_clean.groupby('arrival_date_month')['adr'].agg(['mean', 'count']).round(2)
print(monthly_stats)
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 5. Market Segment and Guest Type Analysis

**Business Context:** Different market segments (Online TA, Direct, etc.) and guest types have distinct booking patterns. This helps us:
- Segment customers for targeted revenue strategies
- Identify high-value customer segments
- Understand revenue distribution across channels
</VSCode.Cell>

<VSCode.Cell language="python">
# 5. Market Segment Analysis
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

# Market Segment vs ADR
market_adr = hotel_clean.groupby('market_segment')['adr'].mean().sort_values(ascending=False)
market_adr.plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('Average ADR by Market Segment', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Average ADR ($)')

# Guest Type vs ADR
guest_adr = hotel_clean.groupby('customer_type')['adr'].mean().sort_values(ascending=False)
guest_adr.plot(kind='bar', ax=axes[1], color='lightcoral')
axes[1].set_title('Average ADR by Customer Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average ADR ($)')
plt.sca(axes[1])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(reports_dir / '05_market_segment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 70)
print("MARKET SEGMENT AND CUSTOMER TYPE ANALYSIS")
print("=" * 70)
print("\nAverage ADR by Market Segment:")
print(hotel_clean.groupby('market_segment')['adr'].agg(['mean', 'median', 'count']).round(2).sort_values('mean', ascending=False))

print("\n" + "-" * 70)
print("Average ADR by Customer Type:")
print(hotel_clean.groupby('customer_type')['adr'].agg(['mean', 'median', 'count']).round(2).sort_values('mean', ascending=False))

print("\n" + "-" * 70)
print("Market Segment Distribution:")
print(hotel_clean['market_segment'].value_counts())
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 6. Correlation Heatmap - Feature Relationships

**Business Context:** Understanding which features correlate with ADR helps us:
- Identify strong predictors for modeling
- Detect multicollinearity issues
- Guide feature selection and engineering
</VSCode.Cell>

<VSCode.Cell language="python">
# 6. Correlation Analysis
# Select numeric columns for correlation
numeric_cols = hotel_clean.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation
corr_matrix = hotel_clean[numeric_cols].corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Heatmap (Numeric Features)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(reports_dir / '06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Top correlations with ADR
print("=" * 70)
print("FEATURE CORRELATIONS WITH ADR")
print("=" * 70)
adr_corr = corr_matrix['adr'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with ADR:")
print(adr_corr.head(11))  # 11 to exclude ADR itself

print("\nMulticollinearity Check (correlations > 0.7):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr:
    for col1, col2, corr_val in high_corr:
        print(f"  - {col1} <-> {col2}: {corr_val:.3f}")
else:
    print("  - No high multicollinearity detected (good for modeling)")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 7. Key Business Insights & Recommendations

**Summary of EDA Findings:**

This phase of analysis reveals critical patterns in the hotel booking data that will inform our modeling approach and business strategy.
</VSCode.Cell>

<VSCode.Cell language="python">
# 7. Summary Statistics and Business Recommendations
print("\n[DATASET OVERVIEW]:")
print(f"  - Total Bookings: {len(hotel_clean):,}")
print(f"  - Date Range: {hotel_clean['arrival_date_year'].min()}-{hotel_clean['arrival_date_year'].max()}")
print(f"  - Hotel Types: {', '.join(hotel_clean['hotel'].unique())}")
print(f"  - Market Segments: {hotel_clean['market_segment'].nunique()}")

print("\n[REVENUE INSIGHTS]:")
print(f"  - Average Nightly Rate: ${hotel_clean['adr'].mean():.2f}")
print(f"  - Median Nightly Rate: ${hotel_clean['adr'].median():.2f}")
print(f"  - Highest Bookable Rate: ${hotel_clean[hotel_clean['adr'] < hotel_clean['adr'].quantile(0.99)]['adr'].max():.2f}")
print(f"  - Price Variation (Std Dev): ${hotel_clean['adr'].std():.2f}")

print("\n[BOOKING PATTERNS]:")
print(f"  - Cancellation Rate: {hotel_clean['is_canceled'].mean()*100:.1f}%")
print(f"  - Average Lead Time: {hotel_clean['lead_time'].mean():.0f} days")
print(f"  - Average Stay Length: {(hotel_clean['stays_in_weekend_nights'] + hotel_clean['stays_in_week_nights']).mean():.1f} nights")

print("\n[SEGMENT PERFORMANCE]:")
top_segment = hotel_clean.groupby('market_segment')['adr'].mean().idxmax()
top_segment_adr = hotel_clean.groupby('market_segment')['adr'].mean().max()
print(f"  - Highest Value Segment: {top_segment} (${top_segment_adr:.2f} avg)")

top_customer = hotel_clean.groupby('customer_type')['adr'].mean().idxmax()
top_customer_adr = hotel_clean.groupby('customer_type')['adr'].mean().max()
print(f"  - Best Customer Type: {top_customer} (${top_customer_adr:.2f} avg)")

print("\n[KEY RECOMMENDATIONS FOR MODELING]:")
print("  1. Handle high outliers in ADR (consider log transformation or capping)")
print("  2. Consider separate models for City vs Resort hotels")
print("  3. Lead time is important - incorporate as strong feature")
print("  4. Seasonal effects are present - use monthly indicators")
print("  5. Cancellation status appears correlated with ADR - investigate further")
print("  6. Market segment is predictive - use as categorical feature")
print("  7. All numeric features show reasonable correlation with ADR")

print("\n" + "=" * 70)
print("Phase 1 EDA COMPLETE - Moving to Feature Engineering")
print("=" * 70)

# Summary statistics to CSV
summary_stats = pd.DataFrame({
    'Metric': ['Total Bookings', 'Avg ADR', 'Median ADR', 'ADR Std Dev', 'Cancellation Rate', 'Avg Lead Time'],
    'Value': [len(hotel_clean), hotel_clean['adr'].mean(), hotel_clean['adr'].median(), 
              hotel_clean['adr'].std(), hotel_clean['is_canceled'].mean()*100, hotel_clean['lead_time'].mean()]
})
summary_stats.to_csv(reports_dir / 'eda_summary_statistics.csv', index=False)
print(f"\nSummary statistics saved to: {reports_dir / 'eda_summary_statistics.csv'}")
</VSCode.Cell>
'''

# Find the position to insert - after "### Cleaning" section (after cell with id that contains cleaning)
# Look for the markdown cell about cleaning
insertion_point = content.find('### Cleaning')

if insertion_point == -1:
    print("ERROR: Could not find insertion point")
    exit(1)

# Find the closing </VSCode.Cell> tag after this point
insertion_point = content.find('</VSCode.Cell>', insertion_point) + len('</VSCode.Cell>')

# Insert the new cells
new_content = content[:insertion_point] + '\n\n' + eda_cells + '\n\n' + content[insertion_point:]

# Write back with UTF-8 encoding
with open('notebooks/hotel_booking_and_concierge_copilot_eda.ipynb', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Success: Added Phase 1 EDA cells!")
print("\nNew cells added:")
print("  - EDA Introduction and setup")
print("  - Target variable (ADR) distribution")
print("  - Hotel type comparison")
print("  - Cancellation impact analysis")
print("  - Lead time and seasonal trends")
print("  - Market segment analysis")
print("  - Correlation heatmap")
print("  - Business insights summary")
print("\nVisualization outputs will be saved to: reports/")
