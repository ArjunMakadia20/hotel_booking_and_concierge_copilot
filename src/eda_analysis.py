"""
Phase 1 EDA Analysis Module
Generates 8-10 focused visualizations for hotel booking demand prediction.
Focus areas: ADR distribution, missing values, hotel types, cancellations, 
lead time, seasonality, market segments, correlations, categoricals, business insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_adr_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 1: Target variable (adr) distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['adr'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Average Daily Rate (ADR) ($)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribution of ADR (All Values)', fontsize=12, fontweight='bold')
    axes[0].axvline(df['adr'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["adr"].mean():.2f}')
    axes[0].axvline(df['adr'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["adr"].median():.2f}')
    axes[0].legend()
    
    adr_filtered = df[df['adr'] <= df['adr'].quantile(0.95)]['adr']
    adr_filtered.plot.kde(ax=axes[1], color='darkblue', linewidth=2)
    axes[1].fill_between(axes[1].get_lines()[0].get_xdata(), axes[1].get_lines()[0].get_ydata(), alpha=0.3, color='steelblue')
    axes[1].set_xlabel('Average Daily Rate (ADR) ($)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1].set_title('ADR Distribution (95th percentile, KDE)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '01_adr_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_adr_distribution.png")


def plot_missing_values(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 2: Missing values visualization."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✓ No missing values found - skipping missing values plot")
        return
    
    missing_pct = (missing / len(df) * 100).round(2)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(missing)), missing_pct.values, color='coral', edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing.index)
    ax.set_xlabel('Percentage of Missing Values (%)', fontsize=11, fontweight='bold')
    ax.set_title('Data Completeness: Missing Values by Feature', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, pct) in enumerate(zip(bars, missing_pct.values)):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / '02_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_missing_values.png")


def plot_hotel_type_analysis(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 3: Hotel type vs ADR comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    adr_q95 = df['adr'].quantile(0.95)
    plot_df = df[df['adr'] <= adr_q95]
    
    hotel_types = sorted(plot_df['hotel'].unique())
    colors = sns.color_palette('Set2', len(hotel_types))
    
    box_plot = ax.boxplot([plot_df[plot_df['hotel'] == ht]['adr'].values for ht in hotel_types],
                          labels=hotel_types,
                          patch_artist=True,
                          widths=0.6)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Average Daily Rate (ADR) ($)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Hotel Type', fontsize=11, fontweight='bold')
    ax.set_title('ADR by Hotel Type (95th percentile)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, ht in enumerate(hotel_types):
        mean_adr = plot_df[plot_df['hotel'] == ht]['adr'].mean()
        ax.text(i+1, mean_adr, f'μ=${mean_adr:.0f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / '03_hotel_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_hotel_type_analysis.png")


def plot_cancellation_impact(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 4: Cancellation impact on ADR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cancellation_rate = df.groupby('hotel')['is_canceled'].mean() * 100
    cancellation_rate.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    axes[0].set_title('Cancellation Rate by Hotel Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cancellation Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Hotel Type', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    for i, v in enumerate(cancellation_rate.values):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    adr_q95 = df['adr'].quantile(0.95)
    plot_df = df[df['adr'] <= adr_q95]
    
    cancelled_adr = plot_df[plot_df['is_canceled'] == 1]['adr']
    not_cancelled_adr = plot_df[plot_df['is_canceled'] == 0]['adr']
    
    data_to_plot = [not_cancelled_adr, cancelled_adr]
    bp = axes[1].boxplot(data_to_plot, labels=['Completed', 'Cancelled'], patch_artist=True, widths=0.6)
    
    colors_box = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    axes[1].set_ylabel('Average Daily Rate (ADR) ($)', fontsize=11, fontweight='bold')
    axes[1].set_title('ADR: Completed vs Cancelled Bookings', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / '04_cancellation_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_cancellation_impact.png")


def plot_lead_time_analysis(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 5: Lead time vs ADR."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lead_time_q95 = df['lead_time'].quantile(0.95)
    adr_q95 = df['adr'].quantile(0.95)
    plot_df = df[(df['lead_time'] <= lead_time_q95) & (df['adr'] <= adr_q95)]
    
    scatter = ax.scatter(plot_df['lead_time'], plot_df['adr'], 
                        alpha=0.4, s=20, c=plot_df['is_canceled'], 
                        cmap='RdYlGn_r', edgecolors='none')
    
    z = np.polyfit(plot_df['lead_time'], plot_df['adr'], 1)
    p = np.poly1d(z)
    ax.plot(plot_df['lead_time'].sort_values(), p(plot_df['lead_time'].sort_values()), 
            "r--", linewidth=2, label=f'Trend (y={z[0]:.4f}x+{z[1]:.0f})')
    
    ax.set_xlabel('Lead Time (days)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Daily Rate (ADR) ($)', fontsize=11, fontweight='bold')
    ax.set_title('Lead Time vs ADR (95th percentile)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cancelled', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path / '05_lead_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_lead_time_analysis.png")


def plot_seasonal_trends(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 6: Monthly/seasonal ADR trends."""
    if 'arrival_date_month' not in df.columns:
        print("✓ Skipping seasonal trends plot - no date information available")
        return
    
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    df_temp = df.copy()
    df_temp['month_num'] = df_temp['arrival_date_month'].map(month_map)
    
    monthly_adr = df_temp.groupby('month_num')['adr'].mean().sort_index()
    monthly_bookings = df_temp.groupby('month_num').size().sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(monthly_adr.index, monthly_adr.values, marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[0].fill_between(monthly_adr.index, monthly_adr.values, alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Month', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Average ADR ($)', fontsize=11, fontweight='bold')
    axes[0].set_title('Seasonal ADR Trends (Mean by Month)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(alpha=0.3)
    
    axes[1].bar(monthly_bookings.index, monthly_bookings.values, color='coral', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Month', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Number of Bookings', fontsize=11, fontweight='bold')
    axes[1].set_title('Booking Volume by Month (Demand Pattern)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig(save_path / '06_seasonal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_seasonal_trends.png")


def plot_market_segment_analysis(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 7: Market segment analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'customer_type' in df.columns:
        customer_revenue = df.groupby('customer_type')['adr'].mean().sort_values(ascending=False)
        
        axes[0].bar(range(len(customer_revenue)), customer_revenue.values, 
                   color=sns.color_palette('husl', len(customer_revenue)), 
                   alpha=0.8, edgecolor='black')
        axes[0].set_xticks(range(len(customer_revenue)))
        axes[0].set_xticklabels(customer_revenue.index, rotation=45, ha='right')
        axes[0].set_ylabel('Average ADR ($)', fontsize=11, fontweight='bold')
        axes[0].set_title('Average ADR by Customer Type', fontsize=12, fontweight='bold')
        
        for i, v in enumerate(customer_revenue.values):
            axes[0].text(i, v + 2, f'${v:.0f}', ha='center', fontweight='bold', fontsize=9)
    
    if 'distribution_channel' in df.columns:
        dist_channel_count = df['distribution_channel'].value_counts().head(6)
        dist_channel_count.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                               colors=sns.color_palette('Set2', len(dist_channel_count)))
        axes[1].set_title('Bookings by Distribution Channel (Top 6)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(save_path / '07_market_segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 07_market_segment_analysis.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 8: Correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'adr' not in numeric_cols:
        print("✓ Skipping correlation heatmap - adr not numeric")
        return
    
    correlations = df[numeric_cols].corr()['adr'].sort_values(ascending=False)
    top_features = correlations.head(15).index.tolist()
    
    corr_matrix = df[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap (Top 15 features with ADR)', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path / '08_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 08_correlation_heatmap.png")


def plot_categorical_summaries(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 9: Top categorical features summary."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    key_cats = [col for col in categorical_cols 
                if col not in ['email', 'phone_number', 'name', 'reservation_status_date']][:4]
    
    if not key_cats:
        print("✓ Skipping categorical summaries - no categorical features")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(key_cats):
        if idx < len(axes):
            top_vals = df[col].value_counts().head(8)
            top_vals.plot(kind='barh', ax=axes[idx], color='steelblue', alpha=0.8, edgecolor='black')
            axes[idx].set_title(f'Top Values: {col}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Count', fontsize=10)
            
            for i, v in enumerate(top_vals.values):
                axes[idx].text(v + 50, i, str(v), va='center', fontweight='bold', fontsize=9)
    
    for idx in range(len(key_cats), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / '09_categorical_summaries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 09_categorical_summaries.png")


def generate_business_insights(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 10: Key business insights summary."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    insights = []
    insights.append("KEY BUSINESS INSIGHTS - Hotel Booking Data")
    insights.append("=" * 60)
    insights.append("")
    
    insights.append("DATASET OVERVIEW")
    insights.append(f"  Total Bookings: {len(df):,}")
    insights.append(f"  Hotel Types: {df['hotel'].nunique() if 'hotel' in df.columns else 'N/A'}")
    insights.append("")
    
    insights.append("REVENUE METRICS")
    total_revenue = (df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])).sum() if 'stays_in_weekend_nights' in df.columns else df['adr'].sum()
    insights.append(f"  Average Daily Rate (ADR): ${df['adr'].mean():.2f}")
    insights.append(f"  Median ADR: ${df['adr'].median():.2f}")
    insights.append(f"  ADR Range: ${df['adr'].min():.2f} to ${df['adr'].max():.2f}")
    insights.append(f"  Total Revenue Estimate: ${total_revenue:,.0f}")
    insights.append("")
    
    insights.append("RISK METRICS")
    cancel_rate = (df['is_canceled'].sum() / len(df) * 100) if 'is_canceled' in df.columns else 0
    insights.append(f"  Cancellation Rate: {cancel_rate:.1f}%")
    insights.append(f"  Cancelled Bookings: {df['is_canceled'].sum() if 'is_canceled' in df.columns else 'N/A'}")
    insights.append("")
    
    insights.append("BOOKING PATTERNS")
    insights.append(f"  Average Lead Time: {df['lead_time'].mean():.0f} days")
    insights.append(f"  Median Lead Time: {df['lead_time'].median():.0f} days")
    insights.append("")
    
    insights.append("DATA QUALITY")
    insights.append(f"  Missing Values: {df.isnull().sum().sum()}")
    insights.append(f"  Features: {df.shape[1]}")
    insights.append("")
    insights.append("=" * 60)
    
    text = '\n'.join(insights)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path / '10_business_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 10_business_insights.png")


def run_eda_analysis(df: pd.DataFrame, output_dir: Path = None) -> None:
    """Run all Phase 1 EDA analysis and save visualizations."""
    if output_dir is None:
        output_dir = Path('reports')
    
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 1 EDA ANALYSIS - Generating 10 Visualizations")
    print("="*60 + "\n")
    
    plot_adr_distribution(df, output_dir)
    plot_missing_values(df, output_dir)
    plot_hotel_type_analysis(df, output_dir)
    plot_cancellation_impact(df, output_dir)
    plot_lead_time_analysis(df, output_dir)
    plot_seasonal_trends(df, output_dir)
    plot_market_segment_analysis(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_categorical_summaries(df, output_dir)
    generate_business_insights(df, output_dir)
    
    print("\n" + "="*60)
    print(f"✓ All EDA visualizations saved to: {output_dir.absolute()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data_loader import load_hotel_booking_data, get_default_hotel_booking_path
    from data_processing import clean_hotel_booking_data
    
    data_path = get_default_hotel_booking_path()
    df = load_hotel_booking_data(data_path)
    df_clean = clean_hotel_booking_data(df)
    
    run_eda_analysis(df_clean)
