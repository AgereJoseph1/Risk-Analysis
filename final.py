import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import inch
import io
from openpyxl.styles import PatternFill

###########################
# 1. CONFIG & THRESHOLDS
###########################
RISK_COLORS = {
    1: ('#006400', 'Very Low'),  # Deep green
    2: ('#90EE90', 'Low'),       # Light green
    3: ('#FFD700', 'Medium'),    # Yellow
    4: ('#FF0000', 'High'),      # Red
    5: ('#722F37', 'Extreme')    # Wine
}

THRESHOLD_RULES = {
    'time': {
        2: 1,    # ≤ 2 days => 1
        3: 2,
        4: 3,
        5: 4,
        float('inf'): 5
    },
    'percentage': {
        10: 1,   # ≤ 10% => 1
        11: 2,
        25: 3,
        26: 4,
        float('inf'): 5
    },
    'target': {
        10: 1,  
        20: 2,
        30: 3,
        40: 4,
        float('inf'): 5
    }
}

st.set_page_config(
    layout="wide",
    page_title="Risk Intelligence Dashboard Hub",
    page_icon="🎯"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .dashboard-header {
            padding: 2.5rem 2rem;
            background: linear-gradient(135deg, #1E3F66 0%, #2E5090 50%, #3A63B8 100%);
            color: white;
            border-radius: 15px;
            margin: 1rem 0 3rem 0;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            text-align: center;
        }
        .dashboard-header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .dashboard-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
        }
        .metric-card {
            background: white;
            padding: 1.8rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .stButton > button {
            background-color: #1E3F66;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
            
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

CALIBRATION_RULES = {
    5: 1,   # 1–5 => 1
    10: 2,  # 6–10 => 2
    15: 3,  # 11–15 => 3
    20: 4,  # 16–20 => 4
    25: 5   # 21–25 => 5
}

ALERT_EMAIL_THRESHOLD = 4

###########################
# 2. RESIDUAL & CALIBRATION
###########################
def calculate_residual_risk(current_risk, previous_risk):
    if pd.isna(previous_risk):
        return 0
    return current_risk - previous_risk

def calibrate_total_risk(total_risk):
    for threshold, category in CALIBRATION_RULES.items():
        if total_risk <= threshold:
            return category
    return max(CALIBRATION_RULES.values())

# def send_email_alert(risk_item):
#     st.info(f"Email Alert Sent for Risk ID: {risk_item['Risk ID']} - {risk_item['Rating']}")

def predict_risk_trends(df):
    risk_trend = df.groupby('Type of Risk')['Risk_Score'].mean() + np.random.uniform(-0.5, 0.5, len(df['Type of Risk'].unique()))
    return risk_trend

###########################
# 3. PARSE & PROCESS
###########################
def categorize_threshold_type(text):
    text = str(text).lower()
    if any(w in text for w in ['day','working day','week']):
        return 'time'
    elif any(w in text for w in ['%','percent','percentage','target']):
        return 'percentage'
    return 'time'

def extract_numeric_value(value, threshold_type):
    if pd.isna(value):
        return 0
    text = str(value).lower()
    if threshold_type == 'time':
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:day|week)', text)
        return float(matches[0]) if matches else 0
    elif threshold_type == 'percentage':
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        return float(matches[0]) if matches else 0
    return 0

def calculate_risk_score(value, threshold_type):
    rules = THRESHOLD_RULES.get(threshold_type, THRESHOLD_RULES['time'])
    score = 5  # default maximum
    for threshold, risk_score in rules.items():
        if value <= threshold:
            score = risk_score
            break
    # Ensure the score doesn't exceed 25
    return min(score, 25)

def process_data(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Determine the Limitation column based on the Appetite Score
    def determine_limitation(appetite_score):
        if 0 <= appetite_score < 10:
            return 1
        elif 10 <= appetite_score < 11:
            return 2
        elif 11 <= appetite_score < 25:
            return 3
        elif 25 <= appetite_score < 26:
            return 4
        elif 26 <= appetite_score <= 45:
            return 5
        else:
            return 5  # Default value if appetite_score is out of bounds

    # Apply the determine_limitation function to the Appetite Score column
    df['Limitation'] = df['Appetite Score'].apply(determine_limitation)

    # Map Limitation values to Risk Levels using RISK_COLORS
    df['Rating'] = df['Limitation'].map({
        1: RISK_COLORS[1][1],  # Very Low
        2: RISK_COLORS[2][1],  # Low
        3: RISK_COLORS[3][1],  # Medium
        4: RISK_COLORS[4][1],  # High
        5: RISK_COLORS[5][1]   # Extreme
    })

    # Map Limitation values to Risk Colors using RISK_COLORS
    df['Risk_Color'] = df['Limitation'].map({
        1: RISK_COLORS[1][0],  # Deep green
        2: RISK_COLORS[2][0],  # Light green
        3: RISK_COLORS[3][0],  # Yellow
        4: RISK_COLORS[4][0],  # Red
        5: RISK_COLORS[5][0]   # Wine
    })

    # Calculate the Risk Score column by multiplying Limitation and Impact
    df['Risk_Score'] = df['Limitation'] * df['Impact']

    # Return the processed DataFrame
    return df

def process_data_with_residuals(df, previous_data=None):
    processed_df = process_data(df)
    
    # Ensure Total_Risk doesn't exceed 25
    processed_df['Risk_Score'] = processed_df['Risk_Score'].clip(upper=25)
    
    return processed_df

def product_color(prod):
    """
    1..6 => green
    7..12 => yellow
    13..18 => red
    19..25 => wine
    """
    if 1 <= prod <= 6:
        return "#008000"  # Green
    elif 7 <= prod <= 12:
        return "#FFFF00"  # Yellow
    elif 13 <= prod <= 18:
        return "#FF0000"  # Red
    else:
        return "#722F37"  # Wine

###########################
# 4. 5X5 MATRIX
###########################
def create_5x5_matrix_chart(df):
    """
    Build a 5×5 grid of squares: (Likelihood=1..5, Impact=1..5).
    The text in each cell = how many items in df have (L,I).
    Color by L×I => 1..25 => green, yellow, red, wine.
    Counts are capped at 25 per cell.
    """

    # Create a local copy of the DataFrame to avoid modifying the original
    df_local = df.copy()

    # Fallback approach
    if 'Likelihood' not in df_local.columns or 'Impact' not in df_local.columns:
        # Handle NaN values in Risk_Score by filling them with 0 or another default value
        df_local['Risk_Score'] = df_local['Risk_Score'].fillna(0)  # Fill NaN with 0
        df_local['Likelihood'] = df_local['Risk_Score'].clip(upper=5)
        df_local['Impact'] = 1

    # Tally with cap at 25
    counts = [[0] * 5 for _ in range(5)]
    for _, row in df_local.iterrows():
        # Ensure Likelihood and Impact are valid integers
        l = int(row['Likelihood']) if not pd.isna(row['Likelihood']) else 0
        i = int(row['Impact']) if not pd.isna(row['Impact']) else 0
        if 1 <= l <= 5 and 1 <= i <= 5:
            counts[l - 1][i - 1] = min(counts[l - 1][i - 1] + 1, 25)  # Cap at 25

    fig = go.Figure()
    for l in range(1, 6):
        for i in range(1, 6):
            product = l * i
            color = product_color(product)
            count = counts[l - 1][i - 1]

            # Add '+' symbol if count is 25
            display_text = f"25" if count == 25 else str(count)

            x0, x1 = i - 1, i
            y0, y1 = 5 - l, 5 - l + 1

            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                fillcolor=color,  # Set the background color
                line=dict(color="#FFFFFF", width=2)
            )

            # Set text color based on background color
            text_color = "#FFFFFF" if color in ["#FF0000", "#722F37"] else "#000000"
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=display_text,
                font=dict(color=text_color, size=14),
                showarrow=False
            )

    fig.update_xaxes(
        range=[0, 5],
        tickmode="array",
        tickvals=[0.5, 1.5, 2.5, 3.5, 4.5],
        ticktext=["1", "2", "3", "4", "5"],
        side="top"
    )
    fig.update_yaxes(
        range=[0, 5],
        tickmode="array",
        tickvals=[0.5, 1.5, 2.5, 3.5, 4.5],
        ticktext=["5", "4", "3", "2", "1"],
        scaleanchor="x",
        scaleratio=1
    )
    fig.update_layout(
        width=500, height=500,
        plot_bgcolor="#FFF",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_risk_distribution_colored_table(df):
    """
    Creates a colored table to visualize the distribution of risk levels for each type of risk.
    Rows: Type of Risk
    Columns: Risk Levels
    Each cell is colored based on its corresponding risk level.
    Padding is added between cells for better visual separation.
    Zero values are replaced with '-'.
    Text size and cell height are increased for better readability.
    The sum of each row is added as the last cell.
    Text color is white for the "High" and "Extreme" columns.
    Returns:
        - fig: The Plotly table figure.
        - risk_distribution: The DataFrame used to create the table.
    """
    # Define all possible risk levels in the correct order
    all_ratings = ['Very Low', 'Low', 'Medium', 'High', 'Extreme']
    
    # Create the crosstab
    risk_distribution = pd.crosstab(df['Type of Risk'], df['Rating'])
    
    # Reindex the crosstab to include all risk levels, filling missing values with 0
    risk_distribution = risk_distribution.reindex(columns=all_ratings, fill_value=0)
    
    # Replace zero values with '-'
    risk_distribution = risk_distribution.replace(0, '-')
    
    # Calculate the sum of each row
    row_sums = risk_distribution.replace('-', 0).sum(axis=1)
    risk_distribution['Total'] = row_sums  # Add the sum as a new column
    
    # Map Risk Levels to their corresponding colors
    rating_mapping = {
        'Very Low': 1,
        'Low': 2,
        'Medium': 3,
        'High': 4,
        'Extreme': 5
    }
    risk_colors = {level: RISK_COLORS[rating_mapping[level]][0] for level in all_ratings}
    
    # Define font colors for each column
    font_colors = {
        'Very Low': 'black',
        'Low': 'black',
        'Medium': 'black',
        'High': 'white',  # White text for "High"
        'Extreme': 'white',  # White text for "Extreme"
        'Total': 'black'  # Black text for "Total"
    }
    
    # Create the colored table
    fig = go.Figure(data=go.Table(
        header=dict(
            values=['Type of Risk'] + list(risk_distribution.columns),
            fill_color='lightgray',
            align='center',
            font=dict(color='black', size=14),  # Increased font size for header
            line=dict(width=1, color='white'),  # Add white borders for padding
            height=50  # Increased header height for better spacing
        ),
        cells=dict(
            values=[risk_distribution.index] + [risk_distribution[level] for level in risk_distribution.columns],
            fill_color=['lightgray'] + [[risk_colors[level]] * len(risk_distribution) for level in all_ratings] + ['lightgray'] * len(risk_distribution),  # Color for the "Total" column
            align='center',
            font=dict(color=[font_colors[level] for level in risk_distribution.columns] + ['black'], size=14),  # Set font color based on column
            line=dict(width=1, color='white'),  # Add white borders for padding
            height=40  # Increased cell height for better spacing
        )
    ))
    
    # Update layout
    fig.update_layout(
        height=600,  # Increased height to accommodate larger cells
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins for better spacing
    )
    
    return fig, risk_distribution

###########################
# 5. DISTRIBUTION & TREND
###########################
def create_risk_distribution(df):
    """
    Creates a pie chart and bar chart to visualize the distribution of risk levels.
    Colors are based on the RISK_COLORS dictionary.
    """
    # Map risk levels to their corresponding colors
    rating_mapping = {
        'Very Low': 1,
        'Low': 2,
        'Medium': 3,
        'High': 4,
        'Extreme': 5
    }
    risk_colors = {level: RISK_COLORS[rating_mapping[level]][0] for level in df['Rating'].unique()}

    # Pie Chart: Risk Level Distribution
    risk_dist = df['Rating'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_dist.index,
        values=risk_dist.values,
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=[risk_colors[level] for level in risk_dist.index])  # Use RISK_COLORS for pie chart
    )])
    fig_pie.update_layout(title="Risk Level Distribution", height=400)

    # Bar Chart: Risk Distribution by Type
    type_dist = df['Type of Risk'].value_counts()
    fig_bar = go.Figure(data=[go.Bar(
        x=type_dist.index,
        y=type_dist.values,
        marker_color='#1E3F66',  # Default color for bar chart (can be customized if needed)
        text=type_dist.values,
        textposition='auto'
    )])
    fig_bar.update_layout(
        title="Risk Distribution by Type",
        xaxis_title="Risk Type",
        yaxis_title="Count",
        height=400
    )

    return fig_pie, fig_bar

def create_trend_analysis(df):
    """
    Creates a bar chart showing the average risk score for each risk type.
    Colors are based on the RISK_COLORS dictionary.
    """
    # Calculate average risk scores
    avg_scores = df.groupby('Type of Risk')['Risk_Score'].mean().sort_values(ascending=False)

    # Map risk levels to their corresponding colors
    rating_mapping = {
        'Very Low': 1,
        'Low': 2,
        'Medium': 3,
        'High': 4,
        'Extreme': 5
    }
    risk_colors = {level: RISK_COLORS[rating_mapping[level]][0] for level in df['Rating'].unique()}

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(
        x=avg_scores.index,
        y=avg_scores.values,
        marker_color=[risk_colors[df[df['Type of Risk'] == risk_type]['Rating'].mode()[0]] for risk_type in avg_scores.index],  # Use RISK_COLORS for bar chart
        text=[f"{x:.2f}" for x in avg_scores.values],
        textposition='auto'
    )])
    fig.update_layout(
        title="Average Risk Score",
        xaxis_title="Risk Type",
        yaxis_title="Average Risk Score",
        height=400
    )

    return fig

###########################
# 6. PDF & UTILS
###########################
def create_report_visualizations(df):
    # Your existing matplotlib code
    charts = {}
    # ...
    return charts

def get_report_styles():
    # ...
    pass

def generate_comprehensive_report(df, timestamp):
    """Generate a PDF report with risk analysis"""
    # Create a buffer to store the PDF
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Initialize story (content) for the PDF
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Add title
    story.append(Paragraph(f"Risk Assessment Report", title_style))
    story.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add summary statistics
    total_risks = len(df)
    extreme_risks = len(df[df['Rating']=='Extreme'])
    high_risks = len(df[df['Rating']=='High'])
    avg_score = df['Risk_Score'].mean()
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Risks', str(total_risks)],
        ['Extreme Risks', str(extreme_risks)],
        ['High Risks', str(high_risks)],
        ['Average Risk Score', f"{avg_score:.2f}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Add Extreme and high risks table
    story.append(Paragraph("Extreme and High Risk Items", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    critical_high_df = df[df['Rating'].isin(['Extreme', 'High'])].copy()
    if len(critical_high_df) > 0:
        risk_data = [['Risk ID', 'Type', 'Description', 'Risk Level', 'Score']]
        for _, row in critical_high_df.iterrows():
            risk_data.append([
                str(row['Risk ID']),
                str(row['Type of Risk']),
                str(row['Description'])[:100] + '...',  # Truncate long descriptions
                str(row['Rating']),
                str(row['Risk_Score'])
            ])
        
        risk_table = Table(risk_data, colWidths=[80, 100, 250, 80, 60])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(risk_table)
    else:
        story.append(Paragraph("No Extreme or High risk items found.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def get_risk_severity_text(score):
    # ...
    pass

def get_impact_description(score):
    # ...
    pass

def generate_recommendations(df):
    # ...
    pass
def create_risk_scoring_scale():
    """Creates a risk scoring scale with colored line segments."""
    
    # Create base figure
    fig = go.Figure()
    
    # Define overall parameters
    y_line = 4  # y-position of main range line
    x_range = [1, 25]  # x-axis range
    
    # Define line segments with colors
    segments = [
        (1, 7, "rgb(0, 128, 0)"),      # Green segment
        (7, 13, "rgb(255, 215, 0)"),   # Yellow segment
        (13, 19, "rgb(255, 0, 0)"),    # Red segment
        (19, 25, "rgb(100, 100, 100)") # Gray segment
    ]
    
    # Add colored line segments
    for start, end, color in segments:
        fig.add_shape(
            type="line",
            x0=start, x1=end,
            y0=y_line, y1=y_line,
            line=dict(color=color, width=3)
        )
    
    # Add "Range" text
    fig.add_annotation(
        x=0, y=y_line,
        text="Range",
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        font=dict(size=10)
    )
    
    # Add scale numbers below line
    scale_numbers = [1, 5, 10, 15, 20, 25]
    for num in scale_numbers:
        fig.add_annotation(
            x=num,
            y=y_line - 0.3,
            text=str(num),
            showarrow=False,
            yanchor="top",
            font=dict(size=10)
        )
    
    # Add circle markers at transition points
    markers = [7, 13, 19]
    fig.add_trace(go.Scatter(
        x=markers,
        y=[y_line] * len(markers),
        mode='markers',
        marker=dict(
            size=8,
            color='white',
            line=dict(color='black', width=1)
        ),
        showlegend=False
    ))
    
    # Add labels (Low, Medium, High)
    labels = [
        (7, "Low"),
        (13, "Medium"),
        (19, "High")
    ]
    
    for x, label in labels:
        fig.add_annotation(
            x=x,
            y=y_line - 0.6,
            text=label,
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="top"
        )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            range=[-1, 26],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[2, 5],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        height=120,
        margin=dict(l=50, r=50, t=30, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        title=dict(
            text="Edit scoring scales",
            x=0.5,
            y=0.95,
            font=dict(size=12)
        )
    )
    
    return fig


###########################
# 7. MAIN DASHBOARD
###########################
st.markdown("""
    <div class='dashboard-header'>
        <h1>🎯 Risk Intelligence Dashboard </h1>
        <div class='dashboard-subtitle'>
            Comprehensive Risk Assessment & Analytics Platform
        </div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Risk Assessment Data",
    type=['xlsx', 'csv'],
    help="Upload your risk assessment Excel file"
)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        
        # Drop columns with default names like 'Unnamed: 11', 'Unnamed: 12', etc.
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Drop entirely empty columns (if any)
        df = df.dropna(axis=1, how='all')
        
        # Drop entirely empty rows (if any)
        df = df.dropna(axis=0, how='all')  # Add this line to skip empty rows
        
        required_columns = ['Risk#','Type of Risk','Risk ID','Description','Lower Limit','Upper Limit', 'Response Level']
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()

        previous_file = st.file_uploader("Upload Previous Risk Data (Optional)", type=['xlsx','csv'])
        previous_data = None
        if previous_file is not None:
            previous_data = pd.read_excel(previous_file) if previous_file.name.endswith('.xlsx') else pd.read_csv(previous_file)

        processed_df = process_data_with_residuals(df, previous_data)
        # Drop the Risk_Color column from the DataFrame
        processed_df = processed_df.drop(columns=['Risk_Color'], errors='ignore')
        # processed_df = processed_df.drop(columns=['Lower_Limit'], errors='ignore')
        # processed_df = processed_df.drop(columns=['Upper_Limit'], errors='ignore')
        # processed_df = processed_df.drop(columns=['Respone_Level'], errors='ignore')

        if processed_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            # Calculate metrics
            total_risks = len(processed_df)
            extreme_risks = len(processed_df[processed_df['Rating']=='Extreme'])
            high_risks = len(processed_df[processed_df['Rating']=='High'])
            avg_score = processed_df['Risk_Score'].mean()

            col1.metric("Extreme Risks", extreme_risks, f"{(extreme_risks/total_risks*100):.1f}%")
            col2.metric("High Risks", high_risks, f"{(high_risks/total_risks*100):.1f}%")
            col3.metric("Avg Risk Score", f"{avg_score:.2f}")

            tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Data"])
            
            with tab1:
                # First add the risk scoring scale
                st.subheader("Risk Scoring Scale")
                fig_scale = create_risk_scoring_scale()
                st.plotly_chart(fig_scale, use_container_width=True)
                
                # Then show the existing risk matrix
                st.subheader("Risk Matrix (Likelihood × Impact)")
                fig_5x5 = create_5x5_matrix_chart(processed_df)
                st.plotly_chart(fig_5x5, use_container_width=True)
                
                # Add the risk distribution heatmap below the risk matrix
                st.subheader("Risk Distribution by Type")
                fig_colored_table, risk_distribution = create_risk_distribution_colored_table(processed_df)
                st.plotly_chart(fig_colored_table, use_container_width=True)

            with tab2:
                colA, colB = st.columns(2)
                
                # Pie Chart and Bar Chart for Risk Distribution
                fig_pie, fig_bar = create_risk_distribution(processed_df)
                colA.plotly_chart(fig_pie, use_container_width=True)
                colB.plotly_chart(fig_bar, use_container_width=True)
                
                # Trend Analysis Bar Chart
                st.plotly_chart(create_trend_analysis(processed_df), use_container_width=True)

            with tab3:
                st.subheader("Report")
                st.dataframe(processed_df, use_container_width=True)

            # Download
            st.subheader("Report Generation")
            rc1, rc2 = st.columns(2)

            with rc1:
                st.info("Download Comprehensive Analysis Report")
                if st.button("📊 Generate Full Report"):
                    with st.spinner("Generating comprehensive report..."):
                        report_buffer = generate_comprehensive_report(
                            processed_df, 
                            datetime.now().strftime('%Y-%m-%d %H:%M')
                        )
                        st.download_button(
                            label="📑 Download Full Analysis Report",
                            data=report_buffer,
                            file_name=f"risk_analysis_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
            
            with rc2:
                st.info("Download Raw Data Report")
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer) as writer:
                    # Add the full data table to the 'Full Data' sheet
                    processed_df.to_excel(writer, sheet_name='Rating', index=False)
                    
                    # Add summary statistics to the 'Summary' sheet
                    summary_stats = pd.DataFrame({
                        'Metric': ['Total', 'Extreme', 'High', 'Avg Score'],
                        'Value': [total_risks, extreme_risks, high_risks, avg_score]
                    })
                    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Add risk distribution to the 'Risk Distribution' sheet
                    risk_distribution.to_excel(writer, sheet_name='Risk Distribution', index=False)
                    
                    # Apply background colors to the rating column in the 'Full Data' sheet
                    workbook = writer.book
                    sheet = workbook['Rating']

                    # Define color mappings for rating
                    risk_color_mapping = {
                        'Very Low': '006400',  # Deep green
                        'Low': '90EE90',       # Light green
                        'Medium': 'FFD700',    # Yellow
                        'High': 'FF0000',      # Red
                        'Extreme': '722F37'    # Wine
                    }

                    # Find the column index of rating
                    rating_col_index = processed_df.columns.tolist().index('Rating') + 1  # +1 for Excel's 1-based indexing

                    # Apply background colors to the rating column
                    for row in range(2, len(processed_df) + 2):  # Start from row 2 (skip header)
                        rating = sheet.cell(row=row, column=rating_col_index).value
                        if rating in risk_color_mapping:
                            fill = PatternFill(start_color=risk_color_mapping[rating], end_color=risk_color_mapping[rating], fill_type='solid')
                            sheet.cell(row=row, column=rating_col_index).fill = fill

                excel_buffer.seek(0)
                st.download_button(
                    label="📈 Download Excel Data",
                    data=excel_buffer,
                    file_name=f"risk_assessment_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your file format and try again.")
        st.write("Error details:", e)
else:
    st.info("Upload your risk assessment file to begin analysis")
    st.markdown("""
    ### Required Format:
    - Risk# 
    - Type of Risk 
    - Risk ID 
    - Description 
    - Lower Limit 
    - Upper Limit
    (Optional: Response Level, etc.)
    """)
