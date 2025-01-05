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

###########################
# 1. CONFIG & THRESHOLDS
###########################
RISK_COLORS = {
    1: ('#006400', 'Low'),      # Deep green
    2: ('#90EE90', 'Minor'),    # Light green
    3: ('#FFD700', 'Medium'),   # Yellow
    4: ('#FF0000', 'High'),     # Red
    5: ('#722F37', 'Critical')  # Wine
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
        15: 1,   # ≤ 15% => 1
        25: 2,
        35: 3,
        50: 4,
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
#     st.info(f"Email Alert Sent for Risk ID: {risk_item['Risk ID']} - {risk_item['Risk_Level']}")

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
    df = df.copy()
    df['Threshold_Type'] = df['Upper Limit'].apply(categorize_threshold_type)
    df['Numeric_Value'] = df.apply(
        lambda row: extract_numeric_value(row['Upper Limit'], row['Threshold_Type']), axis=1
    )
    df['Risk_Score'] = df.apply(
        lambda row: calculate_risk_score(row['Numeric_Value'], row['Threshold_Type']), axis=1
    )
    df['Risk_Color'] = df['Risk_Score'].map(lambda x: RISK_COLORS[x][0])
    df['Risk_Level'] = df['Risk_Score'].map(lambda x: RISK_COLORS[x][1])
    return df

def process_data_with_residuals(df, previous_data=None):
    processed_df = process_data(df)
    if previous_data is not None:
        processed_df['Previous_Total_Risk'] = processed_df['Risk ID'].map(previous_data.set_index('Risk ID')['Total_Risk'])
        processed_df['Residual_Risk'] = processed_df.apply(
            lambda row: calculate_residual_risk(row['Risk_Score'], row['Previous_Total_Risk']),
            axis=1
        )
    else:
        processed_df['Previous_Total_Risk'] = 0
        processed_df['Residual_Risk'] = 0
    
    # Ensure Total_Risk doesn't exceed 25
    processed_df['Risk_Score'] = processed_df['Risk_Score'].clip(upper=25)
    processed_df['Total_Risk_Category'] = processed_df['Risk_Score'].apply(calibrate_total_risk)

    critical_risks = processed_df[processed_df['Risk_Score'] >= ALERT_EMAIL_THRESHOLD]
    return processed_df

###########################
# 4. 5X5 MATRIX
###########################
def product_color(prod):
    """
    1..6 => green
    7..12 => yellow
    13..18 => red
    19..25 => black
    """
    if 1 <= prod <= 6:
        return "#008000"
    elif 7 <= prod <= 12:
        return "#FFFF00"
    elif 13 <= prod <= 18:
        return "#FF0000"
    else:
        return "#000000"

def create_5x5_matrix_chart(df):
    """
    Build a 5×5 grid of squares: (Likelihood=1..5, Impact=1..5).
    The text in each cell = how many items in df have (L,I).
    Color by L×I => 1..25 => green, yellow, red, black.
    Counts are capped at 25 per cell.
    """

    # Fallback approach
    if 'Likelihood' not in df.columns or 'Impact' not in df.columns:
        df['Likelihood'] = df['Risk_Score'].clip(upper=5)
        df['Impact'] = 1

    # Tally with cap at 25
    counts = [[0]*5 for _ in range(5)]
    for _, row in df.iterrows():
        l = int(row['Likelihood'])
        i = int(row['Impact'])
        if 1 <= l <= 5 and 1 <= i <= 5:
            counts[l-1][i-1] = min(counts[l-1][i-1] + 1, 25)  # Cap at 25

    fig = go.Figure()
    for l in range(1,6):
        for i in range(1,6):
            product = l*i
            color = product_color(product)
            count = counts[l-1][i-1]
            
            # Add '+' symbol if count is 25
            display_text = f"25" if count == 25 else str(count)
            
            x0, x1 = i-1, i
            y0, y1 = 5-l, 5-l+1

            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                fillcolor=color,
                line=dict(color="#FFFFFF", width=2)
            )

            text_color = "#FFFFFF" if color in ["#FF0000","#000000"] else "#000000"
            fig.add_annotation(
                x=(x0+x1)/2,
                y=(y0+y1)/2,
                text=display_text,
                font=dict(color=text_color,size=14),
                showarrow=False
            )

    fig.update_xaxes(
        range=[0,5],
        tickmode="array",
        tickvals=[0.5,1.5,2.5,3.5,4.5],
        ticktext=["1","2","3","4","5"],
        side="top"
    )
    fig.update_yaxes(
        range=[0,5],
        tickmode="array",
        tickvals=[0.5,1.5,2.5,3.5,4.5],
        ticktext=["5","4","3","2","1"],
        scaleanchor="x",
        scaleratio=1
    )
    fig.update_layout(
        width=500, height=500,
        plot_bgcolor="#FFF",
        margin=dict(l=20,r=20,t=60,b=20)
    )
    return fig

###########################
# 5. DISTRIBUTION & TREND
###########################
def create_risk_distribution(df):
    risk_dist = df['Risk_Level'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_dist.index,
        values=risk_dist.values,
        hole=0.4,
        textinfo='label+percent'
    )])
    fig_pie.update_layout(title="Risk Level Distribution", height=400)

    type_dist = df['Type of Risk'].value_counts()
    fig_bar = go.Figure(data=[go.Bar(
        x=type_dist.index,
        y=type_dist.values,
        marker_color='#1E3F66',
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
    avg_scores = df.groupby('Type of Risk')['Risk_Score'].mean().sort_values(ascending=False)
    fig = go.Figure(data=[go.Bar(
        x=avg_scores.index,
        y=avg_scores.values,
        marker_color='#1E3F66',
        text=[f"{x:.2f}" for x in avg_scores.values],
        textposition='auto'
    )])
    fig.update_layout(
        title="Average Risk Score by Category",
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
    # ...
    pass

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
        
        required_columns = ['Risk#','Type of Risk','Risk ID','Description','Lower Limit','Upper Limit']
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()

        previous_file = st.file_uploader("Upload Previous Risk Data (Optional)", type=['xlsx','csv'])
        previous_data = None
        if previous_file is not None:
            previous_data = pd.read_excel(previous_file) if previous_file.name.endswith('.xlsx') else pd.read_csv(previous_file)

        processed_df = process_data_with_residuals(df, previous_data)

        if processed_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            total_risks = len(processed_df)
            critical_risks = len(processed_df[processed_df['Risk_Level']=='Critical'])
            high_risks = len(processed_df[processed_df['Risk_Level']=='High'])
            avg_score = processed_df['Risk_Score'].mean()
            residual_risks = len(processed_df[processed_df['Residual_Risk']>0])

         
            col1.metric("Critical Risks", critical_risks, f"{(critical_risks/total_risks*100):.1f}%")
            col2.metric("High Risks", high_risks, f"{(high_risks/total_risks*100):.1f}%")
            col3.metric("Avg Risk Score", f"{avg_score:.2f}")
            col4.metric("Residual Risks", residual_risks)

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
                
                st.subheader("Critical & High Risk Items")
                crit_df = processed_df[processed_df['Risk_Level'].isin(['Critical','High'])]
                st.dataframe(
                    crit_df[['Risk ID','Type of Risk','Description','Risk_Level','Risk_Score']], 
                    use_container_width=True
                )

            with tab2:
                colA, colB = st.columns(2)
                fig_pie, fig_bar = create_risk_distribution(processed_df)
                colA.plotly_chart(fig_pie, use_container_width=True)
                colB.plotly_chart(fig_bar, use_container_width=True)
                st.plotly_chart(create_trend_analysis(processed_df), use_container_width=True)

            with tab3:
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
                    processed_df.to_excel(writer, sheet_name='Risk Data', index=False)
                    summary_stats = pd.DataFrame({
                        'Metric':['Total','Critical','High','Avg Score','Residual>0'],
                        'Value':[total_risks, critical_risks, high_risks, avg_score, residual_risks]
                    })
                    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                    
                    pd.crosstab(processed_df['Type of Risk'], processed_df['Risk_Level'], margins=True)\
                      .to_excel(writer, sheet_name='Risk Distribution')
                    
                    crit_df.to_excel(writer, sheet_name='Critical_High', index=False)

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

