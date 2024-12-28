import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import re
from datetime import datetime

# Configuration
RISK_COLORS = {
    1: ('#006400', 'Low'),      # Deep green
    2: ('#90EE90', 'Minor'),    # Light green
    3: ('#FFD700', 'Medium'),   # Yellow
    4: ('#FF0000', 'High'),     # Red
    5: ('#722F37', 'Critical')  # Wine
}

THRESHOLD_RULES = {
    'time': {
        2: 1,    # ≤ 2 days: Low risk
        3: 2,    # ≤ 3 days: Minor risk
        4: 3,    # ≤ 4 days: Medium risk
        5: 4,    # ≤ 5 days: High risk
        float('inf'): 5  # > 5 days: Critical risk
    },
    'percentage': {
        15: 1,   # ≤ 15%: Low risk
        25: 2,   # ≤ 25%: Minor risk
        35: 3,   # ≤ 35%: Medium risk
        50: 4,   # ≤ 50%: High risk
        float('inf'): 5  # > 50%: Critical risk
    },
    'target': {
        10: 1,   # ≤ 10% off target: Low risk
        20: 2,   # ≤ 20% off target: Minor risk
        30: 3,   # ≤ 30% off target: Medium risk
        40: 4,   # ≤ 40% off target: High risk
        float('inf'): 5  # > 40% off target: Critical risk
    }
}

# Page config
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
    </style>
""", unsafe_allow_html=True)

def categorize_threshold_type(text):
    """Determine the type of threshold from the text"""
    text = str(text).lower()
    if any(word in text for word in ['day', 'working day', 'week']):
        return 'time'
    elif any(word in text for word in ['%', 'percent', 'percentage', 'target']):
        return 'percentage'
    else:
        return 'time'  # Default to time-based

def extract_numeric_value(value, threshold_type):
    """Extract numeric values based on threshold type"""
    try:
        if pd.isna(value):
            return 0
        
        text = str(value).lower()
        
        if threshold_type == 'time':
            matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:day|week)', text)
            return float(matches[0]) if matches else 0
        elif threshold_type == 'percentage':
            matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
            return float(matches[0]) if matches else 0
        
        # Default number extraction
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        return float(matches[0]) if matches else 0
    except:
        return 0

def calculate_risk_score(value, threshold_type):
    """Calculate risk score based on threshold type and rules"""
    try:
        rules = THRESHOLD_RULES.get(threshold_type, THRESHOLD_RULES['time'])
        for threshold, score in rules.items():
            if value <= threshold:
                return score
        return max(rules.values())
    except:
        return 1

def process_data(df):
    """Process data with risk calculations"""
    try:
        processed_df = df.copy()
        
        # Determine threshold type and extract values
        processed_df['Threshold_Type'] = processed_df['Upper Limit'].apply(categorize_threshold_type)
        processed_df['Numeric_Value'] = processed_df.apply(
            lambda row: extract_numeric_value(row['Upper Limit'], row['Threshold_Type']),
            axis=1
        )
        
        # Calculate risk scores
        processed_df['Risk_Score'] = processed_df.apply(
            lambda row: calculate_risk_score(row['Numeric_Value'], row['Threshold_Type']),
            axis=1
        )
        
        # Assign colors and levels
        processed_df['Risk_Color'] = processed_df['Risk_Score'].map(lambda x: RISK_COLORS[x][0])
        processed_df['Risk_Level'] = processed_df['Risk_Score'].map(lambda x: RISK_COLORS[x][1])
        
        return processed_df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_visualizations(df):
    """Create all visualizations"""
    # Risk Level Distribution
    risk_dist = df['Risk_Level'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_dist.index,
        values=risk_dist.values,
        hole=0.4,
        marker=dict(colors=[RISK_COLORS[i][0] for i in range(1, 6)]),
        textinfo='label+percent'
    )])
    fig_pie.update_layout(title="Risk Level Distribution", height=400)
    
    # Risk Matrix
    risk_matrix = pd.crosstab(df['Type of Risk'], df['Risk_Level'])
    fig_matrix = go.Figure(data=go.Heatmap(
        z=risk_matrix.values,
        x=risk_matrix.columns,
        y=risk_matrix.index,
        colorscale=[[0, '#006400'], [0.25, '#90EE90'], [0.5, '#FFD700'], 
                   [0.75, '#FF0000'], [1, '#722F37']],
        showscale=True,
        text=risk_matrix.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 14}
    ))
    fig_matrix.update_layout(
        title="Risk Matrix: Type vs Level",
        height=500,
        xaxis_title="Risk Level",
        yaxis_title="Risk Type"
    )
    
    # Risk Score Trends
    avg_scores = df.groupby('Type of Risk')['Risk_Score'].mean().sort_values(ascending=False)
    fig_trends = go.Figure(data=[go.Bar(
        x=avg_scores.index,
        y=avg_scores.values,
        marker_color='#1E3F66',
        text=[f"{x:.2f}" for x in avg_scores.values],
        textposition='auto'
    )])
    fig_trends.update_layout(
        title="Average Risk Score by Category",
        height=400,
        xaxis_title="Risk Type",
        yaxis_title="Average Risk Score"
    )
    
    return fig_pie, fig_matrix, fig_trends

# Main dashboard
st.markdown("""
    <div class='dashboard-header'>
        <h1>🎯 Risk Intelligence Dashboard</h1>
        <div class='dashboard-subtitle'>
            Comprehensive Risk Assessment & Analytics Platform
        </div>
    </div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Upload Risk Assessment Data",
    type=['xlsx', 'csv'],
    help="Upload your risk assessment file"
)

if uploaded_file is not None:
    try:
        # Read data
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        
        # Validate columns
        required_columns = ['Risk#', 'Type of Risk', 'Risk ID', 'Description', 'Lower Limit', 'Upper Limit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()
        
        # Process data
        processed_df = process_data(df)
        
        if processed_df is not None:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_risks = len(processed_df)
            critical_risks = len(processed_df[processed_df['Risk_Level'] == 'Critical'])
            high_risks = len(processed_df[processed_df['Risk_Level'] == 'High'])
            avg_score = processed_df['Risk_Score'].mean()
            
            with col1:
                st.metric("Total Risks", total_risks)
            with col2:
                st.metric("Critical Risks", critical_risks, f"{(critical_risks/total_risks*100):.1f}%")
            with col3:
                st.metric("High Risks", high_risks, f"{(high_risks/total_risks*100):.1f}%")
            with col4:
                st.metric("Avg Risk Score", f"{avg_score:.2f}")
            
            # Create visualizations
            fig_pie, fig_matrix, fig_trends = create_visualizations(processed_df)
            
            # Display visualizations in tabs
            tab1, tab2, tab3 = st.tabs(["Risk Distribution", "Risk Matrix", "Trend Analysis"])
            
            with tab1:
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # High Risk Items
                st.subheader("Critical and High Risk Items")
                high_risks_df = processed_df[processed_df['Risk_Level'].isin(['Critical', 'High'])]
                if not high_risks_df.empty:
                    st.dataframe(
                        high_risks_df[['Risk ID', 'Type of Risk', 'Description', 'Risk_Level', 'Risk_Score']],
                        use_container_width=True
                    )
                else:
                    st.info("No high-risk items identified")
            
            with tab2:
                st.plotly_chart(fig_matrix, use_container_width=True)
            
            with tab3:
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Export options
            st.subheader("Export Data")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer) as writer:
                processed_df.to_excel(writer, sheet_name='Risk Assessment', index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="📈 Download Excel Report",
                data=excel_buffer,
                file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your file format and try again.")

else:
    st.info("Upload your risk assessment file to begin analysis")
    st.markdown("""
    ### Required Format:
    Your file should contain these columns:
    - Risk# (ID number)
    - Type of Risk (Category)
    - Risk ID (Reference code)
    - Description (Risk details)
    - Lower Limit (Threshold)
    - Upper Limit (Maximum value)
    """)