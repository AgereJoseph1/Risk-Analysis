import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
import re
from datetime import datetime

# Set page config
st.set_page_config(layout="wide", page_title="Enterprise Risk Management Dashboard")

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .risk-header {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 2rem;
            color: #1f2937;
        }
    </style>
""", unsafe_allow_html=True)

def extract_numeric_value(value):
    """Safely extract numeric value from text"""
    try:
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        return float(matches[0]) if matches else 0
    except:
        return 0

def process_data(df):
    """Process the data with enhanced risk calculations"""
    try:
        processed_df = df.copy()
        
        # Extract numeric values
        processed_df['Numeric_Value'] = processed_df['Upper Limit'].apply(extract_numeric_value)
        
        # Calculate risk levels
        def calculate_risk(value):
            try:
                num_val = float(value)
                if num_val <= 7:
                    return 1, '#006400', 'Low'
                elif num_val <= 13:
                    return 2, '#90EE90', 'Medium'
                elif num_val <= 19:
                    return 3, '#FFD700', 'High'
                else:
                    return 4, '#FF0000', 'Critical'
            except:
                return 0, '#CCCCCC', 'Invalid'
        
        # Apply risk calculations
        risk_data = processed_df['Numeric_Value'].apply(calculate_risk)
        processed_df['Risk_Score'] = [x[0] for x in risk_data]
        processed_df['Risk_Color'] = [x[1] for x in risk_data]
        processed_df['Risk_Level'] = [x[2] for x in risk_data]
        
        return processed_df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_summary_metrics(df):
    """Create summary metrics for dashboard"""
    total_risks = int(len(df))
    critical_risks = int(len(df[df['Risk_Level'] == 'Critical']))
    high_risks = int(len(df[df['Risk_Level'] == 'High']))
    avg_score = float(df['Risk_Score'].mean())
    risk_types = int(len(df['Type of Risk'].unique()))
    
    return {
        'total': total_risks,
        'critical': critical_risks,
        'high': high_risks,
        'avg_score': avg_score,
        'types': risk_types
    }

def create_risk_overview(df, chart_id="overview"):
    """Create risk overview visualization"""
    # Risk Level Distribution
    risk_dist = df['Risk_Level'].value_counts().reset_index()
    risk_dist.columns = ['Risk Level', 'Count']
    
    fig = go.Figure()
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=risk_dist['Risk Level'],
            values=risk_dist['Count'],
            hole=0.4,
            name=f"Risk Distribution {chart_id}",
            marker=dict(colors=['#006400', '#90EE90', '#FFD700', '#FF0000'])
        )
    )
    
    fig.update_layout(
        title="Risk Level Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def create_risk_matrix(df, chart_id="matrix"):
    """Create risk matrix visualization"""
    risk_matrix = pd.crosstab(df['Type of Risk'], df['Risk_Level'])
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix.values.astype(float),
        x=risk_matrix.columns,
        y=risk_matrix.index,
        colorscale=[
            [0, '#006400'],
            [0.33, '#90EE90'],
            [0.66, '#FFD700'],
            [1, '#FF0000']
        ],
        name=f"Risk Matrix {chart_id}",
        hoverongaps=False,
        hovertemplate='Type: %{y}<br>Level: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Risk Matrix",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_trend_analysis(df, chart_id="trends"):
    """Create trend analysis visualization"""
    trends = df.groupby('Type of Risk')['Risk_Score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=trends.index,
            y=trends['mean'],
            text=[f"{x:.1f}" for x in trends['mean']],
            textposition='auto',
            name=f"Risk Score Trends {chart_id}",
            marker_color='#1f77b4'
        )
    ])
    
    fig.update_layout(
        title="Average Risk Score by Type",
        xaxis_title="Risk Type",
        yaxis_title="Average Risk Score",
        height=400,
        showlegend=False
    )
    
    return fig

# Main dashboard
st.markdown('<p class="risk-header">Enterprise Risk Management Dashboard</p>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload Risk Assessment Data", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        # Read and validate data
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        
        required_columns = ['Risk#', 'Type of Risk', 'Risk ID', 'Description', 'Lower Limit', 'Upper Limit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()
        
        # Process data
        with st.spinner("Analyzing risk data..."):
            processed_df = process_data(df)
            
            if processed_df is not None:
                # Calculate metrics
                metrics = create_summary_metrics(processed_df)
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Risks", metrics['total'])
                col2.metric("Critical Risks", metrics['critical'], f"{(metrics['critical']/metrics['total']*100):.1f}%")
                col3.metric("High Risks", metrics['high'], f"{(metrics['high']/metrics['total']*100):.1f}%")
                col4.metric("Avg Risk Score", f"{metrics['avg_score']:.2f}")
                col5.metric("Risk Types", metrics['types'])
                
                # Create tabs for analysis
                tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Details"])
                
                with tab1:
                    # Risk Overview
                    st.plotly_chart(create_risk_overview(processed_df, "tab1"), use_container_width=True)
                    st.plotly_chart(create_trend_analysis(processed_df, "tab1"), use_container_width=True)
                
                with tab2:
                    # Risk Matrix and Analysis
                    st.plotly_chart(create_risk_matrix(processed_df, "tab2"), use_container_width=True)
                    
                    # High Risk Items
                    st.subheader("Critical and High Risk Items")
                    high_risks = processed_df[processed_df['Risk_Level'].isin(['Critical', 'High'])]
                    high_risks = high_risks.sort_values('Risk_Score', ascending=False)
                    st.dataframe(
                        high_risks[['Risk ID', 'Type of Risk', 'Description', 'Risk_Level', 'Risk_Score']],
                        use_container_width=True
                    )
                
                with tab3:
                    # Detailed Data View
                    st.subheader("Risk Analysis Details")
                    
                    # Risk Distribution
                    risk_dist = pd.crosstab(processed_df['Type of Risk'], processed_df['Risk_Level'], margins=True)
                    st.dataframe(risk_dist, use_container_width=True)
                    
                    # Full Data Table
                    st.subheader("Complete Risk Register")
                    st.dataframe(
                        processed_df[['Risk ID', 'Type of Risk', 'Description', 'Risk_Level', 'Risk_Score']],
                        use_container_width=True
                    )
                
                # Export options
                st.subheader("Export Options")
                if st.button("Generate Report"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        processed_df.to_excel(writer, sheet_name='Risk Data', index=False)
                        
                        # Summary sheet
                        pd.DataFrame({
                            'Metric': ['Total Risks', 'Critical Risks', 'High Risks', 'Average Score'],
                            'Value': [metrics['total'], metrics['critical'], metrics['high'], metrics['avg_score']]
                        }).to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Risk distribution
                        risk_dist.to_excel(writer, sheet_name='Risk Distribution')
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Download Report",
                        data=buffer,
                        file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your file format and try again.")
else:
    st.info("Please upload your risk assessment file to begin analysis")
    st.markdown("""
    ### Required Format:
    Your Excel file should contain these columns:
    - Risk# (ID number)
    - Type of Risk (Category)
    - Risk ID (Reference code)
    - Description (Risk details)
    - Lower Limit (Threshold)
    - Upper Limit (Maximum value)
    """)