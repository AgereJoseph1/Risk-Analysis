import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import re

# Set page config
st.set_page_config(layout="wide", page_title="Risk Management Dashboard")

# Helper function to extract numbers from text
def extract_numeric_value(value):
    """Safely extract numeric values from text"""
    try:
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return float(value)
        # Extract numbers from text (e.g., "2 days", "3 working days")
        numbers = re.findall(r'\d+(?:\.\d+)?', str(value))
        return float(numbers[0]) if numbers else 0
    except Exception:
        return 0

# Function to determine risk level
def get_risk_level(value):
    """Calculate risk level and color based on value"""
    try:
        value = float(value)
        if value <= 7:
            return {'score': 1, 'color': '#006400', 'level': 'Low'}
        elif value <= 13:
            return {'score': 2, 'color': '#90EE90', 'level': 'Medium'}
        elif value <= 19:
            return {'score': 3, 'color': '#FFD700', 'level': 'High'}
        else:
            return {'score': 4, 'color': '#FF0000', 'level': 'Critical'}
    except:
        return {'score': 0, 'color': '#CCCCCC', 'level': 'Invalid'}

def create_risk_matrix():
    """Create interactive risk matrix visualization"""
    fig = go.Figure()
    
    # Add cells
    for i in range(5):
        for j in range(5):
            value = (5-i) * (j+1)
            risk_info = get_risk_level(value)
            
            # Add cell
            fig.add_shape(
                type="rect",
                x0=j, y0=i,
                x1=j+1, y1=i+1,
                fillcolor=risk_info['color'],
                line=dict(color="white", width=2),
            )
            
            # Add value label
            fig.add_annotation(
                x=j+0.5, y=i+0.5,
                text=str(value),
                showarrow=False,
                font=dict(color='white', size=14)
            )

    # Update layout
    fig.update_layout(
        title="Risk Matrix (Impact vs Likelihood)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=False,
        xaxis=dict(
            range=[-0.1, 5.1],
            title="Impact",
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            range=[-0.1, 5.1],
            title="Likelihood",
            showticklabels=False,
            showgrid=False
        ),
    )
    
    return fig

def process_data(df):
    """Process the Excel data"""
    try:
        # Extract numeric values
        df['Numeric_Value'] = df['Upper Limit'].apply(extract_numeric_value)
        
        # Calculate risk levels
        risk_data = df['Numeric_Value'].apply(get_risk_level)
        
        # Add risk columns
        df['Risk_Score'] = risk_data.apply(lambda x: x['score'])
        df['Risk_Color'] = risk_data.apply(lambda x: x['color'])
        df['Risk_Level'] = risk_data.apply(lambda x: x['level'])
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def create_visualizations(df):
    """Create dashboard visualizations"""
    try:
        # Risk Level Distribution
        risk_dist = df['Risk_Level'].value_counts().reset_index()
        fig_dist = px.pie(
            risk_dist,
            values='count',
            names='Risk_Level',
            title="Risk Level Distribution",
            color='Risk_Level',
            color_discrete_map={
                'Low': '#006400',
                'Medium': '#90EE90',
                'High': '#FFD700',
                'Critical': '#FF0000'
            }
        )

        # Risk Type Analysis
        risk_type = px.bar(
            df['Type of Risk'].value_counts().reset_index(),
            x='Type of Risk',
            y='count',
            title="Risks by Type"
        )
        
        return fig_dist, risk_type
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        return None, None

# Main dashboard
st.title("Risk Assessment Dashboard")
uploaded_file = st.file_uploader("Upload your risk assessment data (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Read the file based on its type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Validate columns
        required_cols = ['Risk#', 'Type of Risk', 'Risk ID', 'Description', 'Lower Limit', 'Upper Limit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        # Process data
        df = process_data(df)
        
        if df is not None:
            # Success message
            st.success("File processed successfully!")
            
            # Summary metrics
            cols = st.columns(4)
            cols[0].metric("Total Risks", len(df))
            cols[1].metric("Critical Risks", len(df[df['Risk_Level'] == 'Critical']))
            cols[2].metric("High Risks", len(df[df['Risk_Level'] == 'High']))
            cols[3].metric("Low Risks", len(df[df['Risk_Level'] == 'Low']))
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Data"])
            
            with tab1:
                fig_dist, fig_type = create_visualizations(df)
                if fig_dist and fig_type:
                    col1, col2 = st.columns(2)
                    col1.plotly_chart(fig_dist, use_container_width=True)
                    col2.plotly_chart(fig_type, use_container_width=True)
                
            with tab2:
                st.plotly_chart(create_risk_matrix(), use_container_width=True)
                
                # Risk breakdown
                st.subheader("Risk Analysis by Type")
                risk_analysis = pd.crosstab(
                    df['Type of Risk'],
                    df['Risk_Level']
                )
                st.dataframe(risk_analysis, use_container_width=True)
                
            with tab3:
                st.dataframe(
                    df.style.background_gradient(
                        subset=['Risk_Score'],
                        cmap='RdYlGn_r'
                    ),
                    use_container_width=True
                )
                
            # Export options
            if st.button("Export Report"):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Risk Data', index=False)
                    risk_analysis.to_excel(writer, sheet_name='Risk Analysis')
                
                buffer.seek(0)
                st.download_button(
                    label="Download Excel Report",
                    data=buffer,
                    file_name=f"risk_assessment_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Upload an Excel file to begin analysis")
    st.markdown("""
    ### Required Columns:
    - Risk# (ID number)
    - Type of Risk (Category)
    - Risk ID (Reference code)
    - Description (Risk description)
    - Lower Limit (Threshold)
    - Upper Limit (Maximum value)
    """)