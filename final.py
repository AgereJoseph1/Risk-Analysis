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
            
          /* Hide Streamlit menu button */
        #MainMenu {visibility: hidden;}
        
        /* Hide GitHub link */
        footer {visibility: hidden;}
        
        /* Hide documentation link */
        header {visibility: hidden;}
        
        /* Optional: Adjust the main content padding if needed */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
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
    """Process data with enhanced risk calculations"""
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

def create_risk_matrix(df):
    """Create risk matrix visualization"""
    risk_matrix = pd.crosstab(df['Type of Risk'], df['Risk_Level'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=risk_matrix.values.astype(float),
        x=risk_matrix.columns,
        y=risk_matrix.index,
        colorscale=[
            [0, '#006400'],
            [0.33, '#90EE90'],
            [0.66, '#FFD700'],
            [1, '#FF0000']
        ],
        showscale=True,
        text=risk_matrix.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title="Risk Matrix: Type vs Level",
        height=600,
        showlegend=False,
        xaxis_title="Risk Level",
        yaxis_title="Risk Type"
    )
    
    return fig

def create_risk_distribution(df):
    """Create risk distribution visualizations"""
    # Risk Level Distribution
    risk_dist = df['Risk_Level'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=risk_dist.index,
        values=risk_dist.values,
        hole=0.4,
        marker=dict(colors=['#006400', '#90EE90', '#FFD700', '#FF0000']),
        textinfo='label+percent'
    )])
    
    fig_pie.update_layout(
        title="Risk Level Distribution",
        height=400
    )
    
    # Risk Type Distribution
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
    """Create trend analysis visualization"""
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

def create_report_visualizations(df):
    """Create visualizations for the PDF report"""
    charts = {}
    
    # Risk Level Distribution
    plt.figure(figsize=(8, 6))
    risk_dist = df['Risk_Level'].value_counts()
    plt.pie(risk_dist, labels=risk_dist.index, autopct='%1.1f%%',
            colors=['#FF0000', '#FFD700', '#90EE90', '#006400'])
    plt.title('Risk Level Distribution')
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    charts['risk_dist'] = img_buffer
    plt.close()
    
    # Risk by Type
    plt.figure(figsize=(10, 6))
    risk_by_type = df.groupby('Type of Risk')['Risk_Level'].value_counts().unstack()
    risk_by_type.plot(kind='bar', stacked=True)
    plt.title('Risk Distribution by Category')
    plt.xlabel('Risk Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    charts['risk_by_type'] = img_buffer
    plt.close()
    
    # Risk Matrix
    plt.figure(figsize=(10, 6))
    risk_matrix = pd.crosstab(df['Type of Risk'], df['Risk_Level'])
    sns.heatmap(risk_matrix, annot=True, fmt='d', cmap='RdYlGn_r')
    plt.title('Risk Matrix')
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    charts['risk_matrix'] = img_buffer
    plt.close()
    
    return charts

def get_report_styles():
    """Get styles for the PDF report"""
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        alignment=1,
        textColor=colors.HexColor('#1E3F66')
    )
    
    
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        alignment=1,
        textColor=colors.HexColor('#2E5090')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#2E5090')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#1E3F66')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=6,
        spaceAfter=6
    )

    date_style = ParagraphStyle(
        'CustomDate',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=15,
        alignment=1,
        textColor=colors.HexColor('#666666')
    )
    
    # Add confidential style
    confidential_style = ParagraphStyle(
        'Confidential',
        parent=styles['Normal'],
        fontSize=14,
        alignment=1,
        textColor=colors.red,
        spaceBefore=30
    )
    
    return {
        'title': title_style,
        'date':date_style,
        'subtitle':  subtitle_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'body': body_style,
        'base': styles,
        'confidential': confidential_style
    }

def generate_comprehensive_report(df, timestamp):
    """Generate an enhanced professional risk analysis report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    story = []
    styles = get_report_styles()
    
    # 1. Cover Page with Organization Logo (if available)
    story.append(Paragraph("Enterprise Risk Assessment", styles['title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Comprehensive Analysis Report", styles['subtitle']))
    story.append(Spacer(1, 40))
    
    # Add organization logo if available
    # story.append(Image("logo.png", width=200, height=100))
    
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"Report Date: {timestamp}", styles['date']))
    story.append(Paragraph("CONFIDENTIAL", styles['confidential']))
    story.append(PageBreak())

    # 2. Executive Summary
    story.append(Paragraph("Executive Summary", styles['heading']))
    story.append(Spacer(1, 20))
    
    total_risks = len(df)
    critical_risks = len(df[df['Risk_Level'] == 'Critical'])
    high_risks = len(df[df['Risk_Level'] == 'High'])
    avg_score = df['Risk_Score'].mean()
    
    summary_text = f"""
    This comprehensive risk assessment analyzes {total_risks} identified risks across the organization.
    The assessment reveals significant findings that require immediate attention and strategic planning.
    
    Key Metrics:
    • Total Identified Risks: {total_risks}
    • Critical Risk Items: {critical_risks} ({(critical_risks/total_risks*100):.1f}%)
    • High Risk Items: {high_risks} ({(high_risks/total_risks*100):.1f}%)
    • Average Risk Score: {avg_score:.2f}
    • Risk Categories: {len(df['Type of Risk'].unique())}
    
    Key Areas of Concern:
    • {df[df['Risk_Level'] == 'Critical']['Type of Risk'].value_counts().index[0]} shows the highest concentration of critical risks
    • {high_risks} high-priority items require immediate attention
    • Average risk score indicates {get_risk_severity_text(avg_score)}
    """
    story.append(Paragraph(summary_text, styles['body']))
    story.append(PageBreak())

    # 3. Risk Distribution Analysis
    story.append(Paragraph("Risk Distribution Analysis", styles['heading']))
    charts = create_report_visualizations(df)
    
    story.append(Paragraph("Risk Level Distribution", styles['subheading']))
    story.append(Paragraph("""
    The following chart illustrates the distribution of risks across different severity levels, 
    providing insights into the overall risk landscape of the organization.
    """, styles['body']))
    story.append(Image(charts['risk_dist'], width=400, height=300))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("Risk Category Analysis", styles['subheading']))
    story.append(Image(charts['risk_by_type'], width=450, height=300))
    story.append(PageBreak())

    # 4. Risk Matrix Analysis
    story.append(Paragraph("Risk Matrix Analysis", styles['heading']))
    story.append(Paragraph("""
    The risk matrix below provides a visual representation of risk distribution across different 
    categories and severity levels, helping identify areas requiring immediate attention.
    """, styles['body']))
    story.append(Image(charts['risk_matrix'], width=450, height=300))
    story.append(PageBreak())

    # 5. Critical and High Risk Analysis
    story.append(Paragraph("Critical and High Risk Analysis", styles['heading']))
    story.append(Paragraph("""
    The following section details the high-priority risks that require immediate attention 
    and mitigation strategies.
    """, styles['body']))
    
    high_priority = df[df['Risk_Level'].isin(['Critical', 'High'])].sort_values('Risk_Score', ascending=False)
    
    if not high_priority.empty:
        data = [["Risk ID", "Type", "Description", "Level", "Score", "Impact"]]
        data.extend([
            [row['Risk ID'], 
             row['Type of Risk'], 
             Paragraph(row['Description'], styles['body']),
             row['Risk_Level'],
             f"{row['Risk_Score']:.1f}",
             get_impact_description(row['Risk_Score'])]
            for _, row in high_priority.iterrows()
        ])
        
        table = Table(data, colWidths=[60, 80, 180, 60, 40, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3F66')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
        ]))
        story.append(table)
    story.append(PageBreak())

    # 6. Recommendations and Action Items
    story.append(Paragraph("Strategic Recommendations", styles['heading']))
    
    recommendations = generate_recommendations(df)
    for i, (title, rec) in enumerate(recommendations.items(), 1):
        story.append(Paragraph(f"{i}. {title}", styles['subheading']))
        story.append(Paragraph(rec, styles['body']))
        story.append(Spacer(1, 10))

    # 7. Appendix
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Risk Assessment Methodology", styles['heading']))
    story.append(Paragraph("""
    Risk Assessment Criteria:
    • Risk scores range from 1 (lowest) to 5 (highest)
    • Risk levels are categorized as: Low, Medium, High, and Critical
    • Assessment considers both likelihood and impact of risks
    • Calculations factor in historical data and current controls
    """, styles['body']))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_risk_severity_text(score):
    """Generate severity description based on risk score"""
    if score <= 1.5:
        return "a relatively low overall risk exposure"
    elif score <= 2.5:
        return "a moderate level of risk exposure requiring attention"
    elif score <= 3.5:
        return "a significant risk exposure requiring immediate action"
    else:
        return "a critical level of risk exposure requiring urgent intervention"

def get_impact_description(score):
    """Generate impact description based on risk score"""
    if score <= 2:
        return "Limited Impact"
    elif score <= 3:
        return "Moderate Impact"
    elif score <= 4:
        return "Major Impact"
    else:
        return "Severe Impact"

def generate_recommendations(df):
    """Generate detailed recommendations based on risk analysis"""
    recommendations = {}
    
    # Critical Risk Recommendations
    critical_risks = df[df['Risk_Level'] == 'Critical']
    if not critical_risks.empty:
        critical_types = critical_risks['Type of Risk'].value_counts()
        recommendations["Immediate Action Required"] = f"""
        Address {len(critical_risks)} critical risks with priority focus on {critical_types.index[0]} category.
        Implement enhanced monitoring and control measures for these high-priority items.
        Schedule weekly progress reviews for critical risk mitigation efforts.
        """

    # Risk Score Management
    avg_score = df['Risk_Score'].mean()
    if avg_score > 2.5:
        recommendations["Risk Exposure Management"] = f"""
        Overall risk score ({avg_score:.2f}) indicates elevated exposure.
        Strengthen control measures across all risk categories.
        Develop comprehensive risk mitigation strategies for high-scoring areas.
        """

    # Category-specific Recommendations
    risk_concentration = df['Type of Risk'].value_counts()
    if risk_concentration.iloc[0] / len(df) > 0.3:
        recommendations["Risk Distribution Strategy"] = f"""
        High concentration in {risk_concentration.index[0]} category ({(risk_concentration.iloc[0]/len(df)*100):.1f}% of risks).
        Implement targeted controls for {risk_concentration.index[0]} risks.
        Consider resource reallocation to address concentration risk.
        """

    return recommendations
# Main dashboard
st.markdown("""
    <div class='dashboard-header'>
        <h1>🎯 Risk Intelligence Dashboard </h1>
        <div class='dashboard-subtitle'>
            Comprehensive Risk Assessment & Analytics Platform
        </div>
    </div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Upload Risk Assessment Data",
    type=['xlsx', 'csv'],
    help="Upload your risk assessment Excel file"
)

if uploaded_file is not None:
    try:
        # Read and process data
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        
        # Validate columns
        required_columns = ['Risk#', 'Type of Risk', 'Risk ID', 'Description', 'Lower Limit', 'Upper Limit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.stop()
        
        processed_df = process_data(df)
        
        if processed_df is not None:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_risks = len(processed_df)
            critical_risks = len(processed_df[processed_df['Risk_Level'] == 'Critical'])
            high_risks = len(processed_df[processed_df['Risk_Level'] == 'High'])
            avg_score = processed_df['Risk_Score'].mean()
            
            col1.metric("Total Risks", total_risks)
            col2.metric("Critical Risks", critical_risks, f"{(critical_risks/total_risks*100):.1f}%")
            col3.metric("High Risks", high_risks, f"{(high_risks/total_risks*100):.1f}%")
            col4.metric("Avg Risk Score", f"{avg_score:.2f}")
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Data"])
            
            with tab1:
                # Risk Distribution
                col1, col2 = st.columns(2)
                fig_pie, fig_bar = create_risk_distribution(processed_df)
                col1.plotly_chart(fig_pie, use_container_width=True)
                col2.plotly_chart(fig_bar, use_container_width=True)
                
                # Trend Analysis
                st.plotly_chart(create_trend_analysis(processed_df), use_container_width=True)
            
            with tab2:
                # Risk Matrix
                st.plotly_chart(create_risk_matrix(processed_df), use_container_width=True)
                
                # High Risk Items
                st.subheader("Critical and High Risk Items")
                high_risks_df = processed_df[processed_df['Risk_Level'].isin(['Critical', 'High'])]
                st.dataframe(
                    high_risks_df[['Risk ID', 'Type of Risk', 'Description', 'Risk_Level', 'Risk_Score']],
                    use_container_width=True
                )
            
            with tab3:
                st.dataframe(processed_df, use_container_width=True)
            
           
            # Export options
            st.subheader("Report Generation")
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
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
                            mime="application/pdf",
                            help="Download detailed risk analysis report with insights and recommendations"
                        )
            
            with report_col2:
                st.info("Download Raw Data Report")
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer) as writer:
                    # Main risk data
                    processed_df.to_excel(writer, sheet_name='Risk Data', index=False)
                    
                    # Summary statistics
                    summary_stats = pd.DataFrame({
                        'Metric': ['Total Risks', 'Critical Risks', 'High Risks', 'Average Risk Score'],
                        'Value': [
                            len(processed_df),
                            len(processed_df[processed_df['Risk_Level'] == 'Critical']),
                            len(processed_df[processed_df['Risk_Level'] == 'High']),
                            processed_df['Risk_Score'].mean()
                        ]
                    })
                    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Risk distribution
                    pd.crosstab(processed_df['Type of Risk'], 
                              processed_df['Risk_Level'], 
                              margins=True).to_excel(writer, sheet_name='Risk Distribution')
                    
                    # High risk details
                    high_risks_df = processed_df[processed_df['Risk_Level'].isin(['Critical', 'High'])]
                    high_risks_df.to_excel(writer, sheet_name='Critical & High Risks', index=False)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="📈 Download Excel Data",
                    data=excel_buffer,
                    file_name=f"risk_assessment_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download complete risk assessment data and statistics"
                )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your file format and try again.")
        st.write("Error details:", e)

else:
    st.info("Upload your risk assessment file to begin analysis")
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