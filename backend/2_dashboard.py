"""
Modern Streamlit Dashboard for Blockchain Fraud Detection MVP
Professional UI with clean design and improved UX
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os


def load_flagged_transactions(filepath='output/flagged_transactions.csv'):
    """Load flagged transactions from CSV"""
    try:
        if not os.path.exists(filepath):
            return None
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_risk_gauge(score):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 40, 'color': '#1e293b'}},
        gauge={
            'axis': {'range': [None, 150], 'tickwidth': 2, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 150], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'}
    )
    
    return fig


def create_status_distribution(df):
    """Create pie chart for status distribution"""
    status_counts = df['final_status'].value_counts()
    
    colors = {
        'DENY': '#ef4444',
        'FLAG_FOR_REVIEW': '#f59e0b',
        'APPROVE': '#10b981'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.5,
        marker=dict(
            colors=[colors.get(status, '#8b5cf6') for status in status_counts.index],
            line=dict(color='white', width=3)
        ),
        textposition='outside',
        textinfo='label+percent',
        textfont=dict(size=13, color='#1e293b')
    )])
    
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'}
    )
    
    return fig


def create_risk_timeline(df):
    """Create timeline of flagged transactions"""
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp')
        
        fig = go.Figure()
        
        color_map = {
            'DENY': '#ef4444',
            'FLAG_FOR_REVIEW': '#f59e0b',
            'APPROVE': '#10b981'
        }
        
        for status in df_sorted['final_status'].unique():
            df_status = df_sorted[df_sorted['final_status'] == status]
            
            fig.add_trace(go.Scatter(
                x=df_status['timestamp'],
                y=df_status['final_score'],
                mode='markers',
                name=status.replace('_', ' ').title(),
                marker=dict(
                    size=14,
                    color=color_map.get(status, '#8b5cf6'),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=df_status['tx_hash'].apply(lambda x: f"{x[:16]}..."),
                hovertemplate='<b>%{text}</b><br>Score: %{y:.1f}<br>Time: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text="Risk Score Timeline", font=dict(size=16, color='#1e293b')),
            xaxis_title="Time",
            yaxis_title="Risk Score",
            height=320,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='white',
            xaxis=dict(gridcolor='#e2e8f0', showline=True, linecolor='#e2e8f0'),
            yaxis=dict(gridcolor='#e2e8f0', showline=True, linecolor='#e2e8f0'),
            font={'family': 'Inter, sans-serif', 'color': '#64748b'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    return None


def create_value_distribution(df):
    """Create histogram of transaction values"""
    fig = px.histogram(
        df,
        x='value_eth',
        nbins=25,
        labels={'value_eth': 'Transaction Value (ETH)', 'count': 'Frequency'},
        color_discrete_sequence=['#3b82f6']
    )
    
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        xaxis=dict(gridcolor='#e2e8f0', showline=True, linecolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0', showline=True, linecolor='#e2e8f0'),
        font={'family': 'Inter, sans-serif', 'color': '#64748b'}
    )
    
    return fig


def main():
    """Main dashboard function"""
    # Page configuration
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Modern CSS styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #1e293b;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.875rem;
        }
        
        h1 {
            color: #0f172a;
            font-weight: 800;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: #1e293b;
            font-weight: 700;
            font-size: 1.5rem !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: #334155;
            font-weight: 600;
            font-size: 1.125rem !important;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-radius: 10px;
            padding: 0.625rem 1.5rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
            transform: translateY(-2px);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            padding: 2rem 1rem;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stButton>button {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stSidebar"] .stButton>button:hover {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        div[data-testid="stDataFrame"] {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        .status-badge {
            display: inline-block;
            padding: 0.375rem 0.875rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-deny {
            background-color: #fee2e2;
            color: #dc2626;
        }
        
        .status-review {
            background-color: #fef3c7;
            color: #d97706;
        }
        
        .status-approve {
            background-color: #dcfce7;
            color: #16a34a;
        }
        
        .info-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
        }
        
        .warning-card {
            background: #fef3c7;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid #f59e0b;
            margin: 1rem 0;
        }
        
        .danger-card {
            background: #fee2e2;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid #ef4444;
            margin: 1rem 0;
        }
        
        hr {
            margin: 2rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        }
        
        .stSelectbox label, .stMultiSelect label, .stSlider label {
            font-weight: 600 !important;
            color: #334155 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 🛡️ Fraud Detection")
        st.markdown("### AI-Powered Security")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # System status
        st.markdown("### 📊 System Status")
        st.success("● **System Online**")
        st.info("● **Active Agents:** 3")
        st.info("● **Mode:** Real-time")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### 🔍 Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        Real-time blockchain transaction monitoring powered by intelligent AI agents.
        
        **Agent Architecture:**
        - 🔍 **Agent 1:** Threat Intel
        - 🧠 **Agent 2:** Behavioral Analysis  
        - ⚖️ **Agent 3:** Decision Engine
        """)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p style='text-align: center; font-size: 0.75rem; opacity: 0.6;'>v2.0.0 | Built with ❤️</p>", unsafe_allow_html=True)
    
    # Header
    st.markdown("# 🛡️ Blockchain Fraud Detection")
    st.markdown("<p style='font-size: 1.125rem; color: #64748b; margin-bottom: 2rem;'>Real-time transaction monitoring and risk assessment powered by AI</p>", unsafe_allow_html=True)
    
    # Load data
    df = load_flagged_transactions()
    
    if df is None:
        st.markdown("""
        <div class='danger-card'>
            <h3 style='margin-top: 0;'>❌ Data File Not Found</h3>
            <p>The file <code>output/flagged_transactions.csv</code> could not be located.</p>
            <p><strong>Solution:</strong> Run the simulation first using:</p>
            <code>python 1_run_simulation.py</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    if df.empty:
        st.markdown("""
        <div class='info-card'>
            <h3 style='margin-top: 0;'>✅ All Systems Clear</h3>
            <p>No suspicious transactions detected. All transactions have passed security checks successfully.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        st.stop()
    
    # Key Metrics
    st.markdown("## 📊 Overview")
    
    denied_count = len(df[df['final_status'] == 'DENY'])
    flagged_count = len(df[df['final_status'] == 'FLAG_FOR_REVIEW'])
    avg_score = df['final_score'].mean()
    total_value = df['value_eth'].sum()
    high_risk = len(df[df['final_score'] >= 70])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Flagged",
            value=len(df),
            delta=f"-{len(df)} transactions",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Denied",
            value=denied_count,
            delta=f"{(denied_count/len(df)*100):.0f}% of total",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Under Review",
            value=flagged_count,
            delta=f"{(flagged_count/len(df)*100):.0f}% of total",
            delta_color="off"
        )
    
    with col4:
        st.metric(
            label="Total Value",
            value=f"{total_value:.2f} ETH",
            delta=f"≈ ${total_value * 2500:,.0f}"
        )
    
    with col5:
        st.metric(
            label="Avg Risk",
            value=f"{avg_score:.1f}",
            delta="Critical" if avg_score >= 70 else ("High" if avg_score >= 50 else "Medium"),
            delta_color="inverse" if avg_score >= 70 else "off"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("## 📈 Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Average Risk Score**")
        fig_gauge = create_risk_gauge(avg_score)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_main")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Status Distribution**")
        fig_status = create_status_distribution(df)
        st.plotly_chart(fig_status, use_container_width=True, key="status_dist")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Value Distribution**")
        fig_value = create_value_distribution(df)
        st.plotly_chart(fig_value, use_container_width=True, key="value_dist")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Timeline
    fig_timeline = create_risk_timeline(df)
    if fig_timeline:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Filters
    st.markdown("## 🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unique_statuses = df['final_status'].unique().tolist()
        status_filter = st.multiselect(
            "Transaction Status",
            options=unique_statuses,
            default=unique_statuses
        )
    
    with col2:
        min_score = float(df['final_score'].min())
        max_score = float(df['final_score'].max())
        score_range = st.slider(
            "Risk Score Range",
            min_value=int(min_score),
            max_value=int(max_score) + 10,
            value=(int(min_score), int(max_score) + 10),
            step=5
        )
    
    with col3:
        max_value = float(df['value_eth'].max())
        value_range = st.slider(
            "Value Range (ETH)",
            min_value=0.0,
            max_value=float(max_value) + 10.0,
            value=(0.0, float(max_value) + 10.0),
            step=1.0
        )
    
    # Apply filters
    filtered_df = df[
        (df['final_status'].isin(status_filter)) &
        (df['final_score'] >= score_range[0]) &
        (df['final_score'] <= score_range[1]) &
        (df['value_eth'] >= value_range[0]) &
        (df['value_eth'] <= value_range[1])
    ]
    
    st.markdown(f"<div class='info-card'><strong>Displaying {len(filtered_df)}</strong> of <strong>{len(df)}</strong> transactions</div>", unsafe_allow_html=True)
    
    # Transactions Table
    st.markdown("## 📋 Flagged Transactions")
    
    if filtered_df.empty:
        st.warning("⚠️ No transactions match your current filter criteria.")
        st.stop()
    
    # Prepare display dataframe
    display_df = filtered_df.copy()
    
    if 'from_address' in display_df.columns:
        display_df['From'] = display_df['from_address'].apply(lambda x: f"{x[:10]}...{x[-8:]}")
    if 'to_address' in display_df.columns:
        display_df['To'] = display_df['to_address'].apply(lambda x: f"{x[:10]}...{x[-8:]}")
    if 'tx_hash' in display_df.columns:
        display_df['Hash'] = display_df['tx_hash'].apply(lambda x: f"{x[:16]}...")
    
    display_columns = ['Hash', 'From', 'To', 'value_eth', 'final_score', 'final_status']
    display_df = display_df[display_columns]
    display_df.columns = ['Tx Hash', 'From', 'To', 'Value (ETH)', 'Risk', 'Status']
    
    # Color code the dataframe
    def color_status(val):
        if val == 'DENY':
            return 'background-color: #fee2e2; color: #dc2626; font-weight: 600'
        elif val == 'FLAG_FOR_REVIEW':
            return 'background-color: #fef3c7; color: #d97706; font-weight: 600'
        return ''
    
    def color_risk(val):
        if val >= 70:
            return 'background-color: #fee2e2; color: #dc2626; font-weight: 700'
        elif val >= 50:
            return 'background-color: #fef3c7; color: #d97706; font-weight: 600'
        return 'color: #16a34a; font-weight: 600'
    
    styled_df = display_df.style.applymap(color_status, subset=['Status'])\
                                .applymap(color_risk, subset=['Risk'])\
                                .format({'Value (ETH)': '{:.4f}', 'Risk': '{:.1f}'})
    
    st.dataframe(styled_df, use_container_width=True, height=450)
    
    # Detailed Analysis
    st.markdown("## 🔬 Transaction Details")
    
    selected_idx = st.selectbox(
        "Select a transaction to analyze:",
        range(len(filtered_df)),
        format_func=lambda i: f"#{i+1} • {filtered_df.iloc[i]['tx_hash'][:24]}... • Score: {filtered_df.iloc[i]['final_score']:.1f}"
    )
    
    if selected_idx is not None:
        tx_data = filtered_df.iloc[selected_idx]
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            status = tx_data['final_status']
            status_class = 'danger-card' if status == 'DENY' else 'warning-card'
            status_emoji = '⛔' if status == 'DENY' else '⚠️'
            
            st.markdown(f"""
            <div class='{status_class}'>
                <h3 style='margin-top: 0;'>{status_emoji} {status.replace('_', ' ').title()}</h3>
                <p style='margin-bottom: 0;'><strong>Risk Score:</strong> <span style='font-size: 1.5rem; font-weight: 700;'>{tx_data['final_score']:.1f}</span> / 150</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='info-card'>
                <h3 style='margin-top: 0;'>📝 Transaction Information</h3>
                <p><strong>Hash:</strong><br><code style='font-size: 0.875rem;'>{tx_data['tx_hash']}</code></p>
                <p><strong>From:</strong><br><code style='font-size: 0.875rem;'>{tx_data['from_address']}</code></p>
                <p><strong>To:</strong><br><code style='font-size: 0.875rem;'>{tx_data['to_address']}</code></p>
                <p><strong>Value:</strong> {tx_data['value_eth']:.4f} ETH <span style='color: #64748b;'>(≈ ${tx_data['value_eth'] * 2500:,.2f})</span></p>
                <p><strong>Gas Price:</strong> {tx_data['gas_price']:.2f} Gwei</p>
                <p style='margin-bottom: 0;'><strong>Timestamp:</strong> {tx_data['timestamp']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reasons
            st.markdown("### 🚩 Flagging Reasons")
            reasons_list = str(tx_data['reasons']).split(" | ")
            
            for i, reason in enumerate(reasons_list, 1):
                is_critical = any(word in reason for word in ['Dark Web', 'Mixer', 'Sanctioned', 'High'])
                card_class = 'danger-card' if is_critical else 'warning-card'
                st.markdown(f"""
                <div class='{card_class}'>
                    <strong>Reason {i}:</strong> {reason}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Risk Assessment")
            fig_tx_gauge = create_risk_gauge(tx_data['final_score'])
            st.plotly_chart(fig_tx_gauge, use_container_width=True, key=f"gauge_detail_{selected_idx}")
            
            st.markdown(f"""
            <div class='info-card'>
                <h4 style='margin-top: 0;'>Agent Scores</h4>
                <p><strong>🔍 Agent 1 (Threat Intel):</strong> {tx_data['agent_1_score']}</p>
                <p><strong>🧠 Agent 2 (Behavioral):</strong> {tx_data['agent_2_score']}</p>
                <p style='margin-bottom: 0;'><strong>⚖️ Final Score:</strong> <span style='font-size: 1.25rem; font-weight: 700; color: {"#dc2626" if tx_data["final_score"] >= 70 else "#d97706"};'>{tx_data['final_score']:.1f}</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
            <p style='font-size: 1rem; margin-bottom: 0.5rem;'><strong>🛡️ Blockchain Fraud Detection MVP</strong></p>
            <p style='font-size: 0.875rem; margin: 0;'>Powered by Multi-Agent AI • Real-time Analysis • Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()