import React from 'react';
import Plot from 'react-plotly.js';

const RiskTimeline = ({ data }) => {
  // Trace for DENY status
  const denyTrace = {
    x: data.filter(t => t.final_status === 'DENY').map(t => t.timestamp),
    y: data.filter(t => t.final_status === 'DENY').map(t => t.final_score),
    mode: 'markers',
    type: 'scatter',
    name: 'Deny',
    marker: { color: '#ef4444', size: 10, opacity: 0.8 }, // Red
  };

  // Trace for FLAG_FOR_REVIEW status
  const reviewTrace = {
    x: data.filter(t => t.final_status === 'FLAG_FOR_REVIEW').map(t => t.timestamp),
    y: data.filter(t => t.final_status === 'FLAG_FOR_REVIEW').map(t => t.final_score),
    mode: 'markers',
    type: 'scatter',
    name: 'Review',
    marker: { color: '#f59e0b', size: 10, opacity: 0.8 }, // Amber
  };

  // --- New Trace for APPROVE status ---
  const approveTrace = {
    x: data.filter(t => t.final_status === 'APPROVE').map(t => t.timestamp),
    y: data.filter(t => t.final_status === 'APPROVE').map(t => t.final_score),
    mode: 'markers',
    type: 'scatter',
    name: 'Approve',
    marker: { color: '#16a34a', size: 8, opacity: 0.6 }, // Green, slightly smaller/lighter
  };
  // --- End New Trace ---

  // Determine the overall maximum score for y-axis range, adding padding
  const maxScore = data.length > 0 ? Math.max(...data.map(t => t.final_score)) : 100; // Default max if no data
  const yAxisRange = [0, Math.max(20, maxScore + 10)]; // Ensure minimum range, add padding

  return (
    <Plot
      // Add approveTrace to the data array
      data={[denyTrace, reviewTrace, approveTrace]}
      layout={{
        height: 350,
        margin: { t: 20, b: 40, l: 40, r: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif', color: '#64748b' },
        xaxis: {
          title: 'Timestamp',
          gridcolor: '#e2e8f0',
        },
        yaxis: {
          title: 'Risk Score',
          gridcolor: '#e2e8f0',
          range: yAxisRange // Use calculated range
        },
        legend: {
          orientation: 'h',
          yanchor: 'bottom',
          y: 1.02,
          xanchor: 'right',
          x: 1
        }
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
};

export default RiskTimeline;