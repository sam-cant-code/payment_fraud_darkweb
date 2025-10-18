import React from 'react';
import Plot from 'react-plotly.js';

const StatusDistribution = ({ data }) => {
  const { denied, review } = data;
  
  // Handle case with no data to avoid Plotly error
  if (denied === 0 && review === 0) {
    return <div className="flex items-center justify-center h-full min-h-[250px] text-brand-gray">No 'Deny' or 'Review' transactions.</div>;
  }

  return (
    <Plot
      data={[
        {
          values: [denied, review],
          labels: ['DENY', 'FLAG FOR REVIEW'],
          type: 'pie',
          hole: 0.5,
          marker: {
            colors: ['#ef4444', '#f59e0b'],
            line: {
              color: 'white',
              width: 3
            }
          },
          textposition: 'outside',
          textinfo: 'label+percent',
        },
      ]}
      layout={{
        height: 250,
        margin: { t: 20, b: 20, l: 20, r: 20 },
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' }
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
};

export default StatusDistribution;