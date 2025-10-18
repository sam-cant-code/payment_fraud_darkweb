import React from 'react';
import Plot from 'react-plotly.js';

const RiskGauge = ({ score }) => {
  return (
    <Plot
      data={[
        {
          type: 'indicator',
          mode: 'gauge+number',
          value: score,
          gauge: {
            axis: { range: [0, 150], tickwidth: 1, tickcolor: 'darkblue' },
            bar: { color: '#0f172a' },
            bgcolor: 'white',
            borderwidth: 2,
            bordercolor: '#e2e8f0',
            steps: [
              { range: [0, 30], color: '#dcfce7' },
              { range: [30, 70], color: '#fef3c7' },
              { range: [70, 150], color: '#fee2e2' },
            ],
            threshold: {
              line: { color: '#dc2626', width: 4 },
              thickness: 0.75,
              value: 70,
            },
          },
        },
      ]}
      layout={{
        height: 250,
        margin: { t: 0, b: 0, l: 30, r: 30 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' }
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
};

export default RiskGauge;