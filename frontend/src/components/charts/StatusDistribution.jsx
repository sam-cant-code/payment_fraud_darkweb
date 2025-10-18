import React from 'react';
import Plot from 'react-plotly.js';

// Now uses approved, denied, and review counts from the data prop
const StatusDistribution = ({ data }) => {
  // Destructure all relevant counts from the data prop passed by DashboardPage
  const { approved = 0, denied = 0, review = 0 } = data; // Default to 0 if undefined

  // Handle case with no data to avoid Plotly error
  if (approved === 0 && denied === 0 && review === 0) {
    return <div className="flex items-center justify-center h-full min-h-[250px] text-brand-gray">No transactions scanned yet.</div>;
  }

  // Define values, labels, and colors including 'APPROVE'
  const plotValues = [approved, denied, review];
  const plotLabels = ['APPROVE', 'DENY', 'FLAG FOR REVIEW'];
  const plotColors = ['#16a34a', '#ef4444', '#f59e0b']; // Green for Approve, Red for Deny, Amber for Review

  return (
    <Plot
      data={[
        {
          values: plotValues,
          labels: plotLabels,
          type: 'pie',
          hole: 0.5, // Keeps the donut chart style
          marker: {
            colors: plotColors,
            line: {
              color: 'white',
              width: 3
            }
          },
          textposition: 'outside', // Position labels outside the slices
          textinfo: 'label+percent', // Show label and percentage
          insidetextorientation: 'radial', // Orientation if text were inside
          hoverinfo: 'label+percent+value', // Show details on hover
          sort: false, // Keep the order defined in labels/values
        },
      ]}
      layout={{
        height: 250, // Adjust height as needed
        margin: { t: 20, b: 20, l: 20, r: 20 },
        showlegend: false, // Legend is redundant with outside labels
        paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' }
      }}
      config={{ responsive: true, displayModeBar: false }} // Responsive and no mode bar
      style={{ width: '100%' }}
    />
  );
};

export default StatusDistribution;