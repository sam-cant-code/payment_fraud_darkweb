// frontend/src/components/charts/NetworkGraph.jsx
import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

/**
 * NetworkGraph Component
 * Visualizes transaction flow patterns from the last N transactions
 * Shows wallet addresses as nodes and transactions as edges
 */
const NetworkGraph = ({ transactions, maxTransactions = 50 }) => {
  const graphData = useMemo(() => {
    if (!transactions || transactions.length === 0) {
      return { nodes: [], edges: [] };
    }

    // Take the most recent transactions
    const recentTxs = transactions.slice(0, maxTransactions);

    // Build node and edge lists
    const nodeSet = new Set();
    const edges = [];
    const edgeWeights = new Map(); // Track multiple transactions between same wallets

    recentTxs.forEach((tx) => {
      const from = tx.from_address;
      const to = tx.to_address;
      
      nodeSet.add(from);
      nodeSet.add(to);

      const edgeKey = `${from}->${to}`;
      edgeWeights.set(edgeKey, (edgeWeights.get(edgeKey) || 0) + 1);
      
      edges.push({
        from,
        to,
        value: tx.value_eth,
        score: tx.final_score,
        status: tx.final_status,
      });
    });

    // Create node list with shortened labels
    const nodes = Array.from(nodeSet).map((address) => ({
      id: address,
      label: `${address.slice(0, 6)}...${address.slice(-4)}`,
      // Count how many times this wallet appears
      txCount: edges.filter(e => e.from === address || e.to === address).length,
    }));

    return { nodes, edges, edgeWeights };
  }, [transactions, maxTransactions]);

  // Detect circular patterns (simple heuristic)
  const circularPaths = useMemo(() => {
    const paths = [];
    const { edges } = graphData;
    
    // Find simple circular patterns (A->B->C->A within 3 hops)
    edges.forEach((edge1) => {
      edges.forEach((edge2) => {
        if (edge1.to === edge2.from) {
          edges.forEach((edge3) => {
            if (edge2.to === edge3.from && edge3.to === edge1.from) {
              paths.push([edge1.from, edge1.to, edge2.to]);
            }
          });
        }
      });
    });
    
    return paths.length;
  }, [graphData]);

  // Detect high fan-out (potential smurfing)
  const highFanOut = useMemo(() => {
    const { nodes, edges } = graphData;
    return nodes.filter((node) => {
      const outgoing = edges.filter(e => e.from === node.id).length;
      return outgoing > 5; // Threshold for "high"
    }).length;
  }, [graphData]);

  // Detect high fan-in (potential collection)
  const highFanIn = useMemo(() => {
    const { nodes, edges } = graphData;
    return nodes.filter((node) => {
      const incoming = edges.filter(e => e.to === node.id).length;
      return incoming > 5; // Threshold for "high"
    }).length;
  }, [graphData]);

  if (graphData.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full min-h-[400px] text-brand-gray">
        No transaction network data available.
      </div>
    );
  }

  // Prepare data for Plotly
  // We'll use a simple force-directed layout approximation
  // Position nodes in a circle for simplicity
  const nodeCount = graphData.nodes.length;
  const nodePositions = graphData.nodes.map((node, i) => {
    const angle = (2 * Math.PI * i) / nodeCount;
    return {
      x: Math.cos(angle) * 100,
      y: Math.sin(angle) * 100,
      ...node,
    };
  });

  // Create edge traces (lines between nodes)
  const edgeTraces = graphData.edges.map((edge, i) => {
    const fromNode = nodePositions.find(n => n.id === edge.from);
    const toNode = nodePositions.find(n => n.id === edge.to);
    
    if (!fromNode || !toNode) return null;

    const color = 
      edge.status === 'DENY' ? '#ef4444' :
      edge.status === 'FLAG_FOR_REVIEW' ? '#f59e0b' :
      '#16a34a';

    return {
      type: 'scatter',
      mode: 'lines',
      x: [fromNode.x, toNode.x],
      y: [fromNode.y, toNode.y],
      line: {
        color: color,
        width: Math.min(1 + edge.value / 10, 5), // Thicker for higher values
      },
      hoverinfo: 'text',
      text: `${edge.from.slice(0, 8)}... → ${edge.to.slice(0, 8)}...<br>${edge.value.toFixed(4)} ETH<br>Risk: ${edge.score.toFixed(0)}`,
      showlegend: false,
    };
  }).filter(Boolean);

  // Create node trace
  const nodeTrace = {
    type: 'scatter',
    mode: 'markers+text',
    x: nodePositions.map(n => n.x),
    y: nodePositions.map(n => n.y),
    text: nodePositions.map(n => n.label),
    textposition: 'top center',
    marker: {
      size: nodePositions.map(n => Math.min(10 + n.txCount * 2, 30)), // Size by activity
      color: nodePositions.map(n => {
        // Color by activity level
        if (n.txCount > 10) return '#ef4444'; // Red for very active
        if (n.txCount > 5) return '#f59e0b'; // Amber for active
        return '#3b82f6'; // Blue for normal
      }),
      line: { color: 'white', width: 2 },
    },
    hoverinfo: 'text',
    hovertext: nodePositions.map(n => `${n.label}<br>Transactions: ${n.txCount}`),
    showlegend: false,
  };

  return (
    <div>
      {/* Pattern Detection Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-100 p-3 rounded-md text-center">
          <p className="text-2xl font-bold text-slate-800">{circularPaths}</p>
          <p className="text-xs text-slate-600 uppercase">Circular Patterns</p>
        </div>
        <div className="bg-slate-100 p-3 rounded-md text-center">
          <p className="text-2xl font-bold text-slate-800">{highFanOut}</p>
          <p className="text-xs text-slate-600 uppercase">High Fan-Out (Smurfing)</p>
        </div>
        <div className="bg-slate-100 p-3 rounded-md text-center">
          <p className="text-2xl font-bold text-slate-800">{highFanIn}</p>
          <p className="text-xs text-slate-600 uppercase">High Fan-In (Collection)</p>
        </div>
      </div>

      {/* Network Graph */}
      <Plot
        data={[...edgeTraces, nodeTrace]}
        layout={{
          height: 500,
          margin: { t: 20, b: 20, l: 20, r: 20 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          xaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false,
          },
          yaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false,
          },
          hovermode: 'closest',
          font: { family: 'Inter, sans-serif' },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-600">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-status-deny-text"></div>
          <span>Denied Transaction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-status-review-text"></div>
          <span>Flagged Transaction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-status-approve-text"></div>
          <span>Approved Transaction</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-0.5 bg-slate-400"></div>
          <span>Larger nodes = More active wallets</span>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500 italic">
        Showing last {Math.min(maxTransactions, graphData.edges.length)} transactions. 
        Node size indicates transaction count. Edge thickness indicates value.
      </p>
    </div>
  );
};

export default NetworkGraph;

// ==============================================
// HOW TO ADD TO DashboardPage.jsx
// ==============================================
/*
1. Import the component at the top:
   import NetworkGraph from '../components/charts/NetworkGraph';

2. Add this section after the Timeline Chart (around line 155):

      {/* 5. Network Analysis Graph * /}
     

3. This will display:
   - Visual network graph of recent transactions
   - Detection metrics for circular patterns, fan-out, and fan-in
   - Color-coded edges by risk status
   - Node sizes based on transaction frequency
*/