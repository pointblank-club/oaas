import React, { useMemo, useState, useCallback } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Ghidra CFG JSON structure
interface GhidraCFGFunction {
  name: string;
  address: string;
  size: number;
  basic_blocks: Array<{
    start_addr: string;
    end_addr: string;
    size: number;
  }>;
  edges: Array<{
    from: string;
    to: string;
    type: 'branch' | 'flow';
  }>;
}

interface GhidraCFG {
  format?: string;
  version?: string;
  binary_info?: {
    path: string;
    architecture: string;
    base_address: string;
  };
  functions: GhidraCFGFunction[];
}

interface BinaryCFGVisualizerProps {
  cfgData: GhidraCFG | null;
  isLoading?: boolean;
  error?: string;
}

/**
 * Convert Ghidra CFG JSON to ReactFlow nodes and edges
 */
function convertCFGToFlow(cfg: GhidraCFG, selectedFunction: string | null): { nodes: Node[]; edges: Edge[] } {
  if (!cfg?.functions?.length) {
    return { nodes: [], edges: [] };
  }

  // If no function selected, show function overview
  if (!selectedFunction) {
    return createFunctionOverview(cfg);
  }

  // Find selected function
  const func = cfg.functions.find(f => f.name === selectedFunction || f.address === selectedFunction);
  if (!func) {
    return { nodes: [], edges: [] };
  }

  return createFunctionCFG(func);
}

/**
 * Create overview showing all functions as nodes
 */
function createFunctionOverview(cfg: GhidraCFG): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const COLS = 4;
  const NODE_WIDTH = 220;
  const NODE_HEIGHT = 100;
  const PADDING = 30;

  cfg.functions.forEach((func, idx) => {
    const col = idx % COLS;
    const row = Math.floor(idx / COLS);

    // Color based on function characteristics
    const isEntry = func.name.includes('main') || func.name.includes('entry') || func.name.includes('start');
    const isExternal = func.name.startsWith('_') || func.name.includes('@');

    const bgColor = isEntry ? '#4a90e2' : isExternal ? '#9b59b6' : '#2c3e50';
    const borderColor = isEntry ? '#357abd' : isExternal ? '#8e44ad' : '#34495e';

    nodes.push({
      id: func.address,
      position: {
        x: col * (NODE_WIDTH + PADDING),
        y: row * (NODE_HEIGHT + PADDING)
      },
      data: {
        label: (
          <div style={{ padding: '8px', textAlign: 'center' }}>
            <div style={{ fontWeight: 'bold', fontSize: '0.9em', marginBottom: '4px', color: '#fff' }}>
              {func.name.length > 25 ? func.name.substring(0, 22) + '...' : func.name}
            </div>
            <div style={{ fontSize: '0.75em', color: 'rgba(255,255,255,0.8)' }}>
              @ {func.address}
            </div>
            <div style={{ fontSize: '0.7em', color: 'rgba(255,255,255,0.7)', marginTop: '4px' }}>
              {func.basic_blocks.length} blocks | {func.edges.length} edges
            </div>
          </div>
        ),
      },
      style: {
        background: bgColor,
        border: `2px solid ${borderColor}`,
        borderRadius: '8px',
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT - 20,
        cursor: 'pointer',
      },
    });
  });

  return { nodes, edges: [] };
}

/**
 * Create detailed CFG for a single function
 */
function createFunctionCFG(func: GhidraCFGFunction): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Create nodes for each basic block
  const blockCount = func.basic_blocks.length;
  const COLS = Math.min(4, Math.ceil(Math.sqrt(blockCount)));
  const NODE_WIDTH = 200;
  const NODE_HEIGHT = 80;
  const PADDING = 50;

  // Map addresses to block indices for edge creation
  const blockMap = new Map<string, string>();
  func.basic_blocks.forEach((block, idx) => {
    blockMap.set(block.start_addr, `block_${idx}`);
  });

  // Determine entry block (first block or lowest address)
  const entryAddr = func.basic_blocks.length > 0 ? func.basic_blocks[0].start_addr : null;

  func.basic_blocks.forEach((block, idx) => {
    const col = idx % COLS;
    const row = Math.floor(idx / COLS);

    const isEntry = block.start_addr === entryAddr;
    const isExit = !func.edges.some(e => e.from.startsWith(block.start_addr.substring(0, 8)));

    const bgColor = isEntry ? '#4a90e2' : isExit ? '#e74c3c' : '#2c3e50';
    const borderColor = isEntry ? '#357abd' : isExit ? '#c0392b' : '#34495e';

    nodes.push({
      id: `block_${idx}`,
      position: {
        x: col * (NODE_WIDTH + PADDING),
        y: row * (NODE_HEIGHT + PADDING)
      },
      data: {
        label: (
          <div style={{ padding: '6px', textAlign: 'center' }}>
            <div style={{ fontWeight: 'bold', fontSize: '0.85em', color: '#fff' }}>
              {isEntry ? 'ENTRY' : isExit ? 'EXIT' : `Block ${idx}`}
            </div>
            <div style={{ fontSize: '0.7em', color: 'rgba(255,255,255,0.8)', marginTop: '2px' }}>
              {block.start_addr}
            </div>
            <div style={{ fontSize: '0.65em', color: 'rgba(255,255,255,0.6)' }}>
              {block.size} bytes
            </div>
          </div>
        ),
      },
      style: {
        background: bgColor,
        border: `2px solid ${borderColor}`,
        borderRadius: '6px',
        width: NODE_WIDTH,
        minHeight: NODE_HEIGHT - 20,
      },
    });
  });

  // Create edges
  func.edges.forEach((edge, idx) => {
    // Find source block
    let sourceId: string | null = null;
    for (const [addr, id] of blockMap.entries()) {
      // Check if edge.from starts with block address (instruction within block)
      if (edge.from.startsWith(addr.substring(0, 8))) {
        sourceId = id;
        break;
      }
    }

    // Find target block
    const targetId = blockMap.get(edge.to);

    if (sourceId && targetId) {
      edges.push({
        id: `edge_${idx}`,
        source: sourceId,
        target: targetId,
        type: 'smoothstep',
        animated: edge.type === 'branch',
        style: {
          stroke: edge.type === 'branch' ? '#f39c12' : '#3498db',
          strokeWidth: 2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: edge.type === 'branch' ? '#f39c12' : '#3498db',
        },
        label: edge.type === 'branch' ? 'branch' : '',
        labelStyle: { fontSize: '0.7em', fill: '#999' },
      });
    }
  });

  return { nodes, edges };
}

export const BinaryCFGVisualizer: React.FC<BinaryCFGVisualizerProps> = ({
  cfgData,
  isLoading,
  error
}) => {
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!cfgData) return { nodes: [], edges: [] };
    return convertCFGToFlow(cfgData, selectedFunction);
  }, [cfgData, selectedFunction]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes/edges when CFG or selection changes
  React.useEffect(() => {
    if (!cfgData) return;
    const { nodes: newNodes, edges: newEdges } = convertCFGToFlow(cfgData, selectedFunction);
    setNodes(newNodes);
    setEdges(newEdges);
  }, [cfgData, selectedFunction, setNodes, setEdges]);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    if (!selectedFunction) {
      // In overview mode, clicking a function shows its CFG
      const func = cfgData?.functions.find(f => f.address === node.id);
      if (func) {
        setSelectedFunction(func.name);
      }
    }
  }, [selectedFunction, cfgData]);

  const toggleFullscreen = () => {
    const element = document.getElementById('binary-cfg-container');
    if (!element) return;

    if (!isFullscreen) {
      if (element.requestFullscreen) {
        element.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
    setIsFullscreen(!isFullscreen);
  };

  if (isLoading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: 'var(--text-secondary)',
        backgroundColor: 'var(--bg-tertiary)',
        borderRadius: '8px'
      }}>
        <div style={{ fontSize: '2em', marginBottom: '10px' }}>⟳</div>
        <p>Loading CFG data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '20px',
        textAlign: 'center',
        color: 'var(--danger)',
        backgroundColor: 'var(--bg-tertiary)',
        borderRadius: '8px',
        border: '1px solid var(--danger)'
      }}>
        <p>Failed to load CFG: {error}</p>
      </div>
    );
  }

  if (!cfgData || !cfgData.functions?.length) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: 'var(--text-secondary)',
        backgroundColor: 'var(--bg-tertiary)',
        borderRadius: '8px'
      }}>
        <p>No CFG data available yet.</p>
        <p style={{ fontSize: '0.9em', marginTop: '10px' }}>
          CFG will be displayed after Ghidra analysis completes.
        </p>
      </div>
    );
  }

  const totalBlocks = cfgData.functions.reduce((sum, f) => sum + f.basic_blocks.length, 0);
  const totalEdges = cfgData.functions.reduce((sum, f) => sum + f.edges.length, 0);

  return (
    <div
      id="binary-cfg-container"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '8px',
        border: '1px solid var(--border-color)',
        overflow: 'hidden',
        ...(isFullscreen ? {
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 9999,
          borderRadius: 0,
        } : {})
      }}
    >
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: 'var(--bg-tertiary)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '1.2em' }}>🔍</span>
            Binary CFG Visualization
          </h4>
          {selectedFunction ? (
            <button
              onClick={() => setSelectedFunction(null)}
              style={{
                padding: '4px 12px',
                backgroundColor: 'var(--accent)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.85em'
              }}
            >
              ← Back to Overview
            </button>
          ) : null}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ fontSize: '0.85em', color: 'var(--text-secondary)' }}>
            {selectedFunction
              ? `Function: ${selectedFunction}`
              : `${cfgData.functions.length} functions | ${totalBlocks} blocks | ${totalEdges} edges`
            }
          </span>
          <button
            onClick={toggleFullscreen}
            style={{
              padding: '6px 12px',
              backgroundColor: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.85em'
            }}
          >
            {isFullscreen ? '⤓ Exit' : '⤢ Fullscreen'}
          </button>
        </div>
      </div>

      {/* Function selector dropdown */}
      {!selectedFunction && (
        <div style={{
          padding: '8px 16px',
          borderBottom: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-secondary)'
        }}>
          <select
            value=""
            onChange={(e) => setSelectedFunction(e.target.value)}
            style={{
              padding: '6px 12px',
              backgroundColor: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              fontSize: '0.9em',
              cursor: 'pointer',
              minWidth: '250px'
            }}
          >
            <option value="">Select a function to view its CFG...</option>
            {cfgData.functions.map((func) => (
              <option key={func.address} value={func.name}>
                {func.name} ({func.basic_blocks.length} blocks)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* ReactFlow Canvas */}
      <div style={{ height: isFullscreen ? 'calc(100vh - 100px)' : '500px' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          style={{ background: 'var(--bg-primary)' }}
        >
          <Background color="var(--border-color)" gap={20} />
          <Controls
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px'
            }}
          />
          <MiniMap
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-color)'
            }}
            nodeColor={(node) => {
              const style = node.style as React.CSSProperties;
              return (style?.background as string) || '#2c3e50';
            }}
          />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div style={{
        padding: '10px 16px',
        borderTop: '1px solid var(--border-color)',
        backgroundColor: 'var(--bg-tertiary)',
        display: 'flex',
        gap: '20px',
        flexWrap: 'wrap',
        fontSize: '0.8em',
        color: 'var(--text-secondary)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '14px', height: '14px', background: '#4a90e2', borderRadius: '3px' }} />
          <span>{selectedFunction ? 'Entry Block' : 'Entry/Main'}</span>
        </div>
        {selectedFunction ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '14px', height: '14px', background: '#e74c3c', borderRadius: '3px' }} />
            <span>Exit Block</span>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '14px', height: '14px', background: '#9b59b6', borderRadius: '3px' }} />
            <span>External/Library</span>
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '14px', height: '14px', background: '#2c3e50', borderRadius: '3px' }} />
          <span>Regular Block</span>
        </div>
        {selectedFunction && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#f39c12' }} />
              <span>Branch Edge</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#3498db' }} />
              <span>Flow Edge</span>
            </div>
          </>
        )}
        <div style={{ marginLeft: 'auto', fontStyle: 'italic' }}>
          {selectedFunction ? 'Click "Back to Overview" to see all functions' : 'Click a function to view its control flow graph'}
        </div>
      </div>
    </div>
  );
};

export default BinaryCFGVisualizer;
