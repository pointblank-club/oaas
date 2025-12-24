import React, { useMemo, useState, useEffect } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

interface CFGVisualizerProps {
  code: string;
  decompilerName: string;
}

interface BasicBlock {
  id: string;
  label: string;
  code: string;
  lineStart: number;
  lineEnd: number;
  type?: 'entry' | 'control' | 'return' | 'regular';
}


function parseCFG(code: string): { nodes: Node[]; edges: Edge[] } {
  const lines = code.split('\n');
  const blocks: BasicBlock[] = [];
  const edges: Edge[] = [];
  
  let currentBlock: BasicBlock | null = null;
  let blockCounter = 0;
  let braceDepth = 0;
  let inFunction = false;
  
  // Track function entry points
  const functionEntries: Map<string, number> = new Map();
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const originalLine = i + 1;
    
    // Detect function definitions
    if (line.match(/^[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*\{?/)) {
      if (currentBlock) {
        currentBlock.lineEnd = originalLine - 1;
        blocks.push(currentBlock);
      }
      const funcName = line.split('(')[0].trim();
      currentBlock = {
        id: `block_${blockCounter++}`,
        label: `Entry: ${funcName}`,
        code: line,
        lineStart: originalLine,
        lineEnd: originalLine,
        type: 'entry',
      };
      inFunction = true;
      functionEntries.set(funcName, blocks.length);
      continue;
    }
    
    // Detect control flow statements
    const isControlFlow = /^\s*(if|else|while|for|switch|case|default|return|goto|break|continue)\s*/.test(line);
    const isReturn = /^\s*return\s*/.test(line);
    const isBraceOpen = line.includes('{');
    const isBraceClose = line.includes('}');
    
    if (isBraceOpen) braceDepth++;
    if (isBraceClose) braceDepth--;
    
    
    if (isControlFlow || (isBraceClose && braceDepth === 0 && inFunction)) {
      if (currentBlock) {
        currentBlock.lineEnd = originalLine - 1;
        blocks.push(currentBlock);
        
        
        if (isControlFlow) {
          const nextBlockId = `block_${blockCounter}`;
          edges.push({
            id: `edge_${currentBlock.id}_${nextBlockId}`,
            source: currentBlock.id,
            target: nextBlockId,
            label: line.substring(0, 30),
            type: 'smoothstep',
            markerEnd: { type: MarkerType.ArrowClosed },
          });
        }
      }
      
      if (isBraceClose && braceDepth === 0) {
        inFunction = false;
        continue;
      }
      
      currentBlock = {
        id: `block_${blockCounter++}`,
        label: isControlFlow ? line.substring(0, 40) : `Block ${blockCounter}`,
        code: line,
        lineStart: originalLine,
        lineEnd: originalLine,
        type: isReturn ? 'return' : isControlFlow ? 'control' : 'regular',
      };
    } else if (currentBlock) {
      
      currentBlock.code += '\n' + line;
      
      if (/return\s*/.test(line) && !currentBlock.type) {
        currentBlock.type = 'return';
      }
    } else if (line && !line.startsWith('//') && !line.startsWith('/*')) {
      
      currentBlock = {
        id: `block_${blockCounter++}`,
        label: `Block ${blockCounter}`,
        code: line,
        lineStart: originalLine,
        lineEnd: originalLine,
        type: 'regular',
      };
    }
  }
  
  
  if (currentBlock) {
    currentBlock.lineEnd = lines.length;
    blocks.push(currentBlock);
  }
  
  
  for (let i = 0; i < blocks.length - 1; i++) {
    const hasExplicitEdge = edges.some(e => e.source === blocks[i].id);
    if (!hasExplicitEdge) {
      edges.push({
        id: `edge_${blocks[i].id}_${blocks[i + 1].id}`,
        source: blocks[i].id,
        target: blocks[i + 1].id,
        type: 'smoothstep',
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    }
  }
  
  
  const getBlockColor = (blockType?: string) => {
    switch (blockType) {
      case 'entry':
        return { bg: '#4a90e2', border: '#357abd', text: '#ffffff' }; // Blue for entry
      case 'control':
        return { bg: '#f39c12', border: '#d68910', text: '#ffffff' }; // Orange for control flow
      case 'return':
        return { bg: '#e74c3c', border: '#c0392b', text: '#ffffff' }; // Red for return
      default:
        return { bg: '#2c3e50', border: '#34495e', text: '#ecf0f1' }; // Dark gray for regular
    }
  };

  
  const nodes: Node[] = blocks.map((block, index) => {
    const codePreview = block.code.split('\n').slice(0, 3).join('\n');
    const truncatedCode = codePreview.length > 100 
      ? codePreview.substring(0, 100) + '...' 
      : codePreview;
    
    const colors = getBlockColor(block.type);
    
    return {
      id: block.id,
      type: 'default',
      position: { 
        x: (index % 3) * 300, 
        y: Math.floor(index / 3) * 150 
      },
      data: {
        label: (
          <div style={{ padding: '8px', fontSize: '0.85em' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '4px', color: colors.text }}>
              {block.label}
            </div>
            <div style={{ 
              fontFamily: 'monospace', 
              fontSize: '0.75em',
              color: colors.text,
              opacity: 0.9,
              maxWidth: '250px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'pre-wrap'
            }}>
              {truncatedCode}
            </div>
            <div style={{ fontSize: '0.7em', color: colors.text, opacity: 0.8, marginTop: '4px' }}>
              Lines {block.lineStart}-{block.lineEnd}
            </div>
          </div>
        ),
      },
      style: {
        background: colors.bg,
        border: `2px solid ${colors.border}`,
        borderRadius: '8px',
        color: colors.text,
        width: 280,
        minHeight: 80,
      },
    };
  });
  
  return { nodes, edges };
}

export const CFGVisualizer: React.FC<CFGVisualizerProps> = ({ code, decompilerName }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const { nodes, edges } = useMemo(() => {
    if (!code) return { nodes: [], edges: [] };
    return parseCFG(code);
  }, [code]);

  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    if (!isFullscreen) {
      const element = document.getElementById('cfg-fullscreen-container');
      if (element) {
        if (element.requestFullscreen) {
          element.requestFullscreen();
        } else if ((element as any).webkitRequestFullscreen) {
          (element as any).webkitRequestFullscreen();
        } else if ((element as any).mozRequestFullScreen) {
          (element as any).mozRequestFullScreen();
        } else if ((element as any).msRequestFullscreen) {
          (element as any).msRequestFullscreen();
        }
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen();
      } else if ((document as any).mozCancelFullScreen) {
        (document as any).mozCancelFullScreen();
      } else if ((document as any).msExitFullscreen) {
        (document as any).msExitFullscreen();
      }
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!(
        document.fullscreenElement ||
        (document as any).webkitFullscreenElement ||
        (document as any).mozFullScreenElement ||
        (document as any).msFullscreenElement
      ));
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);
  
  if (!code || nodes.length === 0) {
    return (
      <div style={{ 
        padding: '40px', 
        textAlign: 'center', 
        color: 'var(--text-secondary)' 
      }}>
        <p>No control flow graph available for this decompiled code.</p>
        <p style={{ fontSize: '0.9em', marginTop: '10px' }}>
          CFG visualization requires structured code with control flow statements.
        </p>
      </div>
    );
  }
  
  const containerStyle: React.CSSProperties = isFullscreen
    ? {
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 9999,
        background: 'var(--bg-primary)',
      }
    : {
        width: '100%',
        height: '600px',
        position: 'relative',
      };
  
  return (
    <div id="cfg-fullscreen-container" style={containerStyle}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        style={{ background: 'var(--bg-primary)', width: '100%', height: '100%' }}
      >
        <Background color="var(--border-color)" gap={16} />
        <Controls 
          style={{ 
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)'
          }} 
        />
        <MiniMap 
          style={{ 
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)'
          }}
          nodeColor="var(--accent)"
        />
      </ReactFlow>
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        padding: '8px 12px',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '4px',
        fontSize: '0.85em',
        color: 'var(--text-secondary)',
        zIndex: 10
      }}>
        <strong>CFG:</strong> {nodes.length} blocks, {edges.length} edges ({decompilerName})
      </div>
      <button
        onClick={toggleFullscreen}
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          padding: '8px 12px',
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '4px',
          fontSize: '0.85em',
          color: 'var(--text-primary)',
          cursor: 'pointer',
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'var(--accent)';
          e.currentTarget.style.color = 'var(--bg-primary)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'var(--bg-secondary)';
          e.currentTarget.style.color = 'var(--text-primary)';
        }}
        title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
      >
        {isFullscreen ? (
          <>
            <span>⤓</span>
            <span>Exit Fullscreen</span>
          </>
        ) : (
          <>
            <span>⤢</span>
            <span>Fullscreen</span>
          </>
        )}
      </button>
      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: '10px',
        left: '10px',
        padding: '10px 12px',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '4px',
        fontSize: '0.75em',
        color: 'var(--text-secondary)',
        zIndex: 10,
        display: 'flex',
        gap: '16px',
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '16px', height: '16px', background: '#4a90e2', border: '2px solid #357abd', borderRadius: '4px' }}></div>
          <span>Entry</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '16px', height: '16px', background: '#f39c12', border: '2px solid #d68910', borderRadius: '4px' }}></div>
          <span>Control Flow</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '16px', height: '16px', background: '#e74c3c', border: '2px solid #c0392b', borderRadius: '4px' }}></div>
          <span>Return</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{ width: '16px', height: '16px', background: '#2c3e50', border: '2px solid #34495e', borderRadius: '4px' }}></div>
          <span>Regular</span>
        </div>
      </div>
    </div>
  );
};

