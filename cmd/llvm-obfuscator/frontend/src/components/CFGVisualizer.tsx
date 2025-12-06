import React, { useMemo } from 'react';
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
}

/**
 * Simple CFG parser for decompiled C code
 * Extracts basic blocks and control flow edges
 */
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
      };
      inFunction = true;
      functionEntries.set(funcName, blocks.length);
      continue;
    }
    
    // Detect control flow statements
    const isControlFlow = /^\s*(if|else|while|for|switch|case|default|return|goto|break|continue)\s*/.test(line);
    const isBraceOpen = line.includes('{');
    const isBraceClose = line.includes('}');
    
    if (isBraceOpen) braceDepth++;
    if (isBraceClose) braceDepth--;
    
    // Start new block on control flow or function end
    if (isControlFlow || (isBraceClose && braceDepth === 0 && inFunction)) {
      if (currentBlock) {
        currentBlock.lineEnd = originalLine - 1;
        blocks.push(currentBlock);
        
        // Create edge for control flow
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
      };
    } else if (currentBlock) {
      // Add line to current block
      currentBlock.code += '\n' + line;
    } else if (line && !line.startsWith('//') && !line.startsWith('/*')) {
      // Start new block for non-comment code
      currentBlock = {
        id: `block_${blockCounter++}`,
        label: `Block ${blockCounter}`,
        code: line,
        lineStart: originalLine,
        lineEnd: originalLine,
      };
    }
  }
  
  // Add final block
  if (currentBlock) {
    currentBlock.lineEnd = lines.length;
    blocks.push(currentBlock);
  }
  
  // Create sequential edges between blocks
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
  
  // Convert to ReactFlow nodes
  const nodes: Node[] = blocks.map((block, index) => {
    const codePreview = block.code.split('\n').slice(0, 3).join('\n');
    const truncatedCode = codePreview.length > 100 
      ? codePreview.substring(0, 100) + '...' 
      : codePreview;
    
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
            <div style={{ fontWeight: 'bold', marginBottom: '4px', color: 'var(--accent)' }}>
              {block.label}
            </div>
            <div style={{ 
              fontFamily: 'monospace', 
              fontSize: '0.75em',
              color: 'var(--text-secondary)',
              maxWidth: '250px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'pre-wrap'
            }}>
              {truncatedCode}
            </div>
            <div style={{ fontSize: '0.7em', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Lines {block.lineStart}-{block.lineEnd}
            </div>
          </div>
        ),
      },
      style: {
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)',
        borderRadius: '8px',
        color: 'var(--text-primary)',
        width: 280,
        minHeight: 80,
      },
    };
  });
  
  return { nodes, edges };
}

export const CFGVisualizer: React.FC<CFGVisualizerProps> = ({ code, decompilerName }) => {
  const { nodes, edges } = useMemo(() => {
    if (!code) return { nodes: [], edges: [] };
    return parseCFG(code);
  }, [code]);
  
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
  
  return (
    <div style={{ width: '100%', height: '600px', position: 'relative' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        style={{ background: 'var(--bg-primary)' }}
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
    </div>
  );
};

