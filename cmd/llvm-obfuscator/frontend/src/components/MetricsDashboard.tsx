/**
 * Advanced Metrics Dashboard Component
 * Displays comprehensive LLVM obfuscation metrics with interactive visualizations
 */

import React, { useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, RadarChart, Radar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Treemap,
} from 'recharts';

interface ControlFlowMetricsData {
  baseline: {
    basic_blocks: number;
    cfg_edges: number;
    cyclomatic_complexity: number;
    functions: number;
    loops: number;
  };
  obfuscated: {
    basic_blocks: number;
    cfg_edges: number;
    cyclomatic_complexity: number;
    functions: number;
    loops: number;
  };
  comparison: {
    complexity_increase_percent: number;
    basic_blocks_added: number;
    cfg_edges_added: number;
  };
}

interface InstructionMetricsData {
  baseline: {
    total_instructions: number;
    instruction_distribution: {
      load: number;
      store: number;
      call: number;
      br: number;
      phi: number;
      arithmetic: number;
      other: number;
    };
  };
  obfuscated: {
    total_instructions: number;
    instruction_distribution: {
      load: number;
      store: number;
      call: number;
      br: number;
      phi: number;
      arithmetic: number;
      other: number;
    };
  };
  comparison: {
    instruction_growth_percent: number;
  };
}

interface ReportMetadata {
  platform?: string;
  architecture?: string;
  binary_format?: string;
  metric_extraction_method?: string;
  [key: string]: any;
}

interface ReportData {
  obfuscation_score?: number;
  overall_protection_index?: number;
  control_flow_metrics?: ControlFlowMetricsData;
  instruction_metrics?: InstructionMetricsData;
  metadata?: ReportMetadata;
  input_parameters?: {
    platform?: string;
    [key: string]: any;
  };
}

interface PhoronixMetrics {
  instruction_count_delta?: number;
  instruction_count_increase_percent?: number;
  performance_overhead_percent?: number;
}

interface Props {
  report: ReportData & { phoronix?: { key_metrics?: PhoronixMetrics } };
}

// Color constants for consistent theming
const COLORS = {
  primary: '#1f6feb',
  success: '#2ea043',
  warning: '#d29922',
  danger: '#da3633',
  purple: '#8957e5',
  orange: '#fb8500',
  cyan: '#00d4ff',
  pink: '#ff006e',
};

const INSTRUCTION_COLORS: { [key: string]: string } = {
  load: '#1f6feb',
  store: '#0969da',
  call: '#2ea043',
  br: '#d29922',
  phi: '#8957e5',
  arithmetic: '#da3633',
  other: '#6e40aa',
};

/**
 * Protection Score Card - Main summary widget
 * Displays score on 0-100 scale (industry-standard PRCS framework)
 */
const ProtectionScoreCard: React.FC<{ score?: number }> = ({ score = 0 }) => {
  // Score is already on 0-100 scale
  const scoreValue = Math.round(score);

  const getGrade = (scoreVal: number): string => {
    if (scoreVal >= 85) return 'A+';
    if (scoreVal >= 75) return 'A';
    if (scoreVal >= 65) return 'B+';
    if (scoreVal >= 55) return 'B';
    if (scoreVal >= 45) return 'C';
    return 'D';
  };

  const getColor = (scoreVal: number): string => {
    if (scoreVal >= 85) return '#2ea043';
    if (scoreVal >= 75) return '#5cb85c';
    if (scoreVal >= 65) return '#d29922';
    if (scoreVal >= 50) return '#ffa500';
    return '#da3633';
  };

  const getEmoji = (scoreVal: number): string => {
    if (scoreVal >= 85) return '🟢';
    if (scoreVal >= 75) return '🟢';
    if (scoreVal >= 65) return '🟡';
    if (scoreVal >= 50) return '🟡';
    return '🔴';
  };

  const color = getColor(scoreValue);
  const grade = getGrade(scoreValue);

  return (
    <div
      style={{
        background: 'linear-gradient(135deg, #161b22 0%, #1f2428 100%)',
        border: `2px solid ${color}`,
        borderRadius: '16px',
        padding: '32px',
        textAlign: 'center',
        boxShadow: `0 0 30px ${color}33`,
        marginBottom: '32px',
        animation: 'fadeIn 0.6s ease-out',
      }}
    >
      <div
        style={{
          fontSize: '48px',
          fontWeight: 'bold',
          color: color,
          fontFamily: "'JetBrains Mono', monospace",
          marginBottom: '8px',
        }}
      >
        {getEmoji(scoreValue)} {scoreValue}/100
      </div>
      <div
        style={{
          fontSize: '32px',
          fontWeight: 'bold',
          color: 'var(--text-primary)',
          marginBottom: '4px',
        }}
      >
        Grade: {grade}
      </div>
      <div
        style={{
          fontSize: '14px',
          color: 'var(--text-secondary)',
        }}
      >
        {scoreValue >= 85
          ? '✓ Exceptional Obfuscation'
          : scoreValue >= 75
          ? '✓ Excellent Obfuscation'
          : scoreValue >= 65
          ? '⚠ Good Obfuscation'
          : scoreValue >= 50
          ? '⚠ Moderate Obfuscation'
          : '• Limited Obfuscation'}
      </div>
    </div>
  );
};

/**
 * Control Flow Comparison Bar Chart
 */
const ControlFlowChart: React.FC<{ metrics: ControlFlowMetricsData }> = ({
  metrics,
}) => {
  const data = [
    {
      name: 'Basic Blocks',
      Baseline: metrics.baseline.basic_blocks,
      Obfuscated: metrics.obfuscated.basic_blocks,
    },
    {
      name: 'CFG Edges',
      Baseline: metrics.baseline.cfg_edges,
      Obfuscated: metrics.obfuscated.cfg_edges,
    },
    {
      name: 'Cyclomatic Complexity',
      Baseline: metrics.baseline.cyclomatic_complexity,
      Obfuscated: metrics.obfuscated.cyclomatic_complexity,
    },
  ];

  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px' }}>
        📊 Control Flow Analysis
      </h3>
      <div
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '16px',
        }}
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
            <XAxis dataKey="name" stroke="var(--text-secondary)" />
            <YAxis stroke="var(--text-secondary)" />
            <Tooltip
              contentStyle={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
              }}
              labelStyle={{ color: 'var(--text-primary)' }}
            />
            <Legend />
            <Bar dataKey="Baseline" fill={COLORS.primary} />
            <Bar dataKey="Obfuscated" fill={COLORS.success} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div
        style={{
          marginTop: '16px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: '12px',
        }}
      >
        <MetricCard
          label="Complexity Increase"
          value={`${Math.round(metrics.comparison.complexity_increase_percent)}%`}
          color={COLORS.warning}
        />
        <MetricCard
          label="Blocks Added"
          value={metrics.comparison.basic_blocks_added}
          color={COLORS.orange}
        />
        <MetricCard
          label="Edges Added"
          value={metrics.comparison.cfg_edges_added}
          color={COLORS.orange}
        />
      </div>
    </div>
  );
};

/**
 * Instruction Distribution Pie Chart
 */
const InstructionChart: React.FC<{ metrics: InstructionMetricsData }> = ({
  metrics,
}) => {
  const baseline_data = Object.entries(
    metrics.baseline.instruction_distribution
  ).map(([name, value]) => ({
    name: name.toUpperCase(),
    value,
    fill: INSTRUCTION_COLORS[name] || COLORS.primary,
  }));

  const obfuscated_data = Object.entries(
    metrics.obfuscated.instruction_distribution
  ).map(([name, value]) => ({
    name: name.toUpperCase(),
    value,
    fill: INSTRUCTION_COLORS[name] || COLORS.primary,
  }));

  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px' }}>
        🔧 Instruction-Level Metrics
      </h3>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '16px',
          marginBottom: '16px',
        }}
      >
        <div
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '16px',
          }}
        >
          <h4
            style={{
              textAlign: 'center',
              color: 'var(--text-primary)',
              marginBottom: '12px',
            }}
          >
            Baseline Distribution
          </h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={baseline_data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {baseline_data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div
          style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '16px',
          }}
        >
          <h4
            style={{
              textAlign: 'center',
              color: 'var(--text-primary)',
              marginBottom: '12px',
            }}
          >
            Obfuscated Distribution
          </h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={obfuscated_data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {obfuscated_data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: '12px',
        }}
      >
        <MetricCard
          label="Baseline Instructions"
          value={metrics.baseline.total_instructions}
          color={COLORS.primary}
        />
        <MetricCard
          label="Obfuscated Instructions"
          value={metrics.obfuscated.total_instructions}
          color={COLORS.success}
        />
        <MetricCard
          label="Instruction Growth"
          value={`${Math.round(metrics.comparison.instruction_growth_percent)}%`}
          color={COLORS.warning}
        />
      </div>
    </div>
  );
};

/**
 * Metric Card - Small informational card
 */
interface MetricCardProps {
  label: string;
  value: string | number;
  color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  color = COLORS.primary,
}) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: `1px solid ${color}44`,
        borderRadius: '8px',
        padding: '12px',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          fontSize: '12px',
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          marginBottom: '8px',
          fontWeight: '600',
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: '24px',
          fontWeight: 'bold',
          color: color,
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        {value}
      </div>
    </div>
  );
};

/**
 * Main MetricsDashboard Component
 */
/**
 * Phoronix Benchmarking Metrics Card
 */
const PhoronixMetricsCard: React.FC<{ metrics?: PhoronixMetrics }> = ({ metrics }) => {
  if (!metrics) return null;

  const instrDelta = metrics.instruction_count_delta;
  const instrPercent = metrics.instruction_count_increase_percent;
  const perfOverhead = metrics.performance_overhead_percent;

  if (instrDelta === undefined && perfOverhead === undefined) return null;

  return (
    <div
      style={{
        background: 'linear-gradient(135deg, #1a1f26 0%, #202833 100%)',
        border: '2px solid #3fb950',
        borderRadius: '12px',
        padding: '24px',
        marginTop: '24px',
        boxShadow: '0 0 20px rgba(63, 185, 80, 0.1)',
      }}
    >
      <h3
        style={{
          margin: '0 0 16px 0',
          color: '#3fb950',
          fontSize: '16px',
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        📊 Obfuscation Impact Metrics
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
        }}
      >
        {instrDelta !== undefined && (
          <div
            style={{
              background: 'rgba(63, 185, 80, 0.1)',
              border: '1px solid #3fb950',
              borderRadius: '8px',
              padding: '16px',
              textAlign: 'center',
            }}
          >
            <div style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
              Code Expansion
            </div>
            <div
              style={{
                color: '#3fb950',
                fontSize: '24px',
                fontWeight: 'bold',
                fontFamily: "'JetBrains Mono', monospace",
                marginBottom: '4px',
              }}
            >
              +{instrDelta}
            </div>
            <div style={{ color: '#8b949e', fontSize: '11px' }}>
              (+{instrPercent}% instructions)
            </div>
          </div>
        )}

        {perfOverhead !== undefined && perfOverhead !== null && (
          <div
            style={{
              background: 'rgba(88, 166, 255, 0.1)',
              border: '1px solid #58a6ff',
              borderRadius: '8px',
              padding: '16px',
              textAlign: 'center',
            }}
          >
            <div style={{ color: '#8b949e', fontSize: '12px', marginBottom: '8px' }}>
              Performance Overhead
            </div>
            <div
              style={{
                color: '#58a6ff',
                fontSize: '24px',
                fontWeight: 'bold',
                fontFamily: "'JetBrains Mono', monospace",
                marginBottom: '4px',
              }}
            >
              +{perfOverhead.toFixed(1)}%
            </div>
            <div style={{ color: '#8b949e', fontSize: '11px' }}>
              Runtime slowdown
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          marginTop: '16px',
          padding: '12px',
          background: 'rgba(139, 148, 158, 0.1)',
          borderRadius: '6px',
          fontSize: '12px',
          color: '#8b949e',
          lineHeight: '1.6',
        }}
      >
        <strong>Note:</strong> Instruction count reflects code expansion from obfuscation passes.
        Performance overhead is based on runtime measurements if available.
      </div>
    </div>
  );
};

export const MetricsDashboard: React.FC<Props> = ({ report }) => {
  const hasControlFlow = report.control_flow_metrics &&
    Object.keys(report.control_flow_metrics).length > 0;
  const hasInstructions = report.instruction_metrics &&
    Object.keys(report.instruction_metrics).length > 0;
  const hasPhoronix = report.phoronix?.key_metrics;

  return (
    <div
      style={{
        animation: 'fadeIn 0.8s ease-out',
      }}
    >
      {/* ✅ NEW: Platform Metadata Indicator (for Windows score fix transparency) */}
      {report.metadata && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '20px',
            padding: '12px 16px',
            backgroundColor: '#f6f8fa',
            borderRadius: '8px',
            border: '1px solid #d0d7de',
            fontSize: '14px',
            fontWeight: '500',
          }}
        >
          <span>📊</span>
          <span>
            <strong>Platform:</strong>{' '}
            {(report.metadata.platform || 'unknown').toUpperCase()}
          </span>
          {report.metadata.binary_format && (
            <span
              style={{
                display: 'inline-block',
                backgroundColor: '#1f6feb',
                color: 'white',
                padding: '2px 8px',
                borderRadius: '4px',
                fontSize: '12px',
                marginLeft: '4px',
              }}
            >
              {report.metadata.binary_format}
            </span>
          )}
          {report.metadata.metric_extraction_method && (
            <span
              style={{
                marginLeft: 'auto',
                fontSize: '12px',
                color: '#666',
              }}
            >
              Metrics: {report.metadata.metric_extraction_method}
            </span>
          )}
        </div>
      )}

      {/* Protection Score */}
      <ProtectionScoreCard score={report.overall_protection_index || report.obfuscation_score} />

      {/* Control Flow Analysis */}
      {hasControlFlow && (
        <ControlFlowChart metrics={report.control_flow_metrics!} />
      )}

      {/* Instruction Metrics */}
      {hasInstructions && (
        <InstructionChart metrics={report.instruction_metrics!} />
      )}

      {/* Phoronix Benchmarking Metrics */}
      {hasPhoronix && report.phoronix && (
        <PhoronixMetricsCard metrics={report.phoronix.key_metrics} />
      )}

      {/* Empty state */}
      {!hasControlFlow && !hasInstructions && !hasPhoronix && (
        <div
          style={{
            textAlign: 'center',
            padding: '32px',
            color: 'var(--text-secondary)',
            fontSize: '14px',
          }}
        >
          📊 Advanced metrics not available for this report. Enable
          ir_metrics_enabled in configuration to see detailed control flow and
          instruction analysis.
        </div>
      )}
    </div>
  );
};

export default MetricsDashboard;
