/**
 * Binary Metrics Dashboard Component
 * Displays comprehensive binary-to-binary obfuscation metrics
 * Adapted from MetricsDashboard for binary obfuscation context
 */

import React, { useMemo } from 'react';
import {
  BarChart, Bar, LineChart, Line, ResponsiveContainer,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts';

interface BinaryMetrics {
  input_size: number;
  output_size: number;
  size_diff_percent: number;
  llvm_inst_before: number;
  llvm_inst_after: number;
  inst_diff_percent: number;
  cfg_complexity_before: number;
  cfg_complexity_after: number;
  cfg_diff_percent: number;
}

interface Props {
  metrics: BinaryMetrics;
}

// Color constants for consistent theming
const COLORS = {
  primary: '#1f6feb',
  success: '#2ea043',
  warning: '#d29922',
  danger: '#da3633',
  orange: '#fb8500',
  cyan: '#00d4ff',
};

/**
 * Metric Card - Small informational card
 */
interface MetricCardProps {
  label: string;
  value: string | number;
  color?: string;
  unit?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  color = COLORS.primary,
  unit = '',
}) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: `1px solid ${color}44`,
        borderRadius: '8px',
        padding: '16px',
        textAlign: 'center',
        transition: 'all 0.2s ease',
      }}
    >
      <div
        style={{
          fontSize: '12px',
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          marginBottom: '8px',
          fontWeight: '600',
          letterSpacing: '0.5px',
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: '28px',
          fontWeight: 'bold',
          color: color,
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        {value}{unit}
      </div>
    </div>
  );
};

/**
 * Format bytes to human-readable size
 */
const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Binary Size Impact Section
 */
const BinarySizeImpact: React.FC<{ metrics: BinaryMetrics }> = ({ metrics }) => {
  const data = [
    {
      name: 'Binary Size',
      'Input Size': metrics.input_size,
      'Output Size': metrics.output_size,
    },
  ];

  const sizeChangeColor = metrics.size_diff_percent > 0 ? COLORS.warning : COLORS.success;
  const sizeChangeSymbol = metrics.size_diff_percent > 0 ? '+' : '';

  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px' }}>
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
          marginBottom: '20px',
        }}
      >
        <MetricCard
          label="Input Binary Size (PE)"
          value={formatBytes(metrics.input_size)}
          color={COLORS.primary}
        />
        <MetricCard
          label="Output Binary Size (PE)"
          value={formatBytes(metrics.output_size)}
          color={COLORS.primary}
        />
        <MetricCard
          label="Size Change"
          value={`${sizeChangeSymbol}${metrics.size_diff_percent.toFixed(2)}`}
          color={sizeChangeColor}
          unit="%"
        />
      </div>

      {/* Bar Chart Comparison */}
      <div
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '16px',
        }}
      >
        <ResponsiveContainer width="100%" height={250}>
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
              formatter={(value: any) => formatBytes(value)}
            />
            <Legend />
            <Bar dataKey="Input Size" fill={COLORS.primary} />
            <Bar dataKey="Output Size" fill={COLORS.orange} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

/**
 * LLVM Instruction Count Impact Section
 */
const InstructionCountImpact: React.FC<{ metrics: BinaryMetrics }> = ({ metrics }) => {
  const data = [
    {
      name: 'Instruction Count',
      'Before Obfuscation': metrics.llvm_inst_before,
      'After Obfuscation': metrics.llvm_inst_after,
    },
  ];

  const instChangeColor = metrics.inst_diff_percent > 0 ? COLORS.warning : COLORS.success;
  const instChangeSymbol = metrics.inst_diff_percent > 0 ? '+' : '';

  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px' }}>
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
          marginBottom: '20px',
        }}
      >
        <MetricCard
          label="Instructions Before"
          value={metrics.llvm_inst_before.toLocaleString()}
          color={COLORS.primary}
        />
        <MetricCard
          label="Instructions After"
          value={metrics.llvm_inst_after.toLocaleString()}
          color={COLORS.orange}
        />
        <MetricCard
          label="Instruction Change"
          value={`${instChangeSymbol}${metrics.inst_diff_percent.toFixed(2)}`}
          color={instChangeColor}
          unit="%"
        />
      </div>

      {/* Line Chart Comparison */}
      <div
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '16px',
        }}
      >
        <ResponsiveContainer width="100%" height={250}>
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
              formatter={(value: any) => value.toLocaleString()}
            />
            <Legend />
            <Bar dataKey="Before Obfuscation" fill={COLORS.primary} />
            <Bar dataKey="After Obfuscation" fill={COLORS.warning} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

/**
 * Control Flow Complexity Impact Section
 */
const ControlFlowComplexityImpact: React.FC<{ metrics: BinaryMetrics }> = ({ metrics }) => {
  const data = [
    {
      name: 'CFG Complexity',
      'Before Obfuscation': metrics.cfg_complexity_before,
      'After Obfuscation': metrics.cfg_complexity_after,
    },
  ];

  const cfgChangeColor = metrics.cfg_diff_percent > 0 ? COLORS.warning : COLORS.success;
  const cfgChangeSymbol = metrics.cfg_diff_percent > 0 ? '+' : '';

  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px' }}>
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
          marginBottom: '20px',
        }}
      >
        <MetricCard
          label="CFG Complexity Before"
          value={metrics.cfg_complexity_before.toFixed(2)}
          color={COLORS.primary}
        />
        <MetricCard
          label="CFG Complexity After"
          value={metrics.cfg_complexity_after.toFixed(2)}
          color={COLORS.orange}
        />
        <MetricCard
          label="Complexity Change"
          value={`${cfgChangeSymbol}${metrics.cfg_diff_percent.toFixed(2)}`}
          color={cfgChangeColor}
          unit="%"
        />
      </div>

      {/* Line Chart for Complexity Trend */}
      <div
        style={{
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '16px',
        }}
      >
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
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
            <Line
              type="monotone"
              dataKey="Before Obfuscation"
              stroke={COLORS.primary}
              strokeWidth={2}
              dot={{ fill: COLORS.primary, r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="After Obfuscation"
              stroke={COLORS.warning}
              strokeWidth={2}
              dot={{ fill: COLORS.warning, r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

/**
 * Summary Statistics Section
 */
const SummaryStatistics: React.FC<{ metrics: BinaryMetrics }> = ({ metrics }) => {
  const totalChange = (
    ((metrics.size_diff_percent + metrics.inst_diff_percent + metrics.cfg_diff_percent) / 3)
  ).toFixed(2);

  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: '16px' }}>
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px',
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '20px',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 'bold' }}>
            Average Impact
          </div>
          <div style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.primary, fontFamily: "'JetBrains Mono', monospace" }}>
            {totalChange}%
          </div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 'bold' }}>
            Size Increase
          </div>
          <div style={{ fontSize: '28px', fontWeight: 'bold', color: metrics.size_diff_percent > 0 ? COLORS.danger : COLORS.success, fontFamily: "'JetBrains Mono', monospace" }}>
            {metrics.size_diff_percent > 0 ? '+' : ''}{metrics.size_diff_percent.toFixed(2)}%
          </div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 'bold' }}>
            Instruction Overhead
          </div>
          <div style={{ fontSize: '28px', fontWeight: 'bold', color: metrics.inst_diff_percent > 0 ? COLORS.danger : COLORS.success, fontFamily: "'JetBrains Mono', monospace" }}>
            {metrics.inst_diff_percent > 0 ? '+' : ''}{metrics.inst_diff_percent.toFixed(2)}%
          </div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 'bold' }}>
            Complexity Growth
          </div>
          <div style={{ fontSize: '28px', fontWeight: 'bold', color: metrics.cfg_diff_percent > 0 ? COLORS.warning : COLORS.success, fontFamily: "'JetBrains Mono', monospace" }}>
            {metrics.cfg_diff_percent > 0 ? '+' : ''}{metrics.cfg_diff_percent.toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Main Binary Metrics Dashboard Component
 */
const BinaryMetricsDashboard: React.FC<Props> = ({ metrics }) => {
  if (!metrics) {
    return (
      <div
        style={{
          textAlign: 'center',
          padding: '32px',
          color: 'var(--text-secondary)',
          fontSize: '14px',
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)',
        }}
      >
        ⚠ Metrics unavailable for this job. Download logs to investigate.
      </div>
    );
  }

  return (
    <div
      style={{
        animation: 'fadeIn 0.8s ease-out',
      }}
    >
      <div style={{ marginBottom: '24px' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', margin: 0 }}>
          Detailed analysis of binary transformations across size, instruction count, and control flow complexity
        </p>
      </div>

      {/* Summary Statistics */}
      <SummaryStatistics metrics={metrics} />

      {/* Binary Size Impact */}
      <BinarySizeImpact metrics={metrics} />

      {/* LLVM Instruction Count Impact */}
      <InstructionCountImpact metrics={metrics} />

      {/* Control Flow Complexity Impact */}
      <ControlFlowComplexityImpact metrics={metrics} />
    </div>
  );
};

export default BinaryMetricsDashboard;
