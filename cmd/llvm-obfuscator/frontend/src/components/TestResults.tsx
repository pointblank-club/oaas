import { useEffect, useState } from 'react';

interface TestMetrics {
  functional_correctness: {
    same_behavior: boolean | null;
    passed: number;
    test_count: number;
  };
  strings: {
    baseline_strings: number;
    obf_strings: number;
    reduction_percent: number;
  };
  binary_properties: {
    baseline_size_bytes: number;
    obf_size_bytes: number;
    size_increase_percent: number;
    baseline_entropy: number;
    obf_entropy: number;
    entropy_increase: number;
  };
  symbols: {
    baseline_symbol_count: number;
    obf_symbol_count: number;
    symbols_reduced: boolean;
  };
  performance: {
    baseline_ms: number | null;
    obf_ms: number | null;
    overhead_percent: number | null;
    acceptable: boolean | null;
    status: string;
  };
  cfg_metrics: {
    comparison: {
      indirect_jumps_ratio: number;
      basic_blocks_ratio: number;
      control_flow_complexity_increase: number;
    };
  };
}

interface TestReportData {
  metadata: {
    timestamp: string;
    program: string;
    baseline: string;
    obfuscated: string;
    metrics_reliability: string;
    functional_correctness_passed: boolean | null;
  };
  test_results: TestMetrics;
  reliability_status: {
    level: string;
    warning: string;
  };
}

interface TestResultsProps {
  jobId: string;
  onError?: (error: string) => void;
}

export function TestResults({ jobId, onError }: TestResultsProps) {
  const [testData, setTestData] = useState<TestReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTestResults = async () => {
      try {
        setLoading(true);
        // Fetch test results from the report endpoint
        const response = await fetch(
          `/api/report/${jobId}?fmt=json`,
          {
            headers: {
              'Accept': 'application/json'
            }
          }
        );

        if (!response.ok) {
          if (response.status === 404) {
            setError('Test results not yet available. Run obfuscation first.');
            onError?.('Test results not available');
            return;
          }
          throw new Error(`Failed to fetch results: ${response.statusText}`);
        }

        const data = await response.json();

        // Check if data includes test results
        if (data.test_results && data.reliability_status) {
          setTestData(data);
        } else {
          setError('Test results not found in report. This may be an older obfuscation without test suite integration.');
          onError?.('Test results not in report');
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
        onError?.(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchTestResults();
  }, [jobId, onError]);

  if (loading) {
    return (
      <section className="section">
        <h2 className="section-title">[6] TEST SUITE RESULTS</h2>
        <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <p>Loading test results...</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="section">
        <h2 className="section-title">[6] TEST SUITE RESULTS</h2>
        <div style={{
          backgroundColor: '#f8d7da',
          border: '1px solid #f5c6cb',
          borderRadius: '4px',
          padding: '12px',
          color: '#721c24'
        }}>
          <strong>⚠️ Test Results Unavailable</strong>
          <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>{error}</p>
        </div>
      </section>
    );
  }

  if (!testData) {
    return null;
  }

  const { reliability_status, test_results, metadata } = testData;
  const isReliable = reliability_status.level === 'RELIABLE';
  const isFailed = reliability_status.level === 'FAILED';
  const isUncertain = reliability_status.level === 'UNCERTAIN';

  // Determine warning color based on reliability
  let warningBgColor = '#d4edda'; // success green
  let warningBorderColor = '#c3e6cb';
  let warningTextColor = '#155724';
  let warningIcon = '✅';

  if (isFailed) {
    warningBgColor = '#f8d7da';
    warningBorderColor = '#f5c6cb';
    warningTextColor = '#721c24';
    warningIcon = '❌';
  } else if (isUncertain) {
    warningBgColor = '#fff3cd';
    warningBorderColor = '#ffc107';
    warningTextColor = '#856404';
    warningIcon = '⚠️';
  }

  return (
    <section className="section report-section">
      <h2 className="section-title">[6] TEST SUITE RESULTS</h2>

      {/* Reliability Status Banner */}
      <div style={{
        backgroundColor: warningBgColor,
        border: `1px solid ${warningBorderColor}`,
        borderRadius: '4px',
        padding: '12px',
        marginBottom: '16px',
        color: warningTextColor
      }}>
        <strong>{warningIcon} {reliability_status.warning}</strong>
        {isFailed && (
          <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
            The obfuscated binary does not behave the same as the baseline. Metrics below may not represent true obfuscation effectiveness.
          </p>
        )}
        {isUncertain && (
          <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
            The functional correctness test was inconclusive. Use results with caution.
          </p>
        )}
      </div>

      <div className="report-grid">
        {/* Metadata */}
        <div className="report-block">
          <h3>TEST METADATA</h3>
          <div className="report-item">Timestamp: {new Date(metadata.timestamp).toLocaleString()}</div>
          <div className="report-item">Program: {metadata.program}</div>
          <div className="report-item">
            Reliability: <strong style={{ color: isReliable ? '#28a745' : isFailed ? '#dc3545' : '#ffc107' }}>
              {metadata.metrics_reliability}
            </strong>
          </div>
          <div className="report-item">
            Functional Test: <strong style={{ color: metadata.functional_correctness_passed ? '#28a745' : '#dc3545' }}>
              {metadata.functional_correctness_passed === true ? 'PASSED ✓' : metadata.functional_correctness_passed === false ? 'FAILED ✗' : 'INCONCLUSIVE'}
            </strong>
          </div>
        </div>

        {/* Functional Correctness */}
        <div className="report-block">
          <h3>FUNCTIONAL CORRECTNESS</h3>
          <div className="report-item">
            Same Behavior: <strong style={{ color: test_results.functional_correctness.same_behavior ? '#28a745' : '#dc3545' }}>
              {test_results.functional_correctness.same_behavior === true ? 'YES ✓' : test_results.functional_correctness.same_behavior === false ? 'NO ✗' : 'UNKNOWN'}
            </strong>
          </div>
          <div className="report-item">Tests Passed: {test_results.functional_correctness.passed}/{test_results.functional_correctness.test_count}</div>
        </div>

        {/* String Analysis */}
        <div className="report-block">
          <h3>STRING OBFUSCATION</h3>
          <div className="report-item">Baseline Strings: {test_results.strings.baseline_strings}</div>
          <div className="report-item">Obfuscated Strings: {test_results.strings.obf_strings}</div>
          <div className="report-item">
            Reduction: <strong style={{ color: test_results.strings.reduction_percent > 0 ? '#28a745' : '#dc3545' }}>
              {test_results.strings.reduction_percent > 0 ? '−' : '+'}{Math.abs(test_results.strings.reduction_percent).toFixed(1)}%
            </strong>
          </div>
        </div>

        {/* Binary Properties */}
        <div className="report-block">
          <h3>BINARY PROPERTIES</h3>
          <div className="report-item">Baseline Size: {(test_results.binary_properties.baseline_size_bytes / 1024).toFixed(1)} KB</div>
          <div className="report-item">Obfuscated Size: {(test_results.binary_properties.obf_size_bytes / 1024).toFixed(1)} KB</div>
          <div className="report-item">
            Size Change: <strong style={{ color: test_results.binary_properties.size_increase_percent < 100 ? '#28a745' : '#ffc107' }}>
              {test_results.binary_properties.size_increase_percent > 0 ? '+' : ''}{test_results.binary_properties.size_increase_percent.toFixed(1)}%
            </strong>
          </div>
          <div className="report-item">Baseline Entropy: {test_results.binary_properties.baseline_entropy.toFixed(4)}</div>
          <div className="report-item">Obfuscated Entropy: {test_results.binary_properties.obf_entropy.toFixed(4)}</div>
        </div>

        {/* Symbol Analysis */}
        <div className="report-block">
          <h3>SYMBOL ANALYSIS</h3>
          <div className="report-item">Baseline Symbols: {test_results.symbols.baseline_symbol_count}</div>
          <div className="report-item">Obfuscated Symbols: {test_results.symbols.obf_symbol_count}</div>
          <div className="report-item">
            Reduced: <strong style={{ color: test_results.symbols.symbols_reduced ? '#28a745' : '#dc3545' }}>
              {test_results.symbols.symbols_reduced ? 'YES ✓' : 'NO ✗'}
            </strong>
          </div>
        </div>

        {/* Performance */}
        <div className="report-block">
          <h3>PERFORMANCE ANALYSIS</h3>
          {test_results.performance.status === 'SKIPPED' ? (
            <>
              <div className="report-item" style={{ color: '#856404' }}>Status: <strong>SKIPPED</strong></div>
              <div className="report-item" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                Performance testing was skipped because functional correctness test failed.
              </div>
            </>
          ) : test_results.performance.status === 'SUCCESS' ? (
            <>
              <div className="report-item">Baseline Time: {test_results.performance.baseline_ms?.toFixed(2)} ms</div>
              <div className="report-item">Obfuscated Time: {test_results.performance.obf_ms?.toFixed(2)} ms</div>
              <div className="report-item">
                Overhead: <strong style={{ color: (test_results.performance.overhead_percent ?? 0) < 100 ? '#28a745' : '#ffc107' }}>
                  {test_results.performance.overhead_percent != null ? `+${test_results.performance.overhead_percent.toFixed(1)}%` : 'N/A'}
                </strong>
              </div>
            </>
          ) : (
            <>
              <div className="report-item" style={{ color: '#dc3545' }}>Status: <strong>{test_results.performance.status}</strong></div>
              <div className="report-item" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                Could not measure performance. Binary may not be executable.
              </div>
            </>
          )}
        </div>

        {/* Control Flow Metrics */}
        <div className="report-block">
          <h3>CONTROL FLOW METRICS</h3>
          <div className="report-item">
            Indirect Jumps Ratio: <strong>{test_results.cfg_metrics.comparison.indirect_jumps_ratio.toFixed(2)}x</strong>
          </div>
          <div className="report-item">
            Basic Blocks Ratio: <strong>{test_results.cfg_metrics.comparison.basic_blocks_ratio.toFixed(2)}x</strong>
          </div>
          <div className="report-item">
            Complexity Increase: <strong style={{ color: test_results.cfg_metrics.comparison.control_flow_complexity_increase > 1 ? '#28a745' : '#dc3545' }}>
              {test_results.cfg_metrics.comparison.control_flow_complexity_increase.toFixed(2)}x
            </strong>
          </div>
        </div>
      </div>

      {/* Summary */}
      {isReliable && (
        <div style={{
          backgroundColor: '#d4edda',
          border: '1px solid #c3e6cb',
          borderRadius: '4px',
          padding: '12px',
          marginTop: '16px',
          color: '#155724'
        }}>
          <strong>✅ Summary</strong>
          <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
            The obfuscated binary maintains functional correctness with the baseline. All metrics shown above are reliable and represent true obfuscation effectiveness.
          </p>
        </div>
      )}
    </section>
  );
}
