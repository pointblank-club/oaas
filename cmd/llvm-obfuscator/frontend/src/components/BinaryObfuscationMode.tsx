import { useState, useRef, useEffect } from 'react';
import BinaryMetricsDashboard from './BinaryMetricsDashboard';
import BinaryCFGVisualizer from './BinaryCFGVisualizer';

type PipelineStage = 'CFG' | 'LIFTING' | 'IR22' | 'OLLVM' | 'FINALIZING' | 'COMPLETED' | 'ERROR';
type JobStatus = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'ERROR' | 'CANCELLED';

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

interface JobState {
  jobId: string;
  status: JobStatus;
  stage: PipelineStage;
  progress: number;
  logs: string;
  artifacts: string[];
  error?: string;
  metrics?: BinaryMetrics;
}

interface FileValidationError {
  type: 'size' | 'extension' | 'signature' | 'filename' | 'empty';
  message: string;
}

interface BinaryObfuscationModeProps {
  onJobStart?: (jobId: string) => void;
  obfuscationMode?: 'source' | 'binary';
  onModeChange?: (mode: 'source' | 'binary') => void;
}

const PIPELINE_STEPS = [
  { id: 1, name: 'Ghidra CFG Export', stage: 'CFG', progress: 20 },
  { id: 2, name: 'McSema LLVM Lift', stage: 'LIFTING', progress: 45 },
  { id: 3, name: 'LLVM 22 IR Upgrade', stage: 'IR22', progress: 65 },
  { id: 4, name: 'OLLVM Obfuscation', stage: 'OLLVM', progress: 80 },
  { id: 5, name: 'Final Binary Generation', stage: 'FINALIZING', progress: 90 },
];

const DISABLED_PASSES = ['flattening', 'bogus_control_flow', 'split', 'linear_mba'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
const MIN_FILE_SIZE = 3 * 1024; // 3 KB

/**
 * Validate binary file before upload
 */
const validateBinaryFile = (file: File): FileValidationError | null => {
  // Check extension
  if (!file.name.toLowerCase().endsWith('.exe')) {
    return {
      type: 'extension',
      message: 'Only .exe files are supported'
    };
  }

  // Check file size
  if (file.size === 0) {
    return {
      type: 'empty',
      message: 'File is empty'
    };
  }

  if (file.size < MIN_FILE_SIZE) {
    return {
      type: 'size',
      message: `File suspiciously small (${(file.size / 1024).toFixed(2)} KB). Binary must be at least 3 KB.`
    };
  }

  if (file.size > MAX_FILE_SIZE) {
    return {
      type: 'size',
      message: `File exceeds maximum size. Max: 50 MB, got: ${(file.size / 1024 / 1024).toFixed(2)} MB`
    };
  }

  // Check filename for spaces/unicode
  if (/[\s\u0080-\uFFFF]/.test(file.name)) {
    return {
      type: 'filename',
      message: 'Filename contains spaces or special characters. Rename and try again.'
    };
  }

  return null;
};

/**
 * Check PE magic bytes (MZ header)
 */
const checkPESignature = async (file: File): Promise<boolean> => {
  try {
    const buffer = await file.slice(0, 2).arrayBuffer();
    const view = new Uint8Array(buffer);
    // MZ header: 0x4D 0x5A
    return view[0] === 0x4D && view[1] === 0x5A;
  } catch {
    return false;
  }
};

export function BinaryObfuscationMode({ onJobStart, onModeChange }: BinaryObfuscationModeProps) {
  const [binaryFile, setBinaryFile] = useState<File | null>(null);
  const [jobState, setJobState] = useState<JobState | null>(null);
  const [passes, setPasses] = useState({
    substitution: true,
    flattening: false,
    bogus_control_flow: false,
    split: false,
    linear_mba: false,
    string_encrypt: false,
    symbol_obfuscate: false,
    constant_obfuscate: false,
    crypto_hash: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [validationWarning, setValidationWarning] = useState<FileValidationError | null>(null);
  const [logsAutoScroll, setLogsAutoScroll] = useState(true);
  const [showLogsPanel, setShowLogsPanel] = useState(true);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showModeSwitchConfirmation, setShowModeSwitchConfirmation] = useState(false);
  const [cfgData, setCfgData] = useState<any>(null);
  const [cfgLoading, setCfgLoading] = useState(false);
  const [cfgError, setCfgError] = useState<string | null>(null);
  const [showCFGPanel, setShowCFGPanel] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom when new content arrives
  useEffect(() => {
    if (logsAutoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [jobState?.logs, logsAutoScroll]);

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const handleBinaryUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file
    const validationError = validateBinaryFile(file);
    if (validationError) {
      setError(validationError.message);
      setValidationWarning(validationError);
      setBinaryFile(null);
      return;
    }

    // Check PE signature
    const isPE = await checkPESignature(file);
    if (!isPE) {
      setError('File signature check failed. Expected Windows PE executable (MZ header).');
      setValidationWarning({
        type: 'signature',
        message: 'File does not appear to be a valid Windows PE binary'
      });
      setBinaryFile(null);
      return;
    }

    setBinaryFile(file);
    setError(null);
    setValidationWarning(null);
  };

  const handlePassToggle = (passName: keyof typeof passes) => {
    if (DISABLED_PASSES.includes(passName) || jobState) return;
    setPasses(prev => ({
      ...prev,
      [passName]: !prev[passName]
    }));
  };

  const pollJobStatus = async (jobId: string) => {
    try {
      const response = await fetch(`/api/binary_obfuscate/status/${jobId}`);
      if (!response.ok) return;

      const statusData = await response.json();

      let metrics: BinaryMetrics | undefined;

      // Fetch metrics on completion
      if (statusData.stage === 'COMPLETED' && statusData.available_artifacts?.includes('metrics.json')) {
        try {
          const metricsResponse = await fetch(`/api/binary_obfuscate/artifact/${jobId}/metrics.json`);
          if (metricsResponse.ok) {
            metrics = await metricsResponse.json();
          }
        } catch (err) {
          console.warn('Failed to fetch metrics:', err);
        }
      }

      setJobState(prev => prev ? {
        ...prev,
        status: statusData.status,
        stage: statusData.stage,
        progress: statusData.progress || 0,
        logs: statusData.logs || '',
        artifacts: statusData.available_artifacts || [],
        error: statusData.error,
        metrics: metrics || prev.metrics
      } : null);

      // Fetch CFG data when available (after Stage 1 completes)
      if (statusData.available_artifacts?.includes('input.cfg') && !cfgData && !cfgLoading) {
        setCfgLoading(true);
        try {
          const cfgResponse = await fetch(`/api/binary_obfuscate/artifact/${jobId}/input.cfg`);
          if (cfgResponse.ok) {
            const cfg = await cfgResponse.json();
            setCfgData(cfg);
            setCfgError(null);
          } else {
            setCfgError('Failed to load CFG');
          }
        } catch (err) {
          setCfgError('Failed to fetch CFG data');
          console.warn('Failed to fetch CFG:', err);
        } finally {
          setCfgLoading(false);
        }
      }

      // Stop polling on completion, error, or cancellation
      if (statusData.stage === 'COMPLETED' || statusData.stage === 'ERROR' || statusData.status === 'CANCELLED') {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
        setIsCancelling(false);
      }
    } catch (err) {
      console.error('Failed to poll job status:', err);
    }
  };

  const handleStartObfuscation = async () => {
    if (!binaryFile) {
      setError('Please upload a binary file');
      return;
    }

    setError(null);
    // Reset CFG state for new job
    setCfgData(null);
    setCfgError(null);
    setCfgLoading(false);

    try {
      const formData = new FormData();
      formData.append('file', binaryFile);
      formData.append('passes', JSON.stringify(passes));

      const response = await fetch('/api/binary_obfuscate', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      const jobId = data.job_id;

      // Initialize job state
      setJobState({
        jobId,
        status: 'QUEUED' as JobStatus,
        stage: 'CFG' as PipelineStage,
        progress: 0,
        logs: '',
        artifacts: []
      });

      if (onJobStart) {
        onJobStart(jobId);
      }

      // Start polling every 2 seconds
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }

      // Poll immediately
      await pollJobStatus(jobId);

      // Then set up interval
      pollIntervalRef.current = setInterval(() => {
        pollJobStatus(jobId);
      }, 2000);

      // Scroll to job section
      setTimeout(() => {
        containerRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMsg);
    }
  };

  const handleCancelJob = async () => {
    if (!jobState) return;

    setIsCancelling(true);

    try {
      const response = await fetch(`/api/binary_obfuscate/job/${jobState.jobId}/cancel`, {
        method: 'POST'
      });

      if (!response.ok) {
        // If cancel endpoint doesn't exist, just reset UI
        setJobState(null);
        setIsCancelling(false);
        return;
      }

      // Poll until cancelled
      const checkCancel = async () => {
        const statusResponse = await fetch(`/api/binary_obfuscate/status/${jobState.jobId}`);
        if (statusResponse.ok) {
          const status = await statusResponse.json();
          if (status.status === 'CANCELLED') {
            setJobState(null);
            setIsCancelling(false);
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
            }
          }
        }
      };

      const cancelCheckInterval = setInterval(checkCancel, 500);
      setTimeout(() => clearInterval(cancelCheckInterval), 10000); // Max 10s wait
    } catch (err) {
      console.error('Failed to cancel job:', err);
      setIsCancelling(false);
    }
  };

  const handleRetry = () => {
    setJobState(null);
    setError(null);
    handleStartObfuscation();
  };

  const handleModeSwitch = () => {
    if (jobState && jobState.stage !== 'COMPLETED' && jobState.stage !== 'ERROR') {
      setShowModeSwitchConfirmation(true);
      return;
    }
    if (onModeChange) {
      onModeChange('source');
    }
  };

  const confirmModeSwitch = () => {
    setShowModeSwitchConfirmation(false);
    handleCancelJob().then(() => {
      if (onModeChange) {
        onModeChange('source');
      }
    });
  };

  const getStepStatus = (stepStage: string): 'pending' | 'completed' | 'active' | 'failed' => {
    if (!jobState) return 'pending';

    const stageOrder = ['CFG', 'LIFTING', 'IR22', 'OLLVM', 'FINALIZING'];
    const currentIndex = stageOrder.indexOf(jobState.stage);
    const stepIndex = stageOrder.indexOf(stepStage);

    if (jobState.stage === 'ERROR') {
      return stepIndex === currentIndex ? 'failed' : stepIndex < currentIndex ? 'completed' : 'pending';
    }

    if (jobState.stage === 'COMPLETED') {
      return 'completed';
    }

    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'active';
    return 'pending';
  };

  const getProgressForStage = (stage: PipelineStage): number => {
    const step = PIPELINE_STEPS.find(s => s.stage === stage);
    return step?.progress || 0;
  };

  const getCurrentProgress = (): number => {
    if (!jobState) return 0;
    return getProgressForStage(jobState.stage);
  };

  const getLastLogLines = (count: number = 10): string[] => {
    if (!jobState?.logs) return [];
    return jobState.logs.split('\n').slice(-count).filter(l => l.trim());
  };

  const formatLogLine = (line: string): { timestamp: string; message: string } => {
    const match = line.match(/^\[([^\]]+)\]\s+(.*)$/);
    if (match) {
      return { timestamp: match[1], message: match[2] };
    }
    return { timestamp: '', message: line };
  };

  const isJobRunning = jobState ? (jobState.stage !== 'COMPLETED' && jobState.stage !== 'ERROR') : false;

  return (
    <div
      ref={containerRef}
      style={{
        padding: '20px',
        maxWidth: '1200px',
        margin: '0 auto'
      }}
    >
      {/* Error Banner */}
      {error && (
        <div style={{
          marginBottom: '20px',
          padding: '12px 16px',
          backgroundColor: 'var(--danger)',
          color: 'white',
          borderRadius: '4px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          animation: 'slideDown 0.3s ease-out'
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            ❌ {error}
          </span>
          <button
            onClick={() => setError(null)}
            style={{
              background: 'none',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              fontSize: '1.2em',
              padding: '0',
              width: '24px',
              height: '24px'
            }}>
            ✕
          </button>
        </div>
      )}

      {/* Mode Switch Confirmation Modal */}
      {showModeSwitchConfirmation && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          animation: 'fadeIn 0.2s ease-out'
        }}>
          <div style={{
            backgroundColor: 'var(--bg-secondary)',
            border: '2px solid var(--warning)',
            borderRadius: '8px',
            padding: '24px',
            maxWidth: '400px',
            animation: 'slideUp 0.3s ease-out'
          }}>
            <h3 style={{ marginTop: 0, color: 'var(--warning)' }}>⚠️ Cancel Running Job?</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '20px' }}>
              A binary obfuscation job is currently running. Switching modes will cancel it permanently.
            </p>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={confirmModeSwitch}
                style={{
                  flex: 1,
                  padding: '10px',
                  backgroundColor: 'var(--danger)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                Cancel Job & Switch
              </button>
              <button
                onClick={() => setShowModeSwitchConfirmation(false)}
                style={{
                  flex: 1,
                  padding: '10px',
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                Stay Here
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Job Running / Show Progress */}
      {jobState ? (
        <>
          {/* Pipeline Steps Visualization */}
          <section style={{
            marginBottom: '30px',
            padding: '20px',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            animation: 'fadeIn 0.5s ease-out'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ color: 'var(--text-primary)', margin: 0 }}>Pipeline Progress</h3>
              {isJobRunning && (
                <button
                  onClick={handleCancelJob}
                  disabled={isCancelling}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: 'var(--danger)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: isCancelling ? 'not-allowed' : 'pointer',
                    opacity: isCancelling ? 0.6 : 1
                  }}
                >
                  {isCancelling ? '⟳ Cancelling...' : '✕ Cancel Job'}
                </button>
              )}
            </div>

            {/* Desktop Pipeline Visualization */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '30px',
              minHeight: '80px',
              position: 'relative'
            }}>
              {PIPELINE_STEPS.map((step, idx) => {
                const stepStatus = getStepStatus(step.stage);
                const statusColor = stepStatus === 'completed' ? 'var(--success)' :
                                   stepStatus === 'failed' ? 'var(--danger)' :
                                   stepStatus === 'active' ? 'var(--accent)' :
                                   'var(--border-color)';

                const getAnimation = () => {
                  if (stepStatus === 'active') return 'pulse 1s ease-in-out infinite';
                  if (stepStatus === 'completed') return 'popIn 0.4s ease-out';
                  if (stepStatus === 'failed') return 'shake 0.4s ease-out';
                  return 'none';
                };

                return (
                  <div key={step.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, position: 'relative' }}>
                    <div style={{
                      width: '48px',
                      height: '48px',
                      borderRadius: '50%',
                      backgroundColor: statusColor,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontWeight: 'bold',
                      fontSize: '1.2em',
                      marginBottom: '12px',
                      transition: 'all 0.3s ease',
                      animation: getAnimation(),
                      boxShadow: stepStatus === 'active' ? `0 0 20px ${statusColor}` : 'none'
                    }}>
                      {stepStatus === 'completed' ? '✓' :
                       stepStatus === 'failed' ? '✕' :
                       stepStatus === 'active' ? '⟳' :
                       step.id}
                    </div>
                    <span style={{
                      fontSize: '0.85em',
                      textAlign: 'center',
                      color: stepStatus === 'active' ? 'var(--accent)' : 'var(--text-secondary)',
                      fontWeight: stepStatus === 'active' ? 'bold' : 'normal',
                      maxWidth: '100px'
                    }}>
                      {step.name}
                    </span>
                  </div>
                );
              })}

              {/* Connecting Lines */}
              <style>{`
                @keyframes pulse {
                  0%, 100% { transform: scale(1); }
                  50% { transform: scale(1.1); }
                }
                @keyframes popIn {
                  0% { transform: scale(0); }
                  100% { transform: scale(1); }
                }
                @keyframes shake {
                  0%, 100% { transform: translateX(0); }
                  25% { transform: translateX(-5px); }
                  75% { transform: translateX(5px); }
                }
                @keyframes fadeIn {
                  from { opacity: 0; }
                  to { opacity: 1; }
                }
                @keyframes slideDown {
                  from { transform: translateY(-20px); opacity: 0; }
                  to { transform: translateY(0); opacity: 1; }
                }
                @keyframes slideUp {
                  from { transform: translateY(20px); opacity: 0; }
                  to { transform: translateY(0); opacity: 1; }
                }
              `}</style>
            </div>

            {/* Progress Bar */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '8px',
                fontSize: '0.9em'
              }}>
                <span style={{ fontWeight: 'bold' }}>Overall Progress</span>
                <span style={{ color: 'var(--accent)' }}>{getCurrentProgress()}%</span>
              </div>
              <div style={{
                width: '100%',
                height: '12px',
                backgroundColor: 'var(--bg-tertiary)',
                borderRadius: '6px',
                overflow: 'hidden',
                border: '1px solid var(--border-color)'
              }}>
                <div style={{
                  height: '100%',
                  width: `${getCurrentProgress()}%`,
                  backgroundColor: jobState.stage === 'ERROR' ? 'var(--danger)' : 'var(--accent)',
                  transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                  borderRadius: '5px'
                }} />
              </div>
            </div>
          </section>

          {/* Logs Panel */}
          <section style={{
            marginBottom: '30px',
            padding: '0',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            overflow: 'hidden',
            animation: 'fadeIn 0.5s ease-out'
          }}>
            <div
              onClick={() => setShowLogsPanel(!showLogsPanel)}
              style={{
                padding: '16px',
                backgroundColor: 'var(--bg-secondary)',
                borderBottom: showLogsPanel ? '1px solid var(--border-color)' : 'none',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                userSelect: 'none'
              }}
            >
              <h3 style={{ margin: 0, fontSize: '1.1em' }}>📋 Pipeline Logs</h3>
              <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                {showLogsPanel && (
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', margin: 0 }}>
                    <input
                      type="checkbox"
                      checked={logsAutoScroll}
                      onChange={(e) => {
                        e.stopPropagation();
                        setLogsAutoScroll(e.target.checked);
                      }}
                      style={{ cursor: 'pointer' }}
                    />
                    <span style={{ fontSize: '0.9em' }}>Auto-scroll</span>
                  </label>
                )}
                <span style={{ fontSize: '1.2em', color: 'var(--text-secondary)' }}>
                  {showLogsPanel ? '▼' : '▶'}
                </span>
              </div>
            </div>

            {showLogsPanel && (
              <div style={{
                height: '350px',
                backgroundColor: 'var(--bg-tertiary)',
                padding: '16px',
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: '0.85em',
                overflow: 'auto',
                color: 'var(--text-secondary)',
                lineHeight: '1.5',
                animation: 'slideDown 0.3s ease-out'
              }}>
                {jobState.logs ? (
                  jobState.logs.split('\n').map((line, idx) => {
                    const formatted = formatLogLine(line);
                    const isError = line.toLowerCase().includes('error') || line.toLowerCase().includes('failed');
                    return (
                      <div
                        key={idx}
                        style={{
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          color: isError ? 'var(--danger)' : 'var(--text-secondary)',
                          fontWeight: isError ? 'bold' : 'normal'
                        }}
                      >
                        {formatted.timestamp && <span style={{ color: 'var(--accent)' }}>[{formatted.timestamp}]</span>}
                        {formatted.timestamp && ' '}{formatted.message}
                      </div>
                    );
                  })
                ) : (
                  <div style={{ color: 'var(--text-secondary)' }}>⟳ Waiting for pipeline output...</div>
                )}
                <div ref={logsEndRef} />
              </div>
            )}
          </section>

          {/* CFG Visualization Panel - shows after Stage 1 completes */}
          {(cfgData || cfgLoading || cfgError) && (
            <section style={{
              marginBottom: '30px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              overflow: 'hidden',
              animation: 'fadeIn 0.5s ease-out'
            }}>
              <div
                onClick={() => setShowCFGPanel(!showCFGPanel)}
                style={{
                  padding: '16px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderBottom: showCFGPanel ? '1px solid var(--border-color)' : 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  userSelect: 'none'
                }}
              >
                <h3 style={{ margin: 0, fontSize: '1.1em', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>🔍</span> Binary Control Flow Graph
                  {cfgData && (
                    <span style={{
                      fontSize: '0.75em',
                      color: 'var(--success)',
                      backgroundColor: 'var(--bg-tertiary)',
                      padding: '2px 8px',
                      borderRadius: '4px'
                    }}>
                      {cfgData.functions?.length || 0} functions
                    </span>
                  )}
                </h3>
                <span style={{ fontSize: '1.2em', color: 'var(--text-secondary)' }}>
                  {showCFGPanel ? '▼' : '▶'}
                </span>
              </div>

              {showCFGPanel && (
                <div style={{ animation: 'slideDown 0.3s ease-out' }}>
                  <BinaryCFGVisualizer
                    cfgData={cfgData}
                    isLoading={cfgLoading}
                    error={cfgError || undefined}
                  />
                </div>
              )}
            </section>
          )}

          {/* Completion View */}
          {jobState.stage === 'COMPLETED' && (
            <section style={{
              marginBottom: '30px',
              padding: '20px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '2px solid var(--success)',
              animation: 'fadeIn 0.5s ease-out'
            }}>
              <h3 style={{ marginTop: 0, color: 'var(--success)' }}>🎉 Obfuscation Successful!</h3>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                gap: '12px',
                marginBottom: '20px'
              }}>
                {jobState.artifacts.includes('final.exe') && (
                  <a
                    href={`/api/binary_obfuscate/artifact/${jobState.jobId}/final.exe`}
                    download={`${binaryFile?.name.replace('.exe', '')}_obfuscated.exe` || 'obfuscated.exe'}
                    style={{
                      padding: '12px',
                      backgroundColor: 'var(--success)',
                      color: 'white',
                      textDecoration: 'none',
                      borderRadius: '4px',
                      textAlign: 'center',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.opacity = '0.9'}
                    onMouseOut={(e) => e.currentTarget.style.opacity = '1'}
                  >
                    📥 Download Binary
                  </a>
                )}

                {jobState.artifacts.includes('program_obf.bc') && (
                  <a
                    href={`/api/binary_obfuscate/artifact/${jobState.jobId}/program_obf.bc`}
                    download={`${binaryFile?.name.replace('.exe', '')}_obfuscated.bc` || 'obfuscated.bc'}
                    style={{
                      padding: '12px',
                      backgroundColor: 'var(--accent)',
                      color: 'white',
                      textDecoration: 'none',
                      borderRadius: '4px',
                      textAlign: 'center',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.opacity = '0.9'}
                    onMouseOut={(e) => e.currentTarget.style.opacity = '1'}
                  >
                    ⚙️ Download IR
                  </a>
                )}

                {jobState.artifacts.includes('metrics.json') && (
                  <a
                    href={`/api/binary_obfuscate/artifact/${jobState.jobId}/metrics.json`}
                    download="metrics.json"
                    style={{
                      padding: '12px',
                      backgroundColor: 'var(--bg-tertiary)',
                      color: 'var(--text-primary)',
                      textDecoration: 'none',
                      borderRadius: '4px',
                      textAlign: 'center',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      border: '1px solid var(--border-color)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--border-color)'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                  >
                    📊 Download Metrics
                  </a>
                )}

                {jobState.artifacts.includes('logs.txt') && (
                  <a
                    href={`/api/binary_obfuscate/artifact/${jobState.jobId}/logs.txt`}
                    download="logs.txt"
                    style={{
                      padding: '12px',
                      backgroundColor: 'var(--bg-tertiary)',
                      color: 'var(--text-primary)',
                      textDecoration: 'none',
                      borderRadius: '4px',
                      textAlign: 'center',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      border: '1px solid var(--border-color)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--border-color)'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                  >
                    📄 Download Logs
                  </a>
                )}
              </div>

              <button
                onClick={() => {
                  setJobState(null);
                  setBinaryFile(null);
                  setValidationWarning(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                  }
                  // Scroll to top
                  containerRef.current?.scrollIntoView({ behavior: 'smooth' });
                }}
                style={{
                  width: '100%',
                  padding: '12px',
                  backgroundColor: 'var(--accent)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  fontSize: '1em'
                }}
              >
                🔄 Start New Obfuscation
              </button>
            </section>
          )}

          {/* Metrics Dashboard */}
          {jobState.metrics && jobState.stage === 'COMPLETED' && (
            <section style={{
              marginBottom: '30px',
              padding: '20px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              animation: 'fadeIn 0.5s ease-out'
            }}>
              <BinaryMetricsDashboard metrics={jobState.metrics} />
            </section>
          )}

          {/* Error View */}
          {jobState.stage === 'ERROR' && (
            <section style={{
              marginBottom: '30px',
              padding: '20px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '2px solid var(--danger)',
              animation: 'fadeIn 0.5s ease-out'
            }}>
              <h3 style={{ marginTop: 0, color: 'var(--danger)' }}>❌ Obfuscation Failed</h3>

              <div style={{
                marginBottom: '20px',
                padding: '16px',
                backgroundColor: 'var(--bg-tertiary)',
                borderRadius: '4px',
                borderLeft: '4px solid var(--danger)'
              }}>
                <p style={{ margin: '0 0 10px 0', fontWeight: 'bold', fontSize: '0.95em' }}>Error Summary:</p>
                <div style={{
                  fontFamily: 'JetBrains Mono, monospace',
                  fontSize: '0.85em',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: '200px',
                  overflow: 'auto',
                  color: 'var(--text-secondary)'
                }}>
                  {getLastLogLines(10).map((line, idx) => (
                    <div key={idx} style={{ color: line.toLowerCase().includes('error') ? 'var(--danger)' : 'inherit' }}>
                      {line}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={handleRetry}
                  style={{
                    flex: 1,
                    padding: '12px',
                    backgroundColor: 'var(--accent)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  🔄 Retry with Same Settings
                </button>

                {jobState.artifacts.includes('logs.txt') && (
                  <a
                    href={`/api/binary_obfuscate/artifact/${jobState.jobId}/logs.txt`}
                    download="logs.txt"
                    style={{
                      flex: 1,
                      padding: '12px',
                      backgroundColor: 'var(--bg-tertiary)',
                      color: 'var(--text-primary)',
                      textDecoration: 'none',
                      borderRadius: '4px',
                      textAlign: 'center',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      border: '1px solid var(--border-color)'
                    }}
                  >
                    📄 Download Full Logs
                  </a>
                )}
              </div>
            </section>
          )}
        </>
      ) : (
        <>
          {/* Upload Binary Section */}
          <section style={{
            marginBottom: '30px',
            padding: '20px',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            animation: 'fadeIn 0.5s ease-out'
          }}>
            <h3 style={{ marginTop: 0 }}>📁 Upload Windows PE Binary</h3>

            {validationWarning && (
              <div style={{
                marginBottom: '16px',
                padding: '12px',
                backgroundColor: 'var(--warning)',
                color: 'var(--bg-primary)',
                borderRadius: '4px',
                fontSize: '0.9em'
              }}>
                ⚠️ {validationWarning.message}
              </div>
            )}

            <div
              onDragOver={(e) => {
                e.preventDefault();
                e.currentTarget.style.backgroundColor = 'var(--accent)';
                e.currentTarget.style.opacity = '0.9';
              }}
              onDragLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                e.currentTarget.style.opacity = '1';
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                e.currentTarget.style.opacity = '1';
                const file = e.dataTransfer.files?.[0];
                if (file) {
                  const input = fileInputRef.current;
                  if (input) {
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    input.files = dt.files;
                    handleBinaryUpload({ target: { files: dt.files } } as any);
                  }
                }
              }}
              style={{
                padding: '40px',
                border: '2px dashed var(--border-color)',
                borderRadius: '8px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                backgroundColor: 'var(--bg-tertiary)'
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".exe"
                onChange={handleBinaryUpload}
                style={{ display: 'none' }}
                disabled={isJobRunning}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isJobRunning}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'var(--accent)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isJobRunning ? 'not-allowed' : 'pointer',
                  fontSize: '1em',
                  fontWeight: 'bold',
                  opacity: isJobRunning ? 0.6 : 1
                }}
              >
                Select .exe File
              </button>
              <p style={{ margin: '10px 0 0 0', color: 'var(--text-secondary)' }}>
                or drag and drop
              </p>
              <p style={{ margin: '8px 0 0 0', fontSize: '0.85em', color: 'var(--text-secondary)' }}>
                Maximum: 50 MB • No spaces in filename • Must be PE binary
              </p>
            </div>

            {binaryFile && (
              <div style={{
                marginTop: '16px',
                padding: '16px',
                backgroundColor: 'var(--bg-tertiary)',
                borderRadius: '4px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                border: '1px solid var(--border-color)',
                animation: 'fadeIn 0.3s ease-out'
              }}>
                <div>
                  <p style={{ margin: 0, fontWeight: 'bold' }}>✓ {binaryFile.name}</p>
                  <p style={{ margin: '5px 0 0 0', color: 'var(--text-secondary)', fontSize: '0.9em' }}>
                    {(binaryFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <button
                  onClick={() => {
                    setBinaryFile(null);
                    setValidationWarning(null);
                    if (fileInputRef.current) {
                      fileInputRef.current.value = '';
                    }
                  }}
                  disabled={isJobRunning}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: 'var(--danger)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: isJobRunning ? 'not-allowed' : 'pointer',
                    opacity: isJobRunning ? 0.6 : 1
                  }}
                >
                  Remove
                </button>
              </div>
            )}
          </section>

          {/* OLLVM Passes Selection */}
          <section style={{
            marginBottom: '30px',
            padding: '20px',
            backgroundColor: 'var(--bg-secondary)',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            animation: 'fadeIn 0.5s ease-out'
          }}>
            <h3 style={{ marginTop: 0 }}>⚙️ Obfuscation Passes</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9em', marginTop: '-10px', marginBottom: '16px' }}>
              Some passes are disabled in binary mode (unsafe for lifter IR)
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
              {Object.entries(passes).map(([passName, enabled]) => {
                const isDisabled = DISABLED_PASSES.includes(passName);
                return (
                  <label
                    key={passName}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                      padding: '12px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      cursor: isDisabled || isJobRunning ? 'not-allowed' : 'pointer',
                      opacity: isDisabled || isJobRunning ? 0.6 : 1,
                      transition: 'all 0.2s ease',
                      border: '1px solid var(--border-color)',
                      userSelect: 'none'
                    }}
                    title={isDisabled ? 'Disabled in Binary Mode (unsafe for lifter IR)' : ''}
                  >
                    <input
                      type="checkbox"
                      checked={enabled}
                      onChange={() => handlePassToggle(passName as keyof typeof passes)}
                      disabled={isDisabled || isJobRunning}
                      style={{ cursor: isDisabled || isJobRunning ? 'not-allowed' : 'pointer' }}
                    />
                    <span style={{ flex: 1 }}>{passName.replace(/_/g, ' ')}</span>
                    {isDisabled && <span style={{ fontSize: '0.9em' }}>🔒</span>}
                  </label>
                );
              })}
            </div>
          </section>

          {/* Start Button */}
          <button
            onClick={handleStartObfuscation}
            disabled={!binaryFile || isJobRunning}
            style={{
              width: '100%',
              padding: '16px',
              backgroundColor: binaryFile && !isJobRunning ? 'var(--accent)' : 'var(--text-secondary)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1.1em',
              fontWeight: 'bold',
              cursor: binaryFile && !isJobRunning ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s ease',
              opacity: binaryFile && !isJobRunning ? 1 : 0.6,
              marginBottom: '20px'
            }}
            onMouseOver={(e) => {
              if (binaryFile && !isJobRunning) {
                e.currentTarget.style.transform = 'scale(1.02)';
              }
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            🚀 Start Obfuscation
          </button>

          {/* Mode Switch Info */}
          <div style={{
            padding: '16px',
            backgroundColor: 'var(--bg-tertiary)',
            borderRadius: '4px',
            fontSize: '0.9em',
            color: 'var(--text-secondary)',
            textAlign: 'center',
            border: '1px solid var(--border-color)',
            animation: 'fadeIn 0.5s ease-out'
          }}>
            💡 Want to obfuscate source code instead?{' '}
            <button
              onClick={handleModeSwitch}
              disabled={isJobRunning}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--accent)',
                cursor: isJobRunning ? 'not-allowed' : 'pointer',
                textDecoration: 'underline',
                fontSize: 'inherit',
                fontWeight: 'bold',
                opacity: isJobRunning ? 0.6 : 1
              }}
            >
              Switch to Source Mode
            </button>
          </div>
        </>
      )}
    </div>
  );
}
