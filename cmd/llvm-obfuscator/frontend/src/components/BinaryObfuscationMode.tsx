import { useState, useRef } from 'react';

interface BinaryObfuscationModeProps {
  onJobStart?: (jobId: string) => void;
}

export function BinaryObfuscationMode({ onJobStart }: BinaryObfuscationModeProps) {
  const [binaryFile, setBinaryFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pipelineSteps, setPipelineSteps] = useState<{
    ghidra: boolean;
    mcsema: boolean;
    llvm22: boolean;
  }>({ ghidra: false, mcsema: false, llvm22: false });
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
  const [progress, setProgress] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [irDownloadUrl, setIrDownloadUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleBinaryUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.exe')) {
      alert('Only .exe files are supported');
      setBinaryFile(null);
      return;
    }

    setBinaryFile(file);
  };

  const handlePassToggle = (passName: keyof typeof passes) => {
    setPasses(prev => ({
      ...prev,
      [passName]: !prev[passName]
    }));
  };

  const handleStartObfuscation = async () => {
    if (!binaryFile) {
      alert('Please upload a binary file');
      return;
    }

    setIsProcessing(true);
    setProgress('Initializing pipeline...');
    setPipelineSteps({ ghidra: false, mcsema: false, llvm22: false });
    setMetrics(null);
    setDownloadUrl(null);
    setIrDownloadUrl(null);

    try {
      const formData = new FormData();
      formData.append('binary_file', binaryFile);
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

      if (onJobStart) {
        onJobStart(jobId);
      }

      // Poll for progress
      let completed = false;
      let stepCheckCount = 0;

      while (!completed && stepCheckCount < 300) { // 5 minute timeout
        await new Promise(resolve => setTimeout(resolve, 1000));

        const statusResponse = await fetch(`/api/binary_obfuscate/status/${jobId}`);
        if (!statusResponse.ok) continue;

        const statusData = await statusResponse.json();

        // Update pipeline steps based on status
        if (statusData.status === 'ghidra_lifting') {
          setPipelineSteps(prev => ({ ...prev, ghidra: false }));
          setProgress('Running Ghidra CFG export...');
        } else if (statusData.status === 'mcsema_lifting') {
          setPipelineSteps(prev => ({ ...prev, ghidra: true, mcsema: false }));
          setProgress('Converting CFG to LLVM IR with McSema...');
        } else if (statusData.status === 'ir_upgrading') {
          setPipelineSteps(prev => ({ ...prev, ghidra: true, mcsema: true, llvm22: false }));
          setProgress('Upgrading IR to LLVM 22...');
        } else if (statusData.status === 'applying_passes') {
          setPipelineSteps(prev => ({ ...prev, llvm22: true }));
          setProgress('Applying OLLVM obfuscation passes...');
        } else if (statusData.status === 'completed') {
          setPipelineSteps(prev => ({ ...prev, ghidra: true, mcsema: true, llvm22: true }));
          setProgress('Compilation complete!');

          // Get metrics and download URLs
          if (statusData.metrics) {
            setMetrics(statusData.metrics);
          }
          if (statusData.download_url) {
            const binaryDownloadName = `${binaryFile.name.replace('.exe', '')}_obfuscated.exe`;
            setDownloadUrl(statusData.download_url);
          }
          if (statusData.ir_download_url) {
            setIrDownloadUrl(statusData.ir_download_url);
          }
          completed = true;
        } else if (statusData.status === 'error') {
          throw new Error(statusData.error || 'Unknown error occurred');
        }

        stepCheckCount++;
      }

      if (!completed) {
        throw new Error('Pipeline timeout');
      }
    } catch (error) {
      setProgress(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      alert(`Obfuscation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1000px' }}>
      {/* Upload Binary Section */}
      <section style={{
        marginBottom: '30px',
        padding: '20px',
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '8px',
        border: '1px solid var(--border-color)'
      }}>
        <h3 style={{ marginTop: 0 }}>Upload Windows PE Binary</h3>

        <div style={{
          padding: '30px',
          border: '2px dashed var(--border-color)',
          borderRadius: '8px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          backgroundColor: 'var(--bg-tertiary)'
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.currentTarget.style.backgroundColor = 'var(--accent)';
        }}
        onDragLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
        }}
        onDrop={(e) => {
          e.preventDefault();
          e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
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
        }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".exe"
            onChange={handleBinaryUpload}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '10px 20px',
              backgroundColor: 'var(--accent)',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '1em'
            }}>
            Select .exe File
          </button>
          <p style={{ margin: '10px 0 0 0', color: 'var(--text-secondary)' }}>
            or drag and drop
          </p>
        </div>

        {binaryFile && (
          <div style={{
            marginTop: '15px',
            padding: '15px',
            backgroundColor: 'var(--bg-tertiary)',
            borderRadius: '4px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <p style={{ margin: 0, fontWeight: 'bold' }}>{binaryFile.name}</p>
              <p style={{ margin: '5px 0 0 0', color: 'var(--text-secondary)', fontSize: '0.9em' }}>
                {(binaryFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={() => {
                setBinaryFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              style={{
                padding: '8px 16px',
                backgroundColor: 'var(--danger)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
              }}>
              Remove
            </button>
          </div>
        )}
      </section>

      {/* Pipeline Progress Section */}
      {isProcessing && (
        <section style={{
          marginBottom: '30px',
          padding: '20px',
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)'
        }}>
          <h3 style={{ marginTop: 0 }}>Obfuscation Pipeline Progress</h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {[
              { name: 'Ghidra CFG Export', done: pipelineSteps.ghidra },
              { name: 'McSema LLVM Lift', done: pipelineSteps.mcsema },
              { name: 'LLVM 22 IR Upgrade', done: pipelineSteps.llvm22 }
            ].map((step, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '24px',
                  height: '24px',
                  borderRadius: '4px',
                  backgroundColor: step.done ? 'var(--success)' : 'var(--border-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold'
                }}>
                  {step.done ? '✓' : '•'}
                </div>
                <span>{step.name}</span>
              </div>
            ))}
          </div>

          {progress && (
            <div style={{
              marginTop: '15px',
              padding: '10px',
              backgroundColor: 'var(--bg-tertiary)',
              borderRadius: '4px',
              fontSize: '0.9em',
              color: 'var(--text-secondary)'
            }}>
              {progress}
            </div>
          )}
        </section>
      )}

      {/* OLLVM Passes Selection */}
      {!isProcessing && (
        <section style={{
          marginBottom: '30px',
          padding: '20px',
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)'
        }}>
          <h3 style={{ marginTop: 0 }}>Obfuscation Passes</h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            {Object.entries(passes).map(([passName, enabled]) => (
              <label key={passName} style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '10px',
                backgroundColor: 'var(--bg-tertiary)',
                borderRadius: '4px',
                cursor: 'pointer'
              }}>
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={() => handlePassToggle(passName as keyof typeof passes)}
                  style={{ cursor: 'pointer' }}
                />
                <span>{passName.replace(/_/g, ' ')}</span>
              </label>
            ))}
          </div>
        </section>
      )}

      {/* Start Button / Results */}
      {!isProcessing ? (
        <button
          onClick={handleStartObfuscation}
          disabled={!binaryFile}
          style={{
            width: '100%',
            padding: '15px',
            backgroundColor: binaryFile ? 'var(--accent)' : 'var(--text-secondary)',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '1.1em',
            fontWeight: 'bold',
            cursor: binaryFile ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s ease'
          }}>
          Start Obfuscation
        </button>
      ) : null}

      {/* Metrics & Downloads */}
      {metrics && (
        <section style={{
          marginTop: '30px',
          padding: '20px',
          backgroundColor: 'var(--bg-secondary)',
          borderRadius: '8px',
          border: '1px solid var(--border-color)'
        }}>
          <h3 style={{ marginTop: 0 }}>Obfuscation Metrics</h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '15px',
            marginBottom: '20px'
          }}>
            {metrics.size_before && (
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9em' }}>Size Before</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '1.2em', fontWeight: 'bold' }}>{metrics.size_before}</p>
              </div>
            )}
            {metrics.size_after && (
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9em' }}>Size After</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '1.2em', fontWeight: 'bold' }}>{metrics.size_after}</p>
              </div>
            )}
            {metrics.instruction_count_before && (
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9em' }}>Instructions Before</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '1.2em', fontWeight: 'bold' }}>{metrics.instruction_count_before}</p>
              </div>
            )}
            {metrics.instruction_count_after && (
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9em' }}>Instructions After</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '1.2em', fontWeight: 'bold' }}>{metrics.instruction_count_after}</p>
              </div>
            )}
            {metrics.cfg_complexity && (
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9em' }}>CFG Complexity</p>
                <p style={{ margin: '5px 0 0 0', fontSize: '1.2em', fontWeight: 'bold' }}>{metrics.cfg_complexity}</p>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            {downloadUrl && (
              <a
                href={downloadUrl}
                download={`${binaryFile?.name.replace('.exe', '')}_obfuscated.exe` || 'obfuscated.exe'}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: 'var(--success)',
                  color: 'white',
                  textDecoration: 'none',
                  borderRadius: '4px',
                  textAlign: 'center',
                  fontWeight: 'bold'
                }}>
                Download Binary
              </a>
            )}
            {irDownloadUrl && (
              <a
                href={irDownloadUrl}
                download={`${binaryFile?.name.replace('.exe', '')}_obfuscated.bc` || 'obfuscated.bc'}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: 'var(--accent)',
                  color: 'white',
                  textDecoration: 'none',
                  borderRadius: '4px',
                  textAlign: 'center',
                  fontWeight: 'bold'
                }}>
                Download IR
              </a>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
