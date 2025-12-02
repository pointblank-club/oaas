import { useState, useCallback, useEffect } from 'react';
import './App.css';
import { GitHubIntegration, FileTree } from './components';

type Platform = 'linux' | 'windows' | 'macos';

// Demo Programs
const DEMO_PROGRAMS = {
  'demo_auth_c': {
    name: 'Authentication System (C)',
    language: 'c' as const,
    code: `#include <stdio.h>
#include <string.h>

char* ADMIN_PASSWORD = "Admin@SecurePass2024!";
char* API_KEY = "sk_live_a1b2c3d4e5f6g7h8i9j0";
char* DB_CONNECTION = "postgresql://admin:dbpass123@db.internal:5432/prod";

int authenticate(char* username, char* password) {
    printf("[AUTH] Checking credentials for: %s\\n", username);
    if (strcmp(username, "admin") == 0 && strcmp(password, ADMIN_PASSWORD) == 0) {
        printf("[AUTH] Admin access granted\\n");
        return 1;
    }
    printf("[AUTH] Authentication failed\\n");
    return 0;
}

void connect_database() {
    printf("[DB] Connecting to: %s\\n", DB_CONNECTION);
    printf("[DB] Connection established\\n");
}

int main() {
    char* username = "admin";
    char* password = ADMIN_PASSWORD;

    printf("=== Authentication System v1.0 ===\\n\\n");

    if (authenticate(username, password)) {
        printf("\\n[SUCCESS] Access granted\\n");
        printf("[API] Using API key: %s\\n", API_KEY);
        connect_database();
        printf("\\n[USER] Profile loaded\\n");
        printf("  Username: admin\\n");
        printf("  Role: Administrator\\n");
        printf("  Access Level: 10\\n");
        return 0;
    }

    printf("\\n[FAIL] Access denied\\n");
    return 1;
}
`,
  },
  'demo_license_cpp': {
    name: 'License Validator (C++)',
    language: 'cpp' as const,
    code: `#include <iostream>
#include <string>
#include <vector>
#include <map>

// Hardcoded secrets (anti-pattern for demo)
const std::string MASTER_KEY = "ENTERPRISE-MASTER-2024-A1B2C3D4E5F6";
const std::string RSA_KEY = "-----BEGIN RSA PRIVATE KEY-----\\nMIIE...";
const std::string AES_KEY = "AES256_PROD_KEY_2024_DO_NOT_SHARE";
const std::string ACTIVATION_SECRET = "activation_secret_xyz_2024";

enum class LicenseType { TRIAL, STANDARD, PROFESSIONAL, ENTERPRISE };

template<typename T>
class SecureContainer {
private:
    std::vector<T> data;
    std::string key;
public:
    SecureContainer(const std::string& k) : key(k) {
        std::cout << "[SECURE] Container initialized\\n";
    }
    void add(const T& item) { data.push_back(item); }
    size_t size() const { return data.size(); }
};

class License {
protected:
    std::string license_key;
    std::string owner;
    LicenseType type;
    bool activated;
public:
    License(const std::string& key, const std::string& own, LicenseType t)
        : license_key(key), owner(own), type(t), activated(false) {}

    virtual bool validate() const {
        std::cout << "[LICENSE] Validating: " << license_key << "\\n";
        if (license_key == MASTER_KEY) {
            std::cout << "[LICENSE] Master key detected\\n";
            return true;
        }
        return activated;
    }

    void activate(const std::string& code) {
        if (code == ACTIVATION_SECRET) {
            activated = true;
            std::cout << "[LICENSE] Activation successful\\n";
        }
    }

    bool is_activated() const { return activated; }
};

class EnterpriseLicense : public License {
private:
    int max_users;
    std::vector<std::string> features;
public:
    EnterpriseLicense(const std::string& key, const std::string& own, int users)
        : License(key, own, LicenseType::ENTERPRISE), max_users(users) {
        features = {"Analytics", "Support", "Cloud", "API"};
    }

    bool validate() const override {
        if (!License::validate()) return false;
        std::cout << "[ENTERPRISE] Max users: " << max_users << "\\n";
        return true;
    }
};

int main(int argc, char** argv) {
    std::cout << "=== License Validator v2.0 ===\\n\\n";

    std::string key = (argc >= 2) ? argv[1] : MASTER_KEY;
    std::string code = (argc >= 3) ? argv[2] : ACTIVATION_SECRET;

    EnterpriseLicense* license = new EnterpriseLicense(key, "Acme Corp", 100);
    license->activate(code);

    bool valid = license->validate();

    if (valid && license->is_activated()) {
        std::cout << "\\n[SUCCESS] License valid\\n";
        std::cout << "[CRYPTO] AES Key: " << AES_KEY << "\\n";
        delete license;
        return 0;
    }

    std::cout << "\\n[FAIL] License invalid\\n";
    delete license;
    return 1;
}
`,
  },
  'password_checker': {
    name: 'Password Strength Checker (C)',
    language: 'c' as const,
    code: `#include <stdio.h>
#include <string.h>
#include <ctype.h>

char* MASTER_BYPASS = "Admin@Override2024!";
char* SECRET_SALT = "s3cr3t_salt_2024";

int check_strength(char* password) {
    int length = strlen(password);
    int has_upper = 0;
    int has_lower = 0;
    int has_digit = 0;
    int has_special = 0;
    int i;
    int score;

    for (i = 0; password[i] != 0; i++) {
        if (isupper(password[i])) has_upper = 1;
        if (islower(password[i])) has_lower = 1;
        if (isdigit(password[i])) has_digit = 1;
        if (!isalnum(password[i])) has_special = 1;
    }

    score = 0;
    if (length >= 8) score = score + 25;
    if (length >= 12) score = score + 25;
    if (has_upper) score = score + 15;
    if (has_lower) score = score + 15;
    if (has_digit) score = score + 10;
    if (has_special) score = score + 10;

    printf("[ANALYSIS] Password: %s\\n", password);
    printf("  Length: %d characters\\n", length);
    printf("  Uppercase: %s\\n", has_upper ? "Yes" : "No");
    printf("  Lowercase: %s\\n", has_lower ? "Yes" : "No");
    printf("  Digits: %s\\n", has_digit ? "Yes" : "No");
    printf("  Special: %s\\n", has_special ? "Yes" : "No");
    printf("\\n[RESULT] Strength: %d/100\\n", score);

    return score;
}

int main() {
    char* password = "TestPass123!";

    printf("=== Password Strength Checker v1.0 ===\\n\\n");

    if (strcmp(password, MASTER_BYPASS) == 0) {
        printf("[BYPASS] Master password detected!\\n");
        printf("[SECRET] Salt: %s\\n", SECRET_SALT);
        printf("\\n[ADMIN] Full access granted\\n");
        return 0;
    }

    check_strength(password);
    return 0;
}
`,
  }
};

interface ReportData {
  input_parameters?: {
    source_file: string;
    platform: string;
    obfuscation_level: number;
    requested_passes: string[];
    applied_passes: string[];
    compiler_flags: string[];
    timestamp: string;
  };
  warnings?: string[];
  baseline_metrics?: {
    file_size: number;
    binary_format: string;
    sections: Record<string, number>;
    symbols_count: number;
    functions_count: number;
    entropy: number;
  };
  output_attributes: {
    file_size: number;
    binary_format: string;
    sections: Record<string, number>;
    symbols_count: number;
    functions_count: number;
    entropy: number;
    obfuscation_methods: string[];
  };
  comparison?: {
    size_change: number;
    size_change_percent: number;
    symbols_removed: number;
    symbols_removed_percent: number;
    functions_removed: number;
    functions_removed_percent: number;
    entropy_increase: number;
    entropy_increase_percent: number;
  };
  bogus_code_info: {
    dead_code_blocks: number;
    opaque_predicates: number;
    junk_instructions: number;
    code_bloat_percentage: number;
  };
  cycles_completed: {
    total_cycles: number;
    per_cycle_metrics: Array<{
      cycle: number;
      passes_applied: string[];
      duration_ms: number;
    }>;
  };
  string_obfuscation: {
    total_strings: number;
    encrypted_strings: number;
    encryption_method: string;
    encryption_percentage: number;
  };
  fake_loops_inserted: {
    count: number;
    types: string[];
    locations: string[];
  };
  symbol_obfuscation: {
    enabled: boolean;
    symbols_obfuscated?: number;
    algorithm?: string;
  };
  obfuscation_score: number;
  symbol_reduction: number;
  function_reduction: number;
  size_reduction: number;
  entropy_increase: number;
  estimated_re_effort: string;
}

interface Modal {
  type: 'error' | 'warning' | 'success';
  title: string;
  message: string;
}

interface RepoFile {
  path: string;
  content: string;
  is_binary: boolean;
}

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [file, setFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'file' | 'paste' | 'github'>('file');
  const [pastedSource, setPastedSource] = useState('');
  const [selectedDemo, setSelectedDemo] = useState<string>('');
  const [jobId, setJobId] = useState<string | null>(null);
  
  // GitHub integration state
  const [repoFiles, setRepoFiles] = useState<RepoFile[]>([]);
  const [selectedRepoFile, setSelectedRepoFile] = useState<RepoFile | null>(null);
  const [repoName, setRepoName] = useState<string>('');
  const [repoBranch, setRepoBranch] = useState<string>('');
  const [repoSessionId, setRepoSessionId] = useState<string | null>(null);  // Fast clone session
  const [repoFileCount, setRepoFileCount] = useState<number>(0);  // File count from fast clone
  const [showGitHubModal, setShowGitHubModal] = useState(false);
  const [downloadUrls, setDownloadUrls] = useState<Record<Platform, string | null>>({
    linux: null,
    windows: null,
    macos: null
  });
  const [binaryName, setBinaryName] = useState<string | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(false);
  const [modal, setModal] = useState<Modal | null>(null);
  const [progress, setProgress] = useState<{ message: string; percent: number } | null>(null);
  const [detectedLanguage, setDetectedLanguage] = useState<'c' | 'cpp' | null>(null);

  // Layer states (execution order: 1→2→3→4)
  const [layer1, setLayer1] = useState(false); // Symbol obfuscation (PRE-COMPILE, FIRST)
  const [layer2, setLayer2] = useState(false); // String encryption (PRE-COMPILE, SECOND)
  const [layer2_5, setLayer2_5] = useState(false); // Indirect calls (PRE-COMPILE, 2.5)
  const [layer3, setLayer3] = useState(false); // OLLVM passes (COMPILE, THIRD - optional)
  const [layer4, setLayer4] = useState(false); // Compiler flags (COMPILE, FINAL)

  // Layer 1: Symbol Obfuscation sub-options
  const [symbolAlgorithm, setSymbolAlgorithm] = useState('sha256');
  const [symbolHashLength, setSymbolHashLength] = useState(12);
  const [symbolPrefix, setSymbolPrefix] = useState('typed');
  const [symbolSalt, setSymbolSalt] = useState('');

  // Layer 2: String Encryption sub-options
  const [fakeLoops, setFakeLoops] = useState(0);

  // Layer 2.5: Indirect Call Obfuscation sub-options
  const [indirectStdlib, setIndirectStdlib] = useState(true);
  const [indirectCustom, setIndirectCustom] = useState(true);

  // Layer 3: OLLVM Passes sub-options
  const [passFlattening, setPassFlattening] = useState(false);
  const [passSubstitution, setPassSubstitution] = useState(false);
  const [passBogusControlFlow, setPassBogusControlFlow] = useState(false);
  const [passSplitBasicBlocks, setPassSplitBasicBlocks] = useState(false);
  const [cycles, setCycles] = useState(1);

  // Layer 4: Compiler Flags sub-options
  const [flagLTO, setFlagLTO] = useState(false);
  const [flagSymbolHiding, setFlagSymbolHiding] = useState(false);
  const [flagOmitFramePointer, setFlagOmitFramePointer] = useState(false);
  const [flagSpeculativeLoadHardening, setFlagSpeculativeLoadHardening] = useState(false);
  const [flagO3, setFlagO3] = useState(false);
  const [flagStripSymbols, setFlagStripSymbols] = useState(false);
  const [flagNoBuiltin, setFlagNoBuiltin] = useState(false);

  // Configuration states
  const [obfuscationLevel, setObfuscationLevel] = useState(5);
  const [targetPlatform, setTargetPlatform] = useState<Platform>('linux');
  const [entrypointCommand, setEntrypointCommand] = useState<string>('./a.out');

  // Build system configuration (for complex projects like CURL)
  type BuildSystem = 'simple' | 'cmake' | 'make' | 'autotools' | 'custom';
  const [buildSystem, setBuildSystem] = useState<BuildSystem>('simple');
  const [customBuildCommand, setCustomBuildCommand] = useState<string>('');
  const [outputBinaryPath, setOutputBinaryPath] = useState<string>('');
  const [cmakeOptions, setCmakeOptions] = useState<string>('');  // Extra cmake flags like -DFOO=OFF

  useEffect(() => {
    document.body.className = darkMode ? 'dark' : 'light';
  }, [darkMode]);

  useEffect(() => {
    fetch('/api/health')
      .then(res => setServerStatus(res.ok ? 'online' : 'offline'))
      .catch(() => setServerStatus('offline'));
  }, []);

  // Auto-detect language
  const detectLanguage = useCallback((filename: string, content?: string): 'c' | 'cpp' => {
    const ext = filename.toLowerCase().split('.').pop();
    if (ext === 'cpp' || ext === 'cc' || ext === 'cxx' || ext === 'c++') return 'cpp';
    if (ext === 'c') return 'c';

    if (content) {
      const cppIndicators = [
        /\bclass\b/, /\bnamespace\b/, /\btemplate\s*</, /\bstd::/,
        /\b(public|private|protected):/, /#include\s*<(iostream|string|vector)>/,
        /\bvirtual\b/, /\boperator\s*\(/
      ];
      if (cppIndicators.some(regex => regex.test(content))) return 'cpp';
    }
    return 'c';
  }, []);

  const onPick = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = e.target.files?.[0] ?? null;
    setFile(nextFile);
    if (nextFile) {
      setInputMode('file');
      const lang = detectLanguage(nextFile.name);
      setDetectedLanguage(lang);
    }
  }, [detectLanguage]);

  const onGitHubFilesLoaded = useCallback((files: RepoFile[], repoName: string, branch: string) => {
    setRepoFiles(files);
    setRepoName(repoName);
    setRepoBranch(branch);
    setInputMode('github');
    setShowGitHubModal(false);

    // Auto-select file containing main() function
    const mainFile = files.find(f => {
      const ext = f.path.toLowerCase().split('.').pop();
      if (ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext)) {
        // Check if file contains main function
        return /\bmain\s*\(/.test(f.content);
      }
      return false;
    });

    // If no main file found, select first C/C++ file
    const sourceFile = mainFile || files.find(f => {
      const ext = f.path.toLowerCase().split('.').pop();
      return ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext);
    });

    if (sourceFile) {
      setSelectedRepoFile(sourceFile);
      const lang = detectLanguage(sourceFile.path, sourceFile.content);
      setDetectedLanguage(lang);
    }
  }, [detectLanguage]);

  // Callback for fast clone - repo stays on backend with all files (including build system files)
  const onRepoCloned = useCallback((sessionId: string, name: string, branch: string, fileCount: number) => {
    setRepoSessionId(sessionId);
    setRepoName(name);
    setRepoBranch(branch);
    setRepoFileCount(fileCount);
    setRepoFiles([]);  // Clear any old files - we don't need them for fast clone
    setInputMode('github');
    setShowGitHubModal(false);
  }, []);

  const onGitHubError = useCallback((error: string) => {
    setModal({
      type: 'error',
      title: 'GitHub Error',
      message: error
    });
  }, []);

  const onSelectDemo = useCallback((demoKey: string) => {
    if (!demoKey) {
      setSelectedDemo('');
      setPastedSource('');
      setDetectedLanguage(null);
      return;
    }

    const demo = DEMO_PROGRAMS[demoKey as keyof typeof DEMO_PROGRAMS];
    if (demo) {
      setSelectedDemo(demoKey);
      setPastedSource(demo.code);
      setDetectedLanguage(demo.language);
      setInputMode('paste');
    }
  }, []);

  // Count active obfuscation layers
  const countLayers = useCallback(() => {
    let count = 0;
    if (layer1) count++; // Symbol obfuscation
    if (layer2) count++; // String encryption
    if (layer2_5) count++; // Indirect calls
    if (layer3 && (passFlattening || passSubstitution || passBogusControlFlow || passSplitBasicBlocks)) count++; // OLLVM passes
    if (layer4 && (flagLTO || flagSymbolHiding || flagOmitFramePointer || flagSpeculativeLoadHardening || flagO3 || flagStripSymbols || flagNoBuiltin)) count++; // Compiler flags
    return count;
  }, [layer1, layer2, layer2_5, layer3, layer4, passFlattening, passSubstitution, passBogusControlFlow, passSplitBasicBlocks, flagLTO, flagSymbolHiding, flagOmitFramePointer, flagSpeculativeLoadHardening, flagO3, flagStripSymbols, flagNoBuiltin]);

  // Validate source code syntax
  const validateCode = (code: string, language: 'c' | 'cpp'): { valid: boolean; error?: string } => {
    if (!code || code.trim().length === 0) {
      return { valid: false, error: 'Source code is empty' };
    }

    // Basic syntax checks
    const hasMainFunction = /\bmain\s*\(/.test(code);
    if (!hasMainFunction) {
      return { valid: false, error: 'No main() function found. Invalid C/C++ program.' };
    }

    // Check for basic C/C++ structure
    const hasInclude = /#include/.test(code);
    const hasFunction = /\w+\s+\w+\s*\([^)]*\)\s*\{/.test(code);

    if (!hasInclude && !hasFunction) {
      return { valid: false, error: 'Invalid C/C++ syntax: missing includes or function definitions' };
    }

    // Check for balanced braces
    const openBraces = (code.match(/\{/g) || []).length;
    const closeBraces = (code.match(/\}/g) || []).length;
    if (openBraces !== closeBraces) {
      return { valid: false, error: `Syntax error: Unbalanced braces (${openBraces} opening, ${closeBraces} closing)` };
    }

    // Check for balanced parentheses in preprocessor directives
    const includeLines = code.match(/#include\s*[<"][^>"]+[>"]/g);
    if (code.includes('#include') && !includeLines) {
      return { valid: false, error: 'Syntax error: Invalid #include directive' };
    }

    return { valid: true };
  };

  const fileToBase64 = (f: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (typeof result === 'string') {
          const [, base64] = result.split(',');
          resolve(base64 ?? result);
        } else {
          reject(new Error('Failed to read file'));
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(f);
    });

  const stringToBase64 = (value: string) => {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(value);
    let binary = '';
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary);
  };

  // Handle cascading layer selection
  const handleLayerChange = (layer: number, value: boolean) => {
    if (layer === 1) setLayer1(value);
    if (layer === 2) setLayer2(value);
    if (layer === 3) {
      setLayer3(value);
      // Disable LTO when Layer 3 is enabled (LLVM version incompatibility)
      if (value && flagLTO) {
        setFlagLTO(false);
      }
    }
    if (layer === 4) setLayer4(value);
  };

  const onSubmit = useCallback(async () => {
    // Validation: Check if source is provided
    if (inputMode === 'file' && !file) {
      setModal({
        type: 'error',
        title: 'No File Selected',
        message: 'Please select a C or C++ source file to obfuscate.'
      });
      return;
    }

    if (inputMode === 'paste' && pastedSource.trim().length === 0) {
      setModal({
        type: 'error',
        title: 'Empty Source Code',
        message: 'Please paste valid C/C++ source code before submitting.'
      });
      return;
    }

    // For GitHub mode: require either repoSessionId (fast clone) or repoFiles (legacy)
    if (inputMode === 'github' && !repoSessionId && repoFiles.length === 0) {
      setModal({
        type: 'error',
        title: 'No Repository Loaded',
        message: 'Please load a GitHub repository first.'
      });
      return;
    }

    // Validation: Check if at least one layer is selected
    const layerCount = countLayers();
    if (layerCount === 0) {
      setModal({
        type: 'warning',
        title: 'No Obfuscation Layers Selected',
        message: 'Please enable at least one obfuscation layer (Layer 1-4) before proceeding.'
      });
      return;
    }

    // Get source code for validation
    let sourceCode: string;
    let filename: string;
    let language: 'c' | 'cpp';

    try {
      if (inputMode === 'file') {
        sourceCode = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error('Failed to read file'));
          reader.readAsText(file as File);
        });
        filename = (file as File).name;
        language = detectLanguage(filename, sourceCode);
      } else if (inputMode === 'paste') {
        sourceCode = pastedSource;
        language = detectLanguage('pasted_source', pastedSource);
        setDetectedLanguage(language);
        filename = language === 'cpp' ? 'pasted_source.cpp' : 'pasted_source.c';
      } else {
        // GitHub mode - for multi-file projects, we don't need a specific file selected
        if (repoSessionId) {
          // Fast clone mode: files are on backend, we can't inspect them locally
          // Use placeholder values - backend will find the main file
          sourceCode = '// Fast clone mode - files on server';
          filename = 'main.c';  // Placeholder - backend finds actual main file
          language = 'c';
        } else if (repoFiles.length > 0) {
          // Legacy mode: use first C/C++ file for language detection
          const firstCppFile = repoFiles.find(f => {
            const ext = f.path.toLowerCase().split('.').pop();
            return ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext);
          });

          if (firstCppFile) {
            sourceCode = firstCppFile.content;
            filename = firstCppFile.path.split('/').pop() || 'repo_file.c';
            language = detectLanguage(filename, sourceCode);
          } else {
            sourceCode = '';
            filename = 'repo_file.c';
            language = 'c';
          }
        } else {
          // Fallback
          sourceCode = '';
          filename = 'repo_file.c';
          language = 'c';
        }
      }

      // Validate code syntax
      if (inputMode === 'github') {
        // Skip validation for fast clone mode - backend handles it
        if (!repoSessionId) {
          // Legacy mode: validate that at least one file has main()
          const hasMainFunction = repoFiles.some(f => {
            const ext = f.path.toLowerCase().split('.').pop();
            if (ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext)) {
              return /\bmain\s*\(/.test(f.content);
            }
            return false;
          });

          if (!hasMainFunction) {
            setModal({
              type: 'error',
              title: 'Invalid Repository',
              message: 'No main() function found in any C/C++ source file. The repository must contain at least one file with a main() function.'
            });
            return;
          }

          // Basic validation: check for C/C++ files
          const hasCppFiles = repoFiles.some(f => {
            const ext = f.path.toLowerCase().split('.').pop();
            return ext && ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++'].includes(ext);
          });

          if (!hasCppFiles) {
            setModal({
              type: 'error',
              title: 'Invalid Repository',
              message: 'No C/C++ source files found in repository.'
            });
            return;
          }
        }
        // For fast clone, backend will validate (it already checks file count > 0)
      } else {
        // For single file mode, validate the file itself
        const validation = validateCode(sourceCode, language);
        if (!validation.valid) {
          setModal({
            type: 'error',
            title: 'Invalid Source Code',
            message: validation.error || 'The provided source code contains syntax errors.'
          });
          return;
        }
      }

    } catch (err) {
      setModal({
        type: 'error',
        title: 'File Read Error',
        message: 'Failed to read the source file. Please try again.'
      });
      return;
    }

    setLoading(true);
    setReport(null);
    setDownloadUrls({ linux: null, windows: null, macos: null });
    setBinaryName(null);
    setProgress({ message: 'Initializing...', percent: 0 });

    // Prepare source files for multi-file projects (GitHub mode)
    // Priority: 1) repoSessionId (fast clone, keeps all files on backend)
    //           2) repoFiles (legacy, only C/C++ files)
    let sourceFiles = null;
    const useSessionId = inputMode === 'github' && repoSessionId;

    if (inputMode === 'github' && repoFiles.length > 0 && !repoSessionId) {
      // Legacy mode: filter only C/C++ source and header files
      const validExtensions = ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++'];
      sourceFiles = repoFiles
        .filter(f => {
          const ext = f.path.toLowerCase().split('.').pop();
          return ext && validExtensions.includes(ext);
        })
        .map(f => ({
          path: f.path,
          content: f.content
        }));
    }

    try {
      const fileCountDisplay = useSessionId ? repoFileCount : (sourceFiles ? sourceFiles.length : 1);
      setProgress({
        message: inputMode === 'file' ? 'Uploading file...' : inputMode === 'github' ? `Processing ${fileCountDisplay} repository file${fileCountDisplay > 1 ? 's' : ''}...` : 'Encoding source...',
        percent: 10
      });
      const source_b64 = inputMode === 'file' ? await fileToBase64(file as File) : stringToBase64(sourceCode);

      // Build compiler flags based on Layer 4 (Compiler Flags) - only selected flags
      const flags: string[] = [];
      if (layer4) {
        // Add non-LTO flags first
        if (flagO3) flags.push('-O3');
        if (flagNoBuiltin) flags.push('-fno-builtin');
        if (flagStripSymbols) flags.push('-Wl,-s');
        if (flagSymbolHiding) flags.push('-fvisibility=hidden');
        if (flagOmitFramePointer) flags.push('-fomit-frame-pointer');
        if (flagSpeculativeLoadHardening) flags.push('-mspeculative-load-hardening');
        // Add LTO at the end (as requested by user)
        if (flagLTO) flags.push('-flto', '-flto=thin');
      }

      // Build OLLVM pass flags based on Layer 3 - only selected passes
      if (layer3) {
        if (passFlattening) flags.push('-mllvm', '-fla');
        if (passBogusControlFlow) flags.push('-mllvm', '-bcf');
        if (passSubstitution) flags.push('-mllvm', '-sub');
        if (passSplitBasicBlocks) flags.push('-mllvm', '-split');
      }

      const payload = {
        source_code: source_b64,
        filename: filename,
        platform: targetPlatform,
        entrypoint_command: buildSystem === 'simple' ? (entrypointCommand.trim() || './a.out') : undefined,
        // Fast clone: use repo_session_id to keep all files on backend (including build system files)
        // Legacy: use source_files (filtered C/C++ only - missing CMakeLists.txt, configure, etc.)
        repo_session_id: useSessionId ? repoSessionId : undefined,
        source_files: useSessionId ? undefined : sourceFiles,
        // Build system configuration for complex projects
        build_system: buildSystem,
        build_command: buildSystem === 'custom' ? customBuildCommand : undefined,
        output_binary_path: buildSystem !== 'simple' && outputBinaryPath ? outputBinaryPath : undefined,
        cmake_options: buildSystem === 'cmake' && cmakeOptions ? cmakeOptions : undefined,
        config: {
          level: obfuscationLevel,
          passes: {
            flattening: layer3 && passFlattening,
            substitution: layer3 && passSubstitution,
            bogus_control_flow: layer3 && passBogusControlFlow,
            split: layer3 && passSplitBasicBlocks
          },
          cycles: layer3 ? cycles : 1,
          string_encryption: layer2,
          fake_loops: layer2 ? fakeLoops : 0,
          symbol_obfuscation: {
            enabled: layer1,
            algorithm: layer1 ? symbolAlgorithm : 'sha256',
            hash_length: layer1 ? symbolHashLength : 12,
            prefix_style: layer1 ? symbolPrefix : 'typed',
            salt: layer1 && symbolSalt ? symbolSalt : null
          },
          indirect_calls: {
            enabled: layer2_5,
            obfuscate_stdlib: layer2_5 ? indirectStdlib : true,
            obfuscate_custom: layer2_5 ? indirectCustom : true
          }
        },
        report_formats: ['json', 'markdown'],
        custom_flags: Array.from(new Set(flags.flatMap(f => f.split(' ')).map(t => t.trim()).filter(t => t.length > 0)))
      };

      setProgress({ message: 'Processing obfuscation...', percent: 30 });

      const res = await fetch('/api/obfuscate/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }

      setProgress({ message: 'Finalizing...', percent: 90 });
      const data = await res.json();

      // Generate custom binary name based on layer count
      const customBinaryName = `obfuscated_${layerCount}_layer`;

      setJobId(data.job_id);

      // Handle response - always single platform (Linux)
      const downloadUrlsMap: Record<Platform, string | null> = {
        linux: null,
        windows: null,
        macos: null
      };

      if (data.download_url) {
        // Single platform build (always Linux)
        downloadUrlsMap.linux = data.download_url;
      }

      setDownloadUrls(downloadUrlsMap);
      setBinaryName(customBinaryName);
      setProgress({ message: 'Complete!', percent: 100 });

      // Show success modal
      setModal({
        type: 'success',
        title: 'Obfuscation Complete',
        message: `Successfully applied ${layerCount} layer${layerCount > 1 ? 's' : ''} of obfuscation. Binary ready for download.`
      });

      // Auto-fetch report
      if (data.report_url) {
        try {
          const reportRes = await fetch(data.report_url);
          if (reportRes.ok) {
            const reportData = await reportRes.json();
            console.log('[DEBUG] Report data received:', reportData);
            setReport(reportData);
          }
        } catch (reportErr) {
          console.error('[DEBUG] Failed to fetch report:', reportErr);
          // Don't fail the whole operation if just report fetch fails
        }
      }
    } catch (err) {
      console.error('[DEBUG] Obfuscation error:', err);
      setProgress(null);
      const errorMsg = err instanceof Error ? err.message : String(err);

      // Extract detailed error message
      let userFriendlyError = errorMsg;

      // Try to parse JSON error from backend (FastAPI sends {detail: "..."})
      try {
        const errorData = JSON.parse(errorMsg);
        if (errorData.detail) {
          userFriendlyError = errorData.detail;
        }
      } catch {
        // Not JSON, use the raw error message (which already contains the backend error)
      }

      // Only apply generic messages for specific non-compilation errors
      if (errorMsg.includes('timeout') || errorMsg.includes('timed out')) {
        userFriendlyError = 'Obfuscation timed out. Try reducing obfuscation complexity or file size.';
      } else if (errorMsg.includes('network') || errorMsg.includes('Failed to fetch')) {
        userFriendlyError = 'Network error. Please check your connection and try again.';
      }
      // For compilation errors, show the FULL compiler output with line numbers

      setModal({
        type: 'error',
        title: 'Obfuscation Failed',
        message: userFriendlyError
      });
    } finally {
      setLoading(false);
      setProgress(null); 
    }
  }, [
    file, inputMode, pastedSource, obfuscationLevel, cycles, targetPlatform, entrypointCommand,
    layer1, layer2, layer3, layer4, layer2_5,
    symbolAlgorithm, symbolHashLength, symbolPrefix, symbolSalt,
    fakeLoops, indirectStdlib, indirectCustom,
    passFlattening, passSubstitution, passBogusControlFlow, passSplitBasicBlocks,
    flagLTO, flagSymbolHiding, flagOmitFramePointer, flagSpeculativeLoadHardening,
    flagO3, flagStripSymbols, flagNoBuiltin,
    buildSystem, customBuildCommand, outputBinaryPath, cmakeOptions,
    detectLanguage, countLayers, selectedRepoFile, repoSessionId, repoFileCount, repoFiles
  ]);

  const onDownloadBinary = useCallback((platform: Platform) => {
    const url = downloadUrls[platform];
    if (!url) return;
    const link = document.createElement('a');
    link.href = url;
    link.download = binaryName || `obfuscated_${platform}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [downloadUrls, binaryName]);

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">LLVM-OBFUSCATOR</h1>
          <button className="dark-toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? '☀' : '☾'}
          </button>
        </div>
        <div className="status-bar">
          <span className={`status-indicator ${serverStatus}`}>
            [{serverStatus === 'online' ? '✓' : serverStatus === 'offline' ? '✗' : '...'}] Backend: {serverStatus}
          </span>
          {detectedLanguage && <span className="lang-indicator">[{detectedLanguage.toUpperCase()}]</span>}
        </div>
      </header>

      {/* Modal */}
      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className={`modal-header ${modal.type}`}>
              <h3>{modal.title}</h3>
              <button className="modal-close" onClick={() => setModal(null)}>×</button>
            </div>
            <div className="modal-body">
              <p>{modal.message}</p>
            </div>
            <div className="modal-footer">
              <button className="modal-btn" onClick={() => setModal(null)}>
                {modal.type === 'success' ? 'OK' : 'Close'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* GitHub Modal */}
      {showGitHubModal && (
        <div className="modal-overlay" onClick={() => setShowGitHubModal(false)}>
          <div className="modal github-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>GitHub Repository</h3>
              <button className="modal-close" onClick={() => setShowGitHubModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <GitHubIntegration
                onFilesLoaded={onGitHubFilesLoaded}
                onRepoCloned={onRepoCloned}
                onError={onGitHubError}
              />
            </div>
          </div>
        </div>
      )}

      <main className="main-content">
        {/* Input Section */}
        <section className="section">
          <h2 className="section-title">[1] SOURCE INPUT</h2>
          <div className="input-section-header">
            <div className="input-mode-toggle">
              <button
                className={inputMode === 'file' ? 'active' : ''}
                onClick={() => setInputMode('file')}
              >
                FILE
              </button>
              <button
                className={inputMode === 'paste' ? 'active' : ''}
                onClick={() => setInputMode('paste')}
              >
                PASTE
              </button>
              <button
                className={inputMode === 'github' ? 'active' : ''}
                onClick={() => setInputMode('github')}
              >
                GITHUB
              </button>
            </div>
            <button
              className="github-btn"
              onClick={() => setShowGitHubModal(true)}
              title="Load from GitHub Repository"
            >
              <span className="github-logo">GitHub Logo</span>
              GitHub
            </button>
          </div>

          {inputMode === 'paste' && (
            <div className="config-grid" style={{ marginBottom: '15px' }}>
              <label>
                Load Demo Program:
                <select value={selectedDemo} onChange={(e) => onSelectDemo(e.target.value)}>
                  <option value="">-- Select Demo --</option>
                  {Object.entries(DEMO_PROGRAMS).map(([key, demo]) => (
                    <option key={key} value={key}>{demo.name}</option>
                  ))}
                </select>
              </label>
            </div>
          )}

          {inputMode === 'file' ? (
            <div className="file-input">
              <label className="file-label">
                <input type="file" accept=".c,.cpp,.cc,.cxx,.txt" onChange={onPick} />
                SELECT FILE
              </label>
              {file && <span className="file-name">{file.name}</span>}
            </div>
          ) : inputMode === 'paste' ? (
            <textarea
              className="code-input"
              placeholder="// Paste your C/C++ source code here..."
              value={pastedSource}
              onChange={(e) => setPastedSource(e.target.value)}
              rows={20}
              style={{ minHeight: '400px', fontFamily: 'monospace', fontSize: '14px' }}
            />
          ) : (
            <div className="github-input">
              {/* Fast clone mode: files are on backend */}
              {repoSessionId ? (
                <div className="github-repo-loaded">
                  <div className="repo-info">
                    <h4>📁 {repoName} ({repoBranch})</h4>
                    <p>{repoFileCount} C/C++ source files ready</p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '5px' }}>
                      ✓ Repository cloned to server (includes build system files: CMakeLists.txt, configure, etc.)
                    </p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '3px' }}>
                      ℹ️ All C/C++ files will be compiled together into a single obfuscated binary
                    </p>
                  </div>
                  <div className="github-content">
                    <div className="file-preview" style={{ width: '100%' }}>
                      <div className="file-preview-header">
                        <h5>🚀 Ready to obfuscate</h5>
                      </div>
                      <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                        <p>Repository is cloned and ready for obfuscation.</p>
                        <p style={{ fontSize: '0.9em', marginTop: '10px' }}>
                          Select your obfuscation layers above and click "Obfuscate" to begin.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : repoFiles.length > 0 ? (
                <div className="github-repo-loaded">
                  <div className="repo-info">
                    <h4>📁 {repoName} ({repoBranch})</h4>
                    <p>{repoFiles.length} files loaded</p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '5px' }}>
                      ℹ️ All C/C++ files will be compiled together into a single obfuscated binary
                    </p>
                  </div>
                  <div className="github-content">
                    <div className="file-tree-container">
                      <FileTree
                        files={repoFiles}
                        selectedFile={selectedRepoFile?.path || null}
                        onFileSelect={setSelectedRepoFile}
                      />
                    </div>
                    {selectedRepoFile && (
                      <div className="file-preview">
                        <div className="file-preview-header">
                          <h5>📄 {selectedRepoFile.path}</h5>
                        </div>
                        <textarea
                          className="code-input"
                          value={selectedRepoFile.content}
                          readOnly
                          rows={12}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="github-empty">
                  <p>No repository loaded. Click the GitHub button to load a repository.</p>
                  <button
                    className="load-github-btn"
                    onClick={() => setShowGitHubModal(true)}
                  >
                    Load GitHub Repository
                  </button>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Layer Selection */}
        <section className="section">
          <h2 className="section-title">[2] OBFUSCATION LAYERS</h2>
          <div className="layer-description">
            Select any combination of layers and their individual options
          </div>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
            <button
              className="select-all-btn"
              onClick={() => {
                const allSelected = layer1 && layer2 && layer2_5 && layer3 && layer4 &&
                  passFlattening && passSubstitution && passBogusControlFlow && passSplitBasicBlocks &&
                  flagLTO && flagSymbolHiding && flagOmitFramePointer && flagSpeculativeLoadHardening &&
                  flagO3 && flagStripSymbols && flagNoBuiltin;

                const newValue = !allSelected;
                setLayer1(newValue);
                setLayer2(newValue);
                setLayer2_5(newValue);
                setLayer3(newValue);
                setLayer4(newValue);
                setPassFlattening(newValue);
                setPassSubstitution(newValue);
                setPassBogusControlFlow(newValue);
                setPassSplitBasicBlocks(newValue);
                setFlagLTO(newValue);
                setFlagSymbolHiding(newValue);
                setFlagOmitFramePointer(newValue);
                setFlagSpeculativeLoadHardening(newValue);
                setFlagO3(newValue);
                setFlagStripSymbols(newValue);
                setFlagNoBuiltin(newValue);
              }}
            >
              {layer1 && layer2 && layer2_5 && layer3 && layer4 &&
                passFlattening && passSubstitution && passBogusControlFlow && passSplitBasicBlocks &&
                flagLTO && flagSymbolHiding && flagOmitFramePointer && flagSpeculativeLoadHardening &&
                flagO3 && flagStripSymbols && flagNoBuiltin
                ? 'Deselect All' : 'Select All'}
            </button>
          </div>
          <div className="layers-grid">
            {/* Layer 1: Symbol Obfuscation */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer1}
                onChange={(e) => handleLayerChange(1, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 1] Symbol Obfuscation (PRE-COMPILE, 1st)
                <small>Cryptographic hash renaming of all symbols</small>
              </span>
            </label>

            {layer1 && (
              <div className="layer-config">
                <label>
                  Symbol Hash Algorithm:
                  <select value={symbolAlgorithm} onChange={(e) => setSymbolAlgorithm(e.target.value)}>
                    <option value="sha256">SHA256</option>
                    <option value="blake2b">BLAKE2B</option>
                    <option value="siphash">SipHash</option>
                  </select>
                </label>
                <label>
                  Hash Length (8-32):
                  <input
                    type="number"
                    min="8"
                    max="32"
                    value={symbolHashLength}
                    onChange={(e) => setSymbolHashLength(parseInt(e.target.value) || 12)}
                  />
                </label>
                <label>
                  Prefix Style:
                  <select value={symbolPrefix} onChange={(e) => setSymbolPrefix(e.target.value)}>
                    <option value="none">none</option>
                    <option value="typed">typed (f_, v_)</option>
                    <option value="underscore">underscore (_)</option>
                  </select>
                </label>
              </div>
            )}

            {/* Layer 2: String Encryption */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer2}
                onChange={(e) => handleLayerChange(2, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 2] String Encryption (PRE-COMPILE, 2nd)
                <small>XOR encryption of string literals + runtime decryption</small>
              </span>
            </label>

            {layer2 && (
              <div className="layer-config">
                <label>
                  Fake Loops (0-50):
                  <input
                    type="number"
                    min="0"
                    max="50"
                    value={fakeLoops}
                    onChange={(e) => setFakeLoops(parseInt(e.target.value) || 0)}
                  />
                </label>
              </div>
            )}

            {/* Layer 2.5: Indirect Call Obfuscation */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer2_5}
                onChange={(e) => setLayer2_5(e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 2.5] Indirect Call Obfuscation (PRE-COMPILE, 2.5)
                <small>Convert direct function calls to indirect calls via function pointers</small>
              </span>
            </label>

            {layer2_5 && (
              <div className="layer-config">
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={indirectStdlib}
                    onChange={(e) => setIndirectStdlib(e.target.checked)}
                  />
                  Obfuscate stdlib functions (printf, malloc, etc.)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={indirectCustom}
                    onChange={(e) => setIndirectCustom(e.target.checked)}
                  />
                  Obfuscate custom functions
                </label>
              </div>
            )}

            {/* Layer 3: OLLVM Passes */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer3}
                onChange={(e) => handleLayerChange(3, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 3] OLLVM Passes (COMPILE, 3rd - Optional)
                <small>Select individual control flow obfuscation passes</small>
              </span>
            </label>

            {layer3 && (
              <div className="layer-config">
                <button
                  className="select-all-btn"
                  style={{ marginBottom: '10px', fontSize: '0.9em' }}
                  onClick={() => {
                    const allPassesSelected = passFlattening && passSubstitution &&
                      passBogusControlFlow && passSplitBasicBlocks;
                    const newValue = !allPassesSelected;
                    setPassFlattening(newValue);
                    setPassSubstitution(newValue);
                    setPassBogusControlFlow(newValue);
                    setPassSplitBasicBlocks(newValue);
                  }}
                >
                  {passFlattening && passSubstitution && passBogusControlFlow && passSplitBasicBlocks
                    ? 'Deselect All' : 'Select All'}
                </button>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passFlattening}
                    onChange={(e) => setPassFlattening(e.target.checked)}
                  />
                  Control Flow Flattening
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passSubstitution}
                    onChange={(e) => setPassSubstitution(e.target.checked)}
                  />
                  Instruction Substitution
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passBogusControlFlow}
                    onChange={(e) => setPassBogusControlFlow(e.target.checked)}
                  />
                  Bogus Control Flow
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passSplitBasicBlocks}
                    onChange={(e) => setPassSplitBasicBlocks(e.target.checked)}
                  />
                  Split Basic Blocks
                </label>
                <label>
                  Cycle Count (1-5):
                  <input
                    type="number"
                    min="1"
                    max="5"
                    value={cycles}
                    onChange={(e) => setCycles(parseInt(e.target.value) || 1)}
                  />
                </label>
              </div>
            )}

            {/* Layer 4: Compiler Flags */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer4}
                onChange={(e) => handleLayerChange(4, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 4] Compiler Flags (COMPILE, FINAL)
                <small>Select individual LLVM optimization and hardening flags</small>
              </span>
            </label>

            {layer4 && (
              <div className="layer-config">
                <button
                  className="select-all-btn"
                  style={{ marginBottom: '10px', fontSize: '0.9em' }}
                  onClick={() => {
                    const allFlagsSelected = flagSymbolHiding &&
                      flagOmitFramePointer && flagSpeculativeLoadHardening &&
                      flagO3 && flagStripSymbols && flagNoBuiltin && flagLTO;
                    const newValue = !allFlagsSelected;
                    setFlagSymbolHiding(newValue);
                    setFlagOmitFramePointer(newValue);
                    setFlagSpeculativeLoadHardening(newValue);
                    setFlagO3(newValue);
                    setFlagStripSymbols(newValue);
                    setFlagNoBuiltin(newValue);
                    setFlagLTO(newValue);
                  }}
                >
                  {flagSymbolHiding && flagOmitFramePointer && flagSpeculativeLoadHardening &&
                    flagO3 && flagStripSymbols && flagNoBuiltin && flagLTO
                    ? 'Deselect All' : 'Select All'}
                </button>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagSymbolHiding}
                    onChange={(e) => setFlagSymbolHiding(e.target.checked)}
                  />
                  Symbol Hiding (-fvisibility=hidden)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagOmitFramePointer}
                    onChange={(e) => setFlagOmitFramePointer(e.target.checked)}
                  />
                  Remove Frame Pointer (-fomit-frame-pointer)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagSpeculativeLoadHardening}
                    onChange={(e) => setFlagSpeculativeLoadHardening(e.target.checked)}
                  />
                  Speculative Load Hardening
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagO3}
                    onChange={(e) => setFlagO3(e.target.checked)}
                  />
                  Maximum Optimization (-O3)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagStripSymbols}
                    onChange={(e) => setFlagStripSymbols(e.target.checked)}
                  />
                  Strip Symbols (-Wl,-s)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagNoBuiltin}
                    onChange={(e) => setFlagNoBuiltin(e.target.checked)}
                  />
                  Disable Built-in Functions (-fno-builtin)
                </label>
                <label
                  className="sub-option"
                  title={layer3 ? "LTO is incompatible with OLLVM passes (LLVM version mismatch: bundled v22 vs system v19)" : ""}
                  style={{
                    opacity: layer3 ? 0.5 : 1,
                    cursor: layer3 ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                      type="checkbox"
                      checked={flagLTO}
                      onChange={(e) => setFlagLTO(e.target.checked)}
                      disabled={layer3}
                    />
                    Link-Time Optimization (-flto)
                  </div>
                  {layer3 && <span style={{ color: '#ff6b6b', fontSize: '0.8em', marginLeft: '20px' }}>⚠ Incompatible with Layer 3</span>}
                </label>
              </div>
            )}
          </div>
        </section>

        {/* Configuration */}
        <section className="section">
          <h2 className="section-title">[3] CONFIGURATION</h2>
          <div className="config-grid">
            <label>
              Obfuscation Level:
              <select
                value={obfuscationLevel}
                onChange={(e) => setObfuscationLevel(parseInt(e.target.value))}
              >
                <option value="1">1 - Minimal</option>
                <option value="2">2 - Low</option>
                <option value="3">3 - Medium</option>
                <option value="4">4 - High</option>
                <option value="5">5 - Maximum</option>
              </select>
            </label>

            <label>
              Target Platform:
              <select
                value={targetPlatform}
                onChange={(e) => setTargetPlatform(e.target.value as Platform)}
              >
                <option value="linux">Linux</option>
                <option value="windows">Windows</option>
                <option value="macos">macOS</option>
              </select>
            </label>

            <label>
              Build System:
              <select
                value={buildSystem}
                onChange={(e) => setBuildSystem(e.target.value as BuildSystem)}
                title="How to compile the project. Use 'Simple' for single files, or select the project's build system for complex projects."
              >
                <option value="simple">Simple (Direct Compilation)</option>
                <option value="cmake">CMake</option>
                <option value="make">Make</option>
                <option value="autotools">Autotools (configure + make)</option>
                <option value="custom">Custom Command</option>
              </select>
            </label>

            {buildSystem === 'custom' && (
              <label>
                Custom Build Command:
                <input
                  type="text"
                  placeholder="e.g., meson build && ninja -C build"
                  value={customBuildCommand}
                  onChange={(e) => setCustomBuildCommand(e.target.value)}
                  title="The exact command to build the project"
                />
              </label>
            )}

            {buildSystem !== 'simple' && (
              <label>
                Output Binary Path (optional):
                <input
                  type="text"
                  placeholder="e.g., build/bin/curl"
                  value={outputBinaryPath}
                  onChange={(e) => setOutputBinaryPath(e.target.value)}
                  title="Hint for where to find the compiled binary after build"
                />
              </label>
            )}

            {buildSystem === 'cmake' && (
              <label>
                CMake Options (optional):
                <input
                  type="text"
                  placeholder="e.g., -DBUILD_TESTING=OFF -DCURL_USE_LIBPSL=OFF"
                  value={cmakeOptions}
                  onChange={(e) => setCmakeOptions(e.target.value)}
                  title="Extra CMake flags to disable features or customize the build"
                />
              </label>
            )}

            {buildSystem === 'simple' && (
              <label>
                Entrypoint Command:
                <input
                  type="text"
                  placeholder="./a.out or gcc main.c -o out && ./out"
                  value={entrypointCommand}
                  onChange={(e) => setEntrypointCommand(e.target.value)}
                  title="Command to run the compiled binary (e.g., ./a.out, make && ./bin/app)"
                />
              </label>
            )}
          </div>

          {buildSystem !== 'simple' && (
            <div className="build-system-info" style={{
              marginTop: '15px',
              padding: '12px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '4px',
              fontSize: '0.9em',
              color: 'var(--text-secondary)'
            }}>
              <strong>Custom Build Mode:</strong> Source-level obfuscation (Layers 1, 2, 2.5) will be applied in-place to all C/C++ files before building.
              Compiler-level obfuscation (Layers 3, 4) will be injected via CC/CXX environment variables.
              {buildSystem === 'cmake' && <span> Build command: <code>cmake -B build && cmake --build build</code></span>}
              {buildSystem === 'make' && <span> Build command: <code>make</code></span>}
              {buildSystem === 'autotools' && <span> Build command: <code>./configure && make</code></span>}
            </div>
          )}
        </section>

        {/* Submit */}
        <section className="section">
          <h2 className="section-title">[4] EXECUTE</h2>
          <button
            className="submit-btn"
            onClick={onSubmit}
            disabled={loading || (
              inputMode === 'file' ? !file :
              inputMode === 'paste' ? pastedSource.trim().length === 0 :
              inputMode === 'github' ? (!repoSessionId && repoFiles.length === 0) : true
            )}
          >
            {loading ? 'PROCESSING...' : '► OBFUSCATE'}
          </button>

          {progress && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
              </div>
              <div className="progress-text">{progress.message} ({progress.percent}%)</div>
            </div>
          )}

          {(downloadUrls.linux || downloadUrls.windows || downloadUrls.macos) && (
            <div className="download-section">
              <h3>Download Obfuscated Binary:</h3>
              <div className="download-buttons">
                {(Object.keys(downloadUrls) as Platform[]).map(platform => (
                  downloadUrls[platform] && (
                    <button
                      key={platform}
                      className="download-btn"
                      onClick={() => onDownloadBinary(platform)}
                    >
                      ⬇ {platform.toUpperCase()}
                    </button>
                  )
                ))}
              </div>
              {binaryName && <div className="binary-name">File: {binaryName}</div>}

              {jobId && (
                <div style={{ marginTop: '15px' }}>
                  <h3>Download Report:</h3>
                  <div className="download-buttons">
                    <button
                      className="download-btn"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = `/api/report/${jobId}?fmt=markdown`;
                        link.download = `report_${jobId}.md`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      📄 MARKDOWN
                    </button>
                    <button
                      className="download-btn"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = `/api/report/${jobId}?fmt=json`;
                        link.download = `report_${jobId}.json`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      📊 JSON
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Report */}
        {report && (
          <section className="section report-section">
            <h2 className="section-title">[5] OBFUSCATION REPORT</h2>

            <div className="report-grid">
              {/* Input Parameters */}
              <div className="report-block">
                <h3>INPUT PARAMETERS</h3>
                <div className="report-item">Source: {report.input_parameters?.source_file || 'N/A'}</div>
                <div className="report-item">Platform: {report.input_parameters?.platform || 'N/A'}</div>
                <div className="report-item">Level: {report.input_parameters?.obfuscation_level ?? 'N/A'}</div>
                <div className="report-item">Timestamp: {report.input_parameters?.timestamp || 'N/A'}</div>
                <div className="report-item">Compiler Flags: {report.input_parameters?.compiler_flags?.join(' ') || 'None'}</div>
              </div>

              {/* Before/After Comparison */}
              {report.baseline_metrics && report.comparison && (
                <div className="report-block comparison-block" style={{ gridColumn: '1 / -1' }}>
                  <h3>⚖️ BEFORE / AFTER COMPARISON</h3>
                  <div className="comparison-grid-ui">
                    <div className="comparison-card">
                      <h4>📦 File Size</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{(report.baseline_metrics.file_size / 1024).toFixed(2)} KB</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{(report.output_attributes.file_size / 1024).toFixed(2)} KB</span>
                      </div>
                      <div className={`comparison-change ${report.comparison.size_change_percent > 0 ? 'negative' : report.comparison.size_change_percent < 0 ? 'positive' : 'neutral'}`}>
                        {report.comparison.size_change_percent > 0 ? '▲' : report.comparison.size_change_percent < 0 ? '▼' : '='} {Math.abs(report.comparison.size_change_percent).toFixed(2)}%
                        <small>({report.comparison.size_change > 0 ? '+' : ''}{report.comparison.size_change} bytes)</small>
                      </div>
                    </div>

                    <div className="comparison-card">
                      <h4>🏷️ Symbol Count</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{report.baseline_metrics.symbols_count}</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{report.output_attributes.symbols_count}</span>
                      </div>
                      <div className="comparison-change positive">
                        ✓ {report.comparison.symbols_removed} removed ({report.comparison.symbols_removed_percent.toFixed(1)}%)
                      </div>
                      <div className="progress-bar-container">
                        <div className="progress-bar-fill" style={{ width: `${Math.min(100, Math.abs(report.comparison.symbols_removed_percent))}%` }}>
                          {report.comparison.symbols_removed_percent.toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="comparison-card">
                      <h4>⚙️ Function Count</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{report.baseline_metrics.functions_count}</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{report.output_attributes.functions_count}</span>
                      </div>
                      <div className="comparison-change positive">
                        ✓ {report.comparison.functions_removed} hidden ({report.comparison.functions_removed_percent.toFixed(1)}%)
                      </div>
                      <div className="progress-bar-container">
                        <div className="progress-bar-fill" style={{ width: `${Math.min(100, Math.abs(report.comparison.functions_removed_percent))}%` }}>
                          {report.comparison.functions_removed_percent.toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="comparison-card">
                      <h4>🔒 Binary Entropy</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{report.baseline_metrics.entropy.toFixed(3)}</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{report.output_attributes.entropy.toFixed(3)}</span>
                      </div>
                      <div className="comparison-change positive">
                        ✓ +{report.comparison.entropy_increase.toFixed(3)} ({report.comparison.entropy_increase_percent > 0 ? '+' : ''}{report.comparison.entropy_increase_percent.toFixed(1)}%)
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Output Attributes */}
              {report.output_attributes && (
                <div className="report-block">
                  <h3>OUTPUT ATTRIBUTES</h3>
                  <div className="report-item">Size: {report.output_attributes.file_size ? (report.output_attributes.file_size / 1024).toFixed(2) : '0'} KB</div>
                  <div className="report-item">Format: {report.output_attributes.binary_format || 'Unknown'}</div>
                  <div className="report-item">Symbols: {report.output_attributes.symbols_count ?? 0}</div>
                  <div className="report-item">Functions: {report.output_attributes.functions_count ?? 0}</div>
                  <div className="report-item">Entropy: {report.output_attributes.entropy?.toFixed(3) ?? 'N/A'}</div>
                  <div className="report-item">Methods: {report.output_attributes.obfuscation_methods?.join(', ') || 'None'}</div>
                </div>
              )}

              {/* Bogus Code */}
              {report.bogus_code_info && (
                <div className="report-block">
                  <h3>BOGUS CODE GENERATION</h3>
                  <div className="report-item">Dead Blocks: {report.bogus_code_info.dead_code_blocks ?? 0}</div>
                  <div className="report-item">Opaque Predicates: {report.bogus_code_info.opaque_predicates ?? 0}</div>
                  <div className="report-item">Junk Instructions: {report.bogus_code_info.junk_instructions ?? 0}</div>
                  <div className="report-item">Code Bloat: {report.bogus_code_info.code_bloat_percentage ?? 0}%</div>
                </div>
              )}

              {/* Cycles */}
              {report.cycles_completed && (
                <div className="report-block">
                  <h3>OBFUSCATION CYCLES</h3>
                  <div className="report-item">Total: {report.cycles_completed.total_cycles ?? 0}</div>
                  {report.cycles_completed.per_cycle_metrics?.map((cycle, idx) => (
                    <div key={idx} className="report-item">
                      Cycle {cycle.cycle}: {cycle.passes_applied?.join(', ') || 'N/A'} ({cycle.duration_ms ?? 0}ms)
                    </div>
                  ))}
                </div>
              )}

              {/* String Obfuscation */}
              {report.string_obfuscation && (
                <div className="report-block">
                  <h3>STRING ENCRYPTION</h3>
                  <div className="report-item">Total Strings: {report.string_obfuscation.total_strings ?? 0}</div>
                  <div className="report-item">Encrypted: {report.string_obfuscation.encrypted_strings ?? 0}</div>
                  <div className="report-item">Method: {report.string_obfuscation.encryption_method || 'None'}</div>
                  <div className="report-item">Rate: {report.string_obfuscation.encryption_percentage?.toFixed(1) ?? '0.0'}%</div>
                </div>
              )}

              {/* Fake Loops */}
              {report.fake_loops_inserted && (
                <div className="report-block">
                  <h3>FAKE LOOPS</h3>
                  <div className="report-item">Count: {report.fake_loops_inserted.count ?? 0}</div>
                  {report.fake_loops_inserted.locations && report.fake_loops_inserted.locations.length > 0 && (
                    <div className="report-item">Locations: {report.fake_loops_inserted.locations.join(', ')}</div>
                  )}
                </div>
              )}

              {/* Symbol Obfuscation */}
              {report.symbol_obfuscation && (
                <div className="report-block">
                  <h3>SYMBOL OBFUSCATION</h3>
                  <div className="report-item">Enabled: {report.symbol_obfuscation.enabled ? 'Yes' : 'No'}</div>
                  {report.symbol_obfuscation.enabled && report.symbol_obfuscation.symbols_obfuscated && (
                    <>
                      <div className="report-item">Symbols Renamed: {report.symbol_obfuscation.symbols_obfuscated}</div>
                      <div className="report-item">Algorithm: {report.symbol_obfuscation.algorithm || 'N/A'}</div>
                    </>
                  )}
                </div>
              )}

              {/* Metrics */}
              <div className="report-block">
                <h3>EFFECTIVENESS METRICS</h3>
                <div className="report-item">Obfuscation Score: {report.obfuscation_score ?? 0}/100</div>
                <div className="report-item">Symbol Reduction: {report.symbol_reduction ?? 0}%</div>
                <div className="report-item">Function Reduction: {report.function_reduction ?? 0}%</div>
                <div className="report-item">Size Change: {(report.size_reduction ?? 0) > 0 ? '+' : ''}{report.size_reduction ?? 0}%</div>
                <div className="report-item">Entropy Increase: {report.entropy_increase ?? 0}%</div>
                <div className="report-item">Est. RE Effort: {report.estimated_re_effort || 'N/A'}</div>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>LLVM-OBFUSCATOR :: Research-backed binary hardening</p>
      </footer>
    </div>
  );
}

export default App;
