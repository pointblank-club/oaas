import React, { useState, useEffect } from 'react';

interface Repository {
  name: string;
  full_name: string;
  html_url: string;
  private: boolean;
  default_branch: string;
  language: string | null;
  description: string | null;
}

interface RepoFile {
  path: string;
  content: string;
  is_binary: boolean;
}

interface RepoData {
  files: RepoFile[];
  total_files: number;
  repo_name: string;
  branch: string;
}

interface GitHubIntegrationProps {
  onFilesLoaded: (files: RepoFile[], repoName: string, branch: string) => void;
  onError: (error: string) => void;
}

export const GitHubIntegration: React.FC<GitHubIntegrationProps> = ({ onFilesLoaded, onError }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authState, setAuthState] = useState<string | null>(null);
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<Repository | null>(null);
  const [branches, setBranches] = useState<string[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string>('');
  const [publicRepoUrl, setPublicRepoUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [showPublicInput, setShowPublicInput] = useState(false);
  const [githubEnabled, setGithubEnabled] = useState(false);

  useEffect(() => {
    // Check if GitHub OAuth is enabled
    fetch('/api/capabilities')
      .then(res => res.json())
      .then(data => {
        setGithubEnabled(data.github_oauth?.enabled || false);
      })
      .catch(err => console.error('Failed to check GitHub capabilities:', err));

    // Check for OAuth callback
    const urlParams = new URLSearchParams(window.location.search);
    const githubAuth = urlParams.get('github_auth');
    const error = urlParams.get('error');
    
    if (error) {
      onError(`GitHub authentication failed: ${error}`);
      window.history.replaceState({}, document.title, window.location.pathname);
    } else if (githubAuth === 'success') {
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
      // Check authentication status
      checkAuthStatus();
    } else {
      // Check if already authenticated
      checkAuthStatus();
    }
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('/api/github/status', {
        credentials: 'include'
      });
      const data = await response.json();
      
      if (data.authenticated) {
        setIsAuthenticated(true);
        loadRepositories();
      }
    } catch (err) {
      console.error('Failed to check auth status:', err);
    }
  };

  const initiateGitHubAuth = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/github/login', {
        credentials: 'include'
      });
      const data = await response.json();
      
      if (response.ok) {
        // Redirect to GitHub OAuth
        window.location.href = data.auth_url;
      } else {
        onError('Failed to initiate GitHub authentication');
      }
    } catch (err) {
      onError('Failed to connect to GitHub');
    } finally {
      setLoading(false);
    }
  };

  const disconnectGitHub = async () => {
    try {
      await fetch('/api/github/disconnect', {
        method: 'POST',
        credentials: 'include'
      });
      setIsAuthenticated(false);
      setRepositories([]);
      setSelectedRepo(null);
      setBranches([]);
    } catch (err) {
      onError('Failed to disconnect GitHub');
    }
  };

  const loadRepositories = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/github/repos', {
        credentials: 'include'
      });
      const data = await response.json();
      
      if (response.ok) {
        setRepositories(data.repositories);
      } else if (response.status === 401) {
        setIsAuthenticated(false);
        onError('GitHub session expired. Please reconnect.');
      } else {
        onError('Failed to load repositories');
      }
    } catch (err) {
      onError('Failed to load repositories');
    } finally {
      setLoading(false);
    }
  };

  const loadBranches = async (repoUrl: string, useAuth: boolean = true) => {
    try {
      setLoading(true);
      const url = `/api/github/repo/branches?repo_url=${encodeURIComponent(repoUrl)}`;
      const response = await fetch(url, {
        credentials: useAuth ? 'include' : 'omit'
      });
      const data = await response.json();
      
      if (response.ok) {
        setBranches(data.branches);
        setSelectedBranch(data.branches.includes('main') ? 'main' : data.branches[0] || '');
      } else if (response.status === 401 && useAuth) {
        setIsAuthenticated(false);
        onError('GitHub session expired. Please reconnect.');
      } else {
        onError('Failed to load branches');
      }
    } catch (err) {
      onError('Failed to load branches');
    } finally {
      setLoading(false);
    }
  };

  const loadRepoFiles = async (repoUrl: string, branch: string, useAuth: boolean = true) => {
    try {
      setLoading(true);
      const payload = {
        repo_url: repoUrl,
        branch: branch,
      };
      
      const response = await fetch('/api/github/repo/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        credentials: useAuth ? 'include' : 'omit'
      });
      
      const data: RepoData = await response.json();
      
      if (response.ok) {
        onFilesLoaded(data.files, data.repo_name, data.branch);
      } else if (response.status === 401 && useAuth) {
        setIsAuthenticated(false);
        onError('GitHub session expired. Please reconnect.');
      } else {
        onError('Failed to load repository files');
      }
    } catch (err) {
      onError('Failed to load repository files');
    } finally {
      setLoading(false);
    }
  };

  const handleRepoSelect = async (repo: Repository) => {
    setSelectedRepo(repo);
    await loadBranches(repo.html_url, true); // Use auth for OAuth repos
  };

  const handlePublicRepoSubmit = async () => {
    if (!publicRepoUrl.trim()) {
      onError('Please enter a repository URL');
      return;
    }
    
    if (!publicRepoUrl.includes('github.com')) {
      onError('Please enter a valid GitHub repository URL');
      return;
    }
    
    await loadBranches(publicRepoUrl, false); // No auth for public repos
  };

  const handleLoadFiles = async () => {
    const repoUrl = selectedRepo ? selectedRepo.html_url : publicRepoUrl;
    const useAuth = !!selectedRepo; // Use auth only for OAuth repos
    
    if (!repoUrl || !selectedBranch) {
      onError('Please select a repository and branch');
      return;
    }
    
    await loadRepoFiles(repoUrl, selectedBranch, useAuth);
  };

  // if (!githubEnabled) {
  //   return (
  //     <div className="github-integration">
  //       <div className="github-disabled">
  //         <p>GitHub integration is not configured. Please set up GitHub OAuth credentials.</p>
  //       </div>
  //     </div>
  //   );
  // }

  return (
    <div className="github-integration">
      <div className="github-header">
        <h3>🐙 GitHub Integration</h3>
        <div className="github-options">
          <button
            className={`github-option-btn ${!showPublicInput ? 'active' : ''}`}
            onClick={() => setShowPublicInput(false)}
          >
            OAuth Repos
          </button>
          <button
            className={`github-option-btn ${showPublicInput ? 'active' : ''}`}
            onClick={() => setShowPublicInput(true)}
          >
            Public URL
          </button>
        </div>
      </div>

      {!showPublicInput ? (
        <div className="github-oauth-section">
          {!isAuthenticated ? (
            <div className="github-auth">
              <p>Connect to GitHub to access your repositories</p>
              <button
                className="github-auth-btn"
                onClick={initiateGitHubAuth}
                disabled={loading}
              >
                {loading ? 'Connecting...' : '🔗 Connect GitHub'}
              </button>
            </div>
          ) : (
            <div className="github-repos">
              <div className="github-connected">
                <p>✅ Connected to GitHub</p>
                <button
                  className="github-disconnect-btn"
                  onClick={disconnectGitHub}
                  disabled={loading}
                >
                  Disconnect
                </button>
              </div>
              <div className="repo-selector">
                <label>
                  Select Repository:
                  <select
                    value={selectedRepo?.full_name || ''}
                    onChange={(e) => {
                      const repo = repositories.find(r => r.full_name === e.target.value);
                      if (repo) handleRepoSelect(repo);
                    }}
                    disabled={loading}
                  >
                    <option value="">-- Select Repository --</option>
                    {repositories.map(repo => (
                      <option key={repo.full_name} value={repo.full_name}>
                        {repo.full_name} {repo.private ? '🔒' : '🌐'} ({repo.language || 'Unknown'})
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="github-public-section">
          <div className="public-repo-input">
            <label>
              Public Repository URL:
              <input
                type="text"
                placeholder="https://github.com/owner/repo"
                value={publicRepoUrl}
                onChange={(e) => setPublicRepoUrl(e.target.value)}
                disabled={loading}
              />
            </label>
            <button
              className="load-branches-btn"
              onClick={handlePublicRepoSubmit}
              disabled={loading || !publicRepoUrl.trim()}
            >
              {loading ? 'Loading...' : 'Load Branches'}
            </button>
          </div>
        </div>
      )}

      {branches.length > 0 && (
        <div className="branch-selector">
          <label>
            Select Branch:
            <select
              value={selectedBranch}
              onChange={(e) => setSelectedBranch(e.target.value)}
              disabled={loading}
            >
              {branches.map(branch => (
                <option key={branch} value={branch}>
                  {branch}
                </option>
              ))}
            </select>
          </label>
          <button
            className="load-files-btn"
            onClick={handleLoadFiles}
            disabled={loading || !selectedBranch}
          >
            {loading ? 'Loading Files...' : '📁 Load Repository Files'}
          </button>
        </div>
      )}
    </div>
  );
};