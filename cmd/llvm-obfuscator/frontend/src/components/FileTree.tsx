import React, { useState, useMemo } from 'react';

interface RepoFile {
  path: string;
  content: string;
  is_binary: boolean;
}

interface FileNode {
  name: string;
  path: string;
  isDirectory: boolean;
  children: FileNode[];
  file?: RepoFile;
}

interface FileTreeProps {
  files: RepoFile[];
  selectedFile: string | null;
  onFileSelect: (file: RepoFile) => void;
}

export const FileTree: React.FC<FileTreeProps> = ({ files, selectedFile, onFileSelect }) => {
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set(['']));

  const fileTree = useMemo(() => {
    const root: FileNode = {
      name: '',
      path: '',
      isDirectory: true,
      children: [],
    };

    files.forEach(file => {
      const parts = file.path.split('/');
      let current = root;

      // Create directory structure
      for (let i = 0; i < parts.length - 1; i++) {
        const dirName = parts[i];
        const dirPath = parts.slice(0, i + 1).join('/');
        
        let dir = current.children.find(child => child.name === dirName && child.isDirectory);
        if (!dir) {
          dir = {
            name: dirName,
            path: dirPath,
            isDirectory: true,
            children: [],
          };
          current.children.push(dir);
        }
        current = dir;
      }

      // Add file
      const fileName = parts[parts.length - 1];
      current.children.push({
        name: fileName,
        path: file.path,
        isDirectory: false,
        children: [],
        file: file,
      });
    });

    
    const sortChildren = (node: FileNode) => {
      node.children.sort((a, b) => {
        if (a.isDirectory && !b.isDirectory) return -1;
        if (!a.isDirectory && b.isDirectory) return 1;
        return a.name.localeCompare(b.name);
      });
      node.children.forEach(sortChildren);
    };

    sortChildren(root);
    return root;
  }, [files]);

  const toggleDirectory = (path: string) => {
    const newExpanded = new Set(expandedDirs);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedDirs(newExpanded);
  };

  const renderNode = (node: FileNode, depth: number = 0): React.ReactNode => {
    if (node.isDirectory) {
      const isExpanded = expandedDirs.has(node.path);
      return (
        <div key={node.path} className="tree-directory">
          <div
            className="tree-directory-header"
            style={{ paddingLeft: `${depth * 20}px` }}
            onClick={() => toggleDirectory(node.path)}
          >
            <span className="tree-icon">
              {isExpanded ? '📂' : '📁'}
            </span>
            <span className="tree-name">{node.name || 'Repository Root'}</span>
            <span className="tree-count">({node.children.length})</span>
          </div>
          {isExpanded && (
            <div className="tree-children">
              {node.children.map(child => renderNode(child, depth + 1))}
            </div>
          )}
        </div>
      );
    } else {
      const isSelected = selectedFile === node.path;
      const fileExt = node.name.split('.').pop()?.toLowerCase() || '';
      const isSourceFile = ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++'].includes(fileExt);
      
      return (
        <div
          key={node.path}
          className={`tree-file ${isSelected ? 'selected' : ''} ${isSourceFile ? 'source-file' : ''}`}
          style={{ paddingLeft: `${depth * 20 + 20}px` }}
          onClick={() => node.file && onFileSelect(node.file)}
        >
          <span className="tree-icon">
            {isSourceFile ? '📄' : '📝'}
          </span>
          <span className="tree-name">{node.name}</span>
          {isSourceFile && <span className="tree-badge">C/C++</span>}
        </div>
      );
    }
  };

  if (files.length === 0) {
    return (
      <div className="file-tree-empty">
        <p>No files loaded</p>
      </div>
    );
  }

  return (
    <div className="file-tree">
      <div className="file-tree-header">
        <h4>📁 Repository Files ({files.length})</h4>
      </div>
      <div className="file-tree-content">
        {renderNode(fileTree)}
      </div>
    </div>
  );
};