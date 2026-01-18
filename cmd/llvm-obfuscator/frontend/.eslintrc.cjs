module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'node_modules'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
  plugins: ['react', 'react-hooks'],
  globals: {
    React: 'readonly',
    NodeJS: 'readonly',
  },
  rules: {
    'react/prop-types': 'off',
    'no-unused-vars': 'warn',
    'react/no-unescaped-entities': 'off',
    'no-undef': 'off', // TypeScript handles this
    'react/no-unknown-property': ['error', { ignore: ['directory', 'webkitdirectory'] }],
    'react-hooks/exhaustive-deps': 'warn',
  },
};
