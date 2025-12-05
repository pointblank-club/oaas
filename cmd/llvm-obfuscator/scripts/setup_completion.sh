#!/bin/bash
# Setup script for CLI tab completion

# Detect shell
SHELL_NAME=$(basename "$SHELL")

echo "Setting up tab completion for OAAS obfuscator..."
echo "Detected shell: $SHELL_NAME"

# Get the command name (try both llvm-obfuscate and obfuscate)
if command -v llvm-obfuscate &> /dev/null; then
    CMD_NAME="llvm-obfuscate"
elif command -v obfuscate &> /dev/null; then
    CMD_NAME="obfuscate"
else
    echo "Error: llvm-obfuscate or obfuscate command not found"
    echo "Please install the package first: pip install -e ."
    exit 1
fi

case "$SHELL_NAME" in
    bash)
        COMPLETION_SCRIPT=$(python -m cli.obfuscate --show-completion bash 2>/dev/null || echo "")
        if [ -n "$COMPLETION_SCRIPT" ]; then
            echo "$COMPLETION_SCRIPT" >> ~/.bash_completion
            echo "Bash completion installed! Run: source ~/.bash_completion"
        else
            echo "Run this command and add to ~/.bashrc:"
            echo "eval \"\$(_${CMD_NAME^^}_COMPLETE=bash_source $CMD_NAME)\""
        fi
        ;;
    zsh)
        COMPLETION_SCRIPT=$(python -m cli.obfuscate --show-completion zsh 2>/dev/null || echo "")
        if [ -n "$COMPLETION_SCRIPT" ]; then
            echo "$COMPLETION_SCRIPT" >> ~/.zshrc
            echo "Zsh completion installed! Run: source ~/.zshrc"
        else
            echo "Run this command and add to ~/.zshrc:"
            echo "eval \"\$(_${CMD_NAME^^}_COMPLETE=zsh_source $CMD_NAME)\""
        fi
        ;;
    fish)
        COMPLETION_DIR="$HOME/.config/fish/completions"
        mkdir -p "$COMPLETION_DIR"
        python -m cli.obfuscate --show-completion fish > "$COMPLETION_DIR/${CMD_NAME}.fish" 2>/dev/null || {
            echo "Run this command:"
            echo "eval (env _${CMD_NAME^^}_COMPLETE=fish_source $CMD_NAME)"
        }
        echo "Fish completion installed!"
        ;;
    *)
        echo "Unsupported shell: $SHELL_NAME"
        echo "Manual setup required. See CLI_USAGE.md for instructions."
        ;;
esac

echo ""
echo "Tab completion setup complete!"
echo "Restart your terminal or run: source ~/.${SHELL_NAME}rc"

