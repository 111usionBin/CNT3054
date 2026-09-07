# ── Configuration ────────────────────────────────────────────────────────────
GIT_NAME="Your Name"
GIT_EMAIL="your.email@example.com"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Install Homebrew only if missing
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH (only if not already in .zprofile)
    if ! grep -q "brew shellenv" ~/.zprofile; then
        (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> ~/.zprofile
    fi
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "✅ Homebrew is already installed."
fi

# 2. Install Git only if missing
if ! command -v git &> /dev/null; then
    echo "🔧 Installing Git via Homebrew..."
    brew install git
else
    echo "✅ Git is already installed ($(git --version))."
fi

# 3. Install VS Code only if not in Applications
if [ ! -d "/Applications/Visual Studio Code.app" ]; then
    echo "💻 Installing Visual Studio Code..."
    brew install --cask visual-studio-code
else
    echo "✅ VS Code is already in your Applications folder."
fi

# 4. Install UV (Python manager) only if missing
if ! command -v uv &> /dev/null; then
    echo "⚡ Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "✅ UV is already installed."
fi

# 5. Set Git identity ONLY if not already set
if [ "$(git config --global user.email)" = "" ]; then
    echo "👤 Setting Git identity..."
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
else
    echo "✅ Git identity already configured: $(git config --global user.email)"
fi

# 6. Install VS Code extensions (Safe to run multiple times)
echo "🧩 Updating VS Code extensions..."
code --install-extension ms-python.python \
     --install-extension ms-python.vscode-python-envs \
     --install-extension ms-toolsai.jupyter \
     --install-extension ms-toolsai.datawrangler \
     --install-extension esbenp.prettier-vscode \
     --install-extension GitHub.copilot --force

# 7. Project initialization
PROJECT_DIR=~/Developer/my-first-project-2026
if [ ! -d "$PROJECT_DIR" ]; then
    echo "📁 Creating project at $PROJECT_DIR..."
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    uv init --python 3.11
    uv add pandas ipykernel
else
    echo "✅ Project directory already exists."
    cd "$PROJECT_DIR"
fi

# 8. Open project in VS Code
echo "🚀 Opening project in VS Code..."
code .
