if ($PSVersionTable.PSVersion.Major -lt 5 -or ($PSVersionTable.PSVersion.Major -eq 5 -and $PSVersionTable.PSVersion.Minor -lt 1)) {
    throw "PowerShell 5.1 or newer is required."
}

# ── Configuration ────────────────────────────────────────────────────────────
$GitName = "Your Name"
$GitEmail = "your.email@example.com"
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

if (-not (Get-Variable -Name NoPause -Scope Script -ErrorAction SilentlyContinue)) {
    $NoPause = $false
}

$isScriptRun = -not [string]::IsNullOrWhiteSpace($PSCommandPath)

$scriptFailed = $false
$transcriptStarted = $false
$logFile = Join-Path $env:TEMP ("install-toolkit-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

try {
    Start-Transcript -Path $logFile -Force | Out-Null
    $transcriptStarted = $true
} catch {
    Write-Host "⚠️ Could not start transcript logging: $($_.Exception.Message)"
}

function Test-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Install-WithWinget {
    param(
        [Parameter(Mandatory = $true)][string]$PackageId,
        [Parameter(Mandatory = $true)][string]$DisplayName
    )

    Write-Host "📦 Installing $DisplayName via winget..."
    winget install --id $PackageId -e --source winget --accept-package-agreements --accept-source-agreements --silent
}

try {
    if (-not (Test-Command -Name "winget")) {
        throw "winget is required but was not found. Install 'App Installer' from Microsoft Store and run again."
    }

    # 1. Install Git only if missing
    if (-not (Test-Command -Name "git")) {
        Install-WithWinget -PackageId "Git.Git" -DisplayName "Git"
        Refresh-Path
    } else {
        Write-Host "✅ Git is already installed ($(git --version))."
    }

    # 2. Install VS Code only if missing
    if (-not (Test-Command -Name "code")) {
        Install-WithWinget -PackageId "Microsoft.VisualStudioCode" -DisplayName "Visual Studio Code"
        Refresh-Path
    } else {
        Write-Host "✅ VS Code is already installed."
    }

    # 3. Install UV (Python manager) only if missing
    if (-not (Test-Command -Name "uv")) {
        Write-Host "⚡ Installing uv..."
        try {
            Install-WithWinget -PackageId "astral-sh.uv" -DisplayName "uv"
        } catch {
            Write-Host "⚠️ winget install for uv failed. Trying official installer..."
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        }
        Refresh-Path
    } else {
        Write-Host "✅ uv is already installed."
    }

    # 4. Set Git identity ONLY if not already set
    $currentGitEmail = git config --global user.email
    if ([string]::IsNullOrWhiteSpace($currentGitEmail)) {
        Write-Host "👤 Setting Git identity..."
        git config --global user.name "$GitName"
        git config --global user.email "$GitEmail"
    } else {
        Write-Host "✅ Git identity already configured: $currentGitEmail"
    }

    # 5. Install VS Code extensions (safe to run multiple times)
    if (Test-Command -Name "code") {
        Write-Host "🧩 Updating VS Code extensions..."
        $extensions = @(
            "ms-python.python",
            "ms-python.vscode-python-envs",
            "ms-toolsai.jupyter",
            "ms-toolsai.datawrangler",
            "esbenp.prettier-vscode",
            "GitHub.copilot"
        )

        foreach ($extension in $extensions) {
            code --install-extension $extension --force | Out-Null
        }
    } else {
        Write-Host "⚠️ VS Code CLI ('code') not found in PATH yet. Skipping extension install."
    }

    # 6. Project initialization
    $projectDir = Join-Path $HOME "Developer\my-first-project-2026"
    if (-not (Test-Path $projectDir)) {
        Write-Host "📁 Creating project at $projectDir..."
        New-Item -ItemType Directory -Path $projectDir -Force | Out-Null
        Set-Location $projectDir

        if (Test-Command -Name "uv") {
            uv init --python 3.11
            uv add pandas ipykernel
        } else {
            Write-Host "⚠️ uv not found in PATH yet. Skipping project dependency setup."
        }
    } else {
        Write-Host "✅ Project directory already exists."
        Set-Location $projectDir
    }

    # 7. Open project in VS Code
    if (Test-Command -Name "code") {
        Write-Host "🚀 Opening project in VS Code..."
        code .
    } else {
        Write-Host "⚠️ VS Code CLI ('code') still not available in PATH. Open VS Code manually and run: code ."
    }

    Write-Host "✅ Toolkit script completed."
} catch {
    $scriptFailed = $true
    Write-Host ""
    Write-Host "❌ Toolkit script failed." -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.InvocationInfo -and $_.InvocationInfo.PositionMessage) {
        Write-Host $_.InvocationInfo.PositionMessage -ForegroundColor DarkGray
    }
} finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }

    if (Test-Path $logFile) {
        Write-Host "📝 Log file: $logFile"
    }

    if (($Host.Name -eq "ConsoleHost") -and (-not $NoPause) -and $isScriptRun) {
        Read-Host "Press Enter to close this window"
    }

    if ($scriptFailed -and $isScriptRun) {
        exit 1
    }
}
