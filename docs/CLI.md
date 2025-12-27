# 🖥️ OLB (Ollama Load Balancer) CLI Tool

The `olb` CLI is a powerful companion tool designed to enhance your experience with the Ollama Load Balancer. It provides features that are missing from the standard `ollama` client, most notably the **interactive model discovery** terminal UI.

This tool is designed to work as a seamless extension of your existing `ollama` command.

---

## 📦 Installation & Setup

Since `olb` is written in Rust, you'll need the Rust toolchain installed.

### 1. Build the Binary

First, navigate to the Rust implementation directory and build the project in release mode for maximum performance.

**Linux / macOS / Windows (PowerShell)**:
```bash
cd Rust_lb
cargo build --release
```

### 2. Install to System Path

You need to place the compiled binary somewhere in your system's PATH.

#### 🐧 Linux / 🍎 macOS
We recommend installing to `~/.local/bin`.

```bash
# Ensure the directory exists
mkdir -p ~/.local/bin

# Copy the binary
cp target/release/olb ~/.local/bin/olb

# Check if it works
olb --help
```
*Tip: Make sure `~/.local/bin` is in your `$PATH`.*

#### 🪟 Windows (PowerShell)
We recommend creating a `bin` folder in your user directory.

```powershell
# Create a local bin directory
New-Item -ItemType Directory -Force -Path "$HOME\bin"

# Copy the binary
Copy-Item "target\release\olb.exe" -Destination "$HOME\bin\olb.exe"

# Add to PATH (Permanent)
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$HOME\bin", [System.EnvironmentVariableTarget]::User)
```
*Restart your terminal after updating the PATH.*

---

## ⚡ Seamless Integration (Recommended)

The best way to use `olb` is to integrate it directly into the `ollama` command. This effectively "patches" Ollama to support the `discover` command natively.

### 🐧 Linux / 🍎 macOS (Bash & Zsh)

Add the following function to your `~/.bashrc` or `~/.zshrc`:

```bash
ollama() {
    if [ "$1" = "discover" ]; then
        olb discover "${@:2}"
    else
        command ollama "$@"
    fi
}
```

**Apply changes**:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### 🐟 Linux / 🍎 macOS (Fish Shell)

Create a function file at `~/.config/fish/functions/ollama.fish`:

```fish
function ollama --wraps ollama
    if test (count $argv) -ge 1; and test $argv[1] = "discover"
        olb discover $argv[2..-1]
    else
        command ollama $argv
    end
end
```

### 🪟 Windows (PowerShell)

Add the following function to your PowerShell profile.

1.  Open your profile:
    ```powershell
    notepad $PROFILE
    ```
2.  Add this code to the end of the file:
    ```powershell
    function ollama {
        param(
            [Parameter(ValueFromRemainingArguments = $true)]
            $args
        )

        if ($args.Count -gt 0 -and $args[0] -eq "discover") {
            # Pass remaining arguments to olb discover
            $rest = $args | Select-Object -Skip 1
            & olb discover @rest
        } else {
            # Pass all arguments to native ollama
            & Get-Command -Name ollama -CommandType Application $args
        }
    }
    ```
3.  Save and restart your terminal.

---

## 🔍 Features & Usage

Once integrated, you can use `ollama discover` just like any other command.

### 1. Interactive Model Discovery (`discover`)

Search and browse the entire [ollama.com](https://ollama.com) library directly from your terminal.

**Syntax**:
```bash
ollama discover <query>
```

**Example**:
```bash
ollama discover deepseek
```

**The Interface**:
*   **Family List**: Shows model families (e.g., `llama3`, `mistral`) with high-level stats.
    *   `[ N]`: Index number for quick selection.
    *   `⬇️`: Total pulls (popularity).
    *   `🏷️`: Number of tags (variants).
    *   `🕒`: Last updated time.
*   **Variant List**: After selecting a family, see detailed tags (e.g., `8b-instruct-q4_0`).
    *   `💾`: Image size on disk (e.g., `5.6GB`).
    *   `🧠`: Context window size (e.g., `128K`).
    *   `⌨️`: Supported capabilities (e.g., `Text`, `Image`, `Tools`).

**Navigation controls**:
*   **Typer Number (e.g., `1`)**: Instantly jump to and select the item at that index. This ignores fuzzy matching rules for precision. Spaces are handled smartly (e.g., ` 1 ` works).
*   **Type Text (e.g., `code`)**: Fuzzy filter the list by name.
*   **Enter**: Select the highlighted item.
*   **Go Back (`0`)**: Return to the previous menu.

### 2. Pull Wrappers (`pull`)
You can pull models directly through `olb` if needed, though usually `ollama discover` handles this for you by auto-running the pull command after selection.

```bash
olb pull llama3
```

---

## 💡 Troubleshooting

*   **"Command not found: olb"**: Ensure `olb` is in your system `$PATH` and that you've restarted your terminal after installation.
*   **"Text file busy" (Linux)**: If you try to update `olb` while it's running, the copy will fail. Exit all instances of the tool before updating.
*   **PowerShell loop**: If the PowerShell function causes a recursion loop, make sure you are using `Get-Command ... -CommandType Application` to call the actual executable, not the function itself.
