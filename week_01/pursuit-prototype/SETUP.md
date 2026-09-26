# Dependencies and Windows setup

Tested environment: Windows, Python 3.13.3, .NET SDK 8.0.425, Ollama 0.34.0 and qwen3:8b (Q4_K_M). [dependencies.json](dependencies.json) records repository revisions and the tested model digest. [requirements-windows-py313.txt](requirements-windows-py313.txt) is the full tested Python snapshot; it is Windows/Python-3.13-specific, not a cross-platform lockfile.

No API key is needed for local Ollama. Model weights, game source, .NET binaries and the virtual environment are dependencies, not submission files.

## Existing runtime

Run `./run-designer.ps1 -RuntimeRoot 'YOUR_RUNTIME_DIRECTORY' -CheckOnly`. Remove -CheckOnly to execute. Alternatively set OPENRA_RUNTIME_ROOT or put the local absolute directory in the ignored `.runtime-path` file beside run-designer.ps1. The author's existing runtime can remain outside the course repository.

Expected layout: runtime/source is the OpenRA-RL checkout; runtime/source/OpenRA is its matching engine checkout with built bin directory; runtime/venv is the Python environment. Optional runtime/dotnet contains a portable SDK/runtime; otherwise the Python scripts use the installed dotnet on PATH.

## New runtime

Install Git, Python 3.13, a .NET 8 SDK and Ollama. Commands below run from pursuit-prototype. They document the verified versions and directory layout; a second fresh-machine installation has not been tested.

1. `New-Item -ItemType Directory -Force .runtime`
2. `git clone --no-checkout https://github.com/yxc20089/OpenRA-RL.git .runtime/source`
3. `git -C .runtime/source checkout --detach 5dadd449c912ac2d4021cc8ed84fc0b385b1543c`
4. `git clone https://github.com/yxc20089/OpenRA.git .runtime/source/OpenRA`
5. `git -C .runtime/source/OpenRA checkout --detach 9b271c1a5562a07deeb1c3f81d9a556cd563a676`
6. `py -3.13 -m venv .runtime/venv`
7. `./.runtime/venv/Scripts/python.exe -m pip install -r requirements-windows-py313.txt`
8. `./.runtime/venv/Scripts/python.exe -m pip check`
9. `dotnet build .runtime/source/OpenRA/OpenRA.sln -c Release -p:TargetPlatform=win-x64`
10. Start Ollama and run `ollama pull qwen3:8b`.
11. `./run-designer.ps1 -CheckOnly`, followed by `./run-designer.ps1`.

The Python entry points import the pinned source directly from runtime/source; an editable installation of OpenRA-RL is unnecessary. The tested headless scenario uses the pinned Red Alert mod and Singles map, without requiring a Red Alert 2 installation. Model tags can change: compare the installed digest with dependencies.json when reproducing the exact model.

## Compatibility notes from testing

- Use the implemented multi-session backend. A legacy single-session move request timed out during initial verification.
- Keep openenv-core=0.3.0, mcp=1.30.0, fastmcp=3.4.7 and fastmcp-slim=3.4.7. A newer MCP major version broke an imported interface. If upgrading an existing environment causes overlapping FastMCP package-file problems, the tested recovery was reinstalling fastmcp-slim 3.4.7 and rerunning pip check.
- The original machine needed a temporary local adapter for NuGet downloads because .NET HTTPS failed while Python HTTPS worked. It is not required after compilation and is not a mandatory setup step. Try normal NuGet sources first; network/certificate failures need machine-specific diagnosis.
- Ollama uses localhost:11434; the game bridge uses port 19100. Run experiments sequentially.
- CheckOnly verifies paths and availability of a dotnet executable, not its version, model generation, or successful game execution.

Without game/model dependencies, `python -m unittest discover -s tests` checks validation and feedback logic using only Python. Real game evidence is committed separately.

The fixed baseline can be run separately with `./baseline/run.ps1` from the project root. Its default configuration is baseline/scenario.json; output still goes to the ignored runs/ directory in the project root. The main designer entry point remains ./run-designer.ps1.
