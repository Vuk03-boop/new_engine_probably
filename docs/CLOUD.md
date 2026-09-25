# CLOUD — rules for cloud sessions only

Read this only when `CLAUDE_CODE_REMOTE=true`. Local sessions ignore it; nothing here changes local work.

## What the cloud container can and cannot do

- **Can:** edit code and docs; build and run the pure CPU crates (`cargo test --release -j 2`); reach crates.io, PyPI and GitHub.
- **Can, as a supplemental compile check (4A):** build `gpu` and `viewer`, run clippy on the workspace, and run the device-free tests (`gpu --lib`, CPU-only diagnostics). Setup, in the session's scratch space, never in the repo: download the pinned slangc's Linux release (`https://github.com/shader-slang/slang/releases/download/v2026.13.1/slang-2026.13.1-linux-x86_64.tar.gz`), `apt-get install spirv-tools`, `rustup component add clippy`, and point `VULKAN_SDK` at a folder whose `Bin/slangc.exe` and `Bin/spirv-val.exe` are shell wrappers calling them. This `spirv-val` is not the SDK's, so it is not ADR-0001 proof.
- **Cannot:** no GPU, no Vulkan driver, no display. So GPU tests, `viewer` runs, the gate, FLIP and every perf or fps number are **NOT RUN** in the cloud, never "passed".
- CPU timings from the cloud are not evidence for the laptop. Do not report them as performance.

## The one-click handoff

Anything the cloud cannot run goes to the user as **one script**: `run-local.cmd` in the repo root (the laptop is Windows, RTX 3050). The user pulls, double-clicks it, and it does the rest.

The script must:

1. `cd /d "%~dp0engine"` so it works from a double-click in any folder.
2. Check prerequisites first (`cargo`, `VULKAN_SDK`) and stop with a plain message saying what is missing.
3. Run only commands from `engine/README.md` (verified list) or ones the change record introduces; use `-j 2` and `--test-threads=1` for GPU tests.
4. Write each step's full log and true exit code to `engine/results/local-run/<date>-<task>/`, plus a short `summary.txt` (step, PASS/FAIL/NOT RUN, log file).
5. Continue past a failed step unless later steps depend on it; never hide a failure.
6. End with `pause` and print where the summary is. It should say which files to send back (push the results folder, or paste `summary.txt`).

The script must not:

- install, download or upgrade anything beyond `cargo`'s locked dependencies;
- modify tracked source, reset/clean the repo, commit or push;
- start background processes, watchers or benchmarks that outlive the script;
- need an argument or answer mid-run (default to the task's intended run).

## Session rules

- Overwrite `run-local.cmd` for the current task. Record in the change record and `docs/NOW.md` which steps it runs and that they are **NOT RUN** until the user returns results.
- The script cannot be executed in the cloud: say it is untested, and re-read it against the rules above before pushing.
- When results come back, analyze the logs like any local run; do not rerun or rebaseline to get green.
