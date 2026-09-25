# Workspace cleanup and preservation

**2026-09-22 · User-selected scope: workspace cleanup only**

## What was removed

One generated file:

- `research-review/experiments/__pycache__/inspect_shader_source.cpython-313.pyc` — **4,064 bytes** of reproducible Python bytecode.

Its now-empty `__pycache__` directory was removed. No other file was judged safely disposable within the approved scope. The workspace had already undergone an earlier, substantial leftovers cleanup; repeating that deletion count would be misleading.

[Deletion receipt](cleanup-receipt.json) records the exact path, size, hash and reason. [Opening preservation inventory](preservation-baseline.json) records the 1,149 pre-existing regular files before this cleanup.

## What was deliberately kept

| Material | Why it is not dead workspace clutter |
|---|---|
| All 295 current-engine baseline files | User selected preservation of engine code; inactive-looking code was not audited for deletion. |
| All 16 retained PDFs and their indexes | Research source corpus; no duplicate paper copies were introduced. |
| Review reports, models, pinned dependency sources and integrity receipts | Evidence behind the architecture; includes useful negative findings. |
| Three uploaded AI reports | Inputs to the completed cross-audit, hash-referenced by receipts. |
| Historical reports, foam patch/probes and logs | Lessons and provenance, not an active patch queue. |
| Reconstruction scripts/manifests | Document how the project was recovered; some need deleted input bundles to rerun, as their historical README already explains. |
| Dependency `Cargo.toml.orig` files | Published/recovered source-package material, not stray editor backups. |
| Historical handoffs | Context worth retaining; their old next-step instructions are superseded by the workspace index and current review status. |

A `.log`, `.orig` or old date alone is not a deletion criterion. Missing historical linked binaries were not newly deleted here and were not fabricated to make checks green. The prior [missing-link census](../research-review/sources/batch-09/historical-link-audit.json) remains unchanged.

## Organizational cleanup

- Added the [workspace index](../README.md), separating the new design from the frozen engine, completed research and historical archive.
- Created this independent proposal folder with architecture, prerequisites, technique mapping, construction gates, lessons and validation contracts.
- Did not rename/move existing evidence or rewrite historical instructions inside immutable reports.
- Did not create a code project, build artifacts, dependencies, branches or Git metadata.

## Preservation verification

[verification.json](verification.json) is the closing machine-readable receipt. It checks every remaining pre-existing regular file against the opening inventory, including the old engine/research/archive/upload trees; checks the original engine/PDF inventories; and validates local file links in the new Markdown documents.

The intended result is **1,148 retained pre-existing files byte-identical**, one explicit bytecode deletion, no additional deletions or modifications, and only Markdown/JSON additions. This is a workspace/document check—not shader/API/GPU validation, a current Git clean-status claim, or an audit proving every historical engine file is live.

**Further destructive cleanup is not authorized by this result.** Removing engine code or discarding historical evidence would need a separate scope and dependency/reachability review.
