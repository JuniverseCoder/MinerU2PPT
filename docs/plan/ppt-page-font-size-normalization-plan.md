# Task Blueprint: PPT Page Font Size Normalization Pre-Render

## I100
- Subject: Implement page font-size normalization pre-render step
- Description: PRD Section Refs: [PRD §2.1, PRD §2.2, PRD §3.1, PRD §4 FR-001, FR-002, FR-003, FR-004, FR-005]. Runtime Task ActiveForm: Implementing page font-size normalization pre-render step. Add a page-level post-process function in `converter/generator.py` and invoke it in `PPTGenerator.process_page` after text run materialization and before render dispatch. The function must collect text sizing samples per page, bucket by bold state, split each bold bucket into groups that satisfy `max_size/min_size <= 1.1`, and compute median target size per group. Use effective size precedence `run.style.font_size -> elem.style.font_size -> bbox-derived fallback` and write back integer-pt sizes (rounded) to run-level first, element-level when runs are absent. Keep behavior page-local and avoid changing non-text element flow. Include unit-test intent by exposing deterministic grouping behavior suitable for isolated unit assertions.
- Blocked By: None
- Acceptance: `process_page` executes the normalization exactly once per page before `render_to_slide`; grouping honors bold partition + 10% constraint; writeback follows run-first fallback rule; non-text rendering path remains unchanged.

## Q200
- Subject: Add deterministic grouping and guardrail handling
- Description: PRD Section Refs: [PRD §1.1 D3-D9, PRD §3.2 EX-1..EX-4, PRD §4 FR-003, FR-004, FR-005, PRD §7 Q1]. Runtime Task ActiveForm: Adding deterministic grouping and guardrail handling. Implement deterministic intra-bucket grouping strategy (stable sort + deterministic greedy grouping) so repeated runs produce identical groups for identical inputs. Add guards for empty text pages, missing size samples, single-member groups, and no-op rewrites when rounded median equals current size. Ensure exceptions on one sample do not stop page processing.
- Blocked By: I100
- Acceptance: Same input page produces stable grouping/output across runs; EX-1..EX-4 behaviors are covered in code path; single bad sample does not abort whole-page processing.

## T300
- Subject: Add and update integration tests for pipeline insertion and font unification
- Description: PRD Section Refs: [PRD §3.1, PRD §4 FR-001..FR-005, PRD §5 AG-001..AG-006, PRD §6 Integration tests required]. Runtime Task ActiveForm: Adding and updating integration tests for pipeline insertion and font unification. Extend integration coverage in existing generator/merge test modules to verify the pre-render insertion point, page-local scope, mixed-source sample handling, and final font-size unification effects. Include assertions that non-text elements are unaffected and that run-level versus elem-level writeback behavior matches expectations.
- Blocked By: Q200
- Acceptance: Integration tests assert AG-001..AG-006 mappings with at least one mixed-source page scenario and one non-text-preservation assertion.

## E400
- Subject: Run end-to-end smoke validation on demo conversion outputs
- Description: PRD Section Refs: [PRD §3.1, PRD §5 AG-004..AG-006, PRD §6 Functional/smoke tests required]. Runtime Task ActiveForm: Running end-to-end smoke validation on demo conversion outputs. Execute conversion on selected demo assets through existing CLI/flow and validate output PPT text sizes are normalized per bold class grouping logic while preserving non-text layers. Capture before/after evidence (size distribution snapshot or debug evidence) suitable for sign-off.
- Blocked By: T300
- Acceptance: At least one demo output demonstrates expected per-page font-size convergence per bold class and unchanged non-text rendering behavior, with retained verification evidence.

## D500
- Subject: Add focused unit tests for grouping, median rounding, and writeback priority
- Description: PRD Section Refs: [PRD §4 FR-002..FR-005, PRD §5 AG-003..AG-005, PRD §6 Unit tests required]. Runtime Task ActiveForm: Adding focused unit tests for grouping, median rounding, and writeback priority. Add or extend unit tests to cover bold bucketing, 10% boundary behavior (including exact-threshold cases), odd/even median rounding to integer pt, group-size>=2 replacement condition, and run-first/elem-fallback writeback semantics.
- Blocked By: E400
- Acceptance: Unit suite contains deterministic cases for threshold/median/writeback rules and passes with clear assertions for AG-003..AG-005.

## C900
- Subject: Run full test checks and finalize evidence summary
- Description: PRD Section Refs: [PRD §5 AG-001..AG-006, PRD §6 Evidence needed for sign-off]. Runtime Task ActiveForm: Running full test checks and finalizing evidence summary. Run targeted unit + integration suites related to generator/IR/merge plus any impacted regression tests, confirm all acceptance gates map to passing evidence, and prepare concise implementation summary with changed files and verification outputs.
- Blocked By: D500
- Acceptance: All relevant tests pass; AG-001..AG-006 each have explicit passing evidence; summary is ready for coding handoff completion.
