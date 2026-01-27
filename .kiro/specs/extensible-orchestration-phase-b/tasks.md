# Implementation Plan: Extensible Orchestration Phase B

## Overview

This implementation plan transforms the MuAI orchestration system from a fixed pipeline to an extensible framework with graph-based execution, registry management, comprehensive observability, and trainable routing. The plan is organized into phases matching the requirements iterations, with each task building incrementally on previous work.

## Tasks

- [x] 1. Phase B1: Core Architecture Refactoring
  - Implement unified Step API, State-driven execution, and Graph Executor
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_

  - [x] 1.1 Define Step interface and State TypedDict
    - Create `mm_orch/orchestration/step.py` with Step Protocol
    - Create `mm_orch/orchestration/state.py` with State TypedDict
    - Include all core fields (question, search_results, docs, summaries, final_answer, citations)
    - Include lesson pack fields (lesson_topic, lesson_objectives, etc.)
    - Include meta dict for execution metadata
    - _Requirements: 1.1, 2.1, 2.2, 2.4_

  - [ ]* 1.2 Write property test for Step protocol compliance
    - **Property 1: Step Protocol Compliance**
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [ ]* 1.3 Write property test for State field preservation
    - **Property 2: State Field Preservation**
    - **Validates: Requirements 2.3**

  - [x] 1.4 Implement BaseStep abstract class
    - Create `mm_orch/orchestration/base_step.py`
    - Implement validation of input_keys in State
    - Implement helper methods for State updates
    - Support both function-based and class-based steps
    - _Requirements: 1.4_

  - [x] 1.5 Implement Graph Executor
    - Create `mm_orch/orchestration/graph_executor.py`
    - Implement GraphNode dataclass
    - Implement execute method with linear chain support
    - Add conditional branching support
    - Integrate tracing before/after each step
    - Add cycle detection in graph validation
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 1.6 Write property tests for graph execution
    - **Property 3: Graph Linear Execution Order**
    - **Property 4: Graph Conditional Branching**
    - **Property 5: Universal Step Tracing**
    - **Validates: Requirements 3.2, 3.3, 3.4**

  - [x] 1.7 Refactor existing workflow steps to use new Step API
    - Convert WebSearchStep, FetchUrlStep, SummarizeStep, AnswerGenerateStep
    - Update each step to implement Step protocol
    - Ensure input_keys and output_keys are defined
    - Test that refactored steps work with Graph Executor
    - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Phase B2: Registry System and Enhanced Model Management
  - Implement Tool Registry, Model Registry, Workflow Registry, and enhanced model lifecycle
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4_

  - [x] 2.1 Implement Tool Registry
    - Create `mm_orch/registries/tool_registry.py`
    - Implement ToolMetadata dataclass
    - Implement register, get, and find_by_capability methods
    - Register existing tools (web_search, fetch_url)
    - Add placeholder for calculator and translator
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 2.2 Write property tests for Tool Registry

    - **Property 6: Tool Metadata Persistence**
    - **Property 8: Capability Query Correctness** (Tool Registry part)
    - **Validates: Requirements 4.3, 4.4**

  - [x]  2.3 Implement Model Registry
    - Create `mm_orch/registries/model_registry.py`
    - Implement ModelMetadata dataclass with all required fields
    - Implement register method with validation
    - Implement get and find_by_capability methods
    - Add validation for required metadata fields
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] * 2.4 Write property tests for Model Registry
    - **Property 7: Model Metadata Validation**
    - **Property 8: Capability Query Correctness** (Model Registry part)
    - **Validates: Requirements 5.2, 5.3, 5.4**

  - [x]  2.5 Enhance Model Manager with usage tracking
    - Update `mm_orch/runtime/model_manager.py`
    - Add ModelUsageStats dataclass
    - Implement usage counter increment on load
    - Implement 30-second residency tracking
    - Implement cleanup_stale method
    - Add priority-based model retention logic
    - _Requirements: 6.1, 6.2, 6.4_

  - [x]  2.6 Write property tests for model lifecycle
    - **Property 10: Usage Counter Increment**
    - **Property 11: Residency Timeout Eligibility**
    - **Property 12: Model Priority Ordering**
    - **Validates: Requirements 6.1, 6.2, 6.4**

  - [x] 2.7 Add quantization support to Model Manager
    - Integrate bitsandbytes for 8-bit and 4-bit quantization
    - Update _load_model to support quantization parameter
    - Add quantization_config to ModelMetadata
    - Implement fallback to standard loading if quantization fails
    - _Requirements: 6.3_

  - [x] 2.8 Implement cost tracking in steps
    - Update BaseStep to record latency, VRAM peak, model loads
    - Create `mm_orch/orchestration/cost_tracker.py`
    - Implement cost calculation with weighted formula
    - Integrate with Graph Executor to track per-step costs
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x]  2.9 Write property tests for cost tracking
    - **Property 13: Cost Calculation Invariant**
    - **Property 14: Cost Statistics Convergence**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 12.1, 12.2, 12.3**

  - [x] 2.10 Implement Workflow Registry
    - Create `mm_orch/registries/workflow_registry.py`
    - Implement WorkflowDefinition dataclass
    - Implement register method with step validation
    - Implement get and list_all methods
    - Register existing workflows (search_qa, rag_qa, lesson_pack, chat_generate)
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x]  2.11 Write property tests for Workflow Registry
    - **Property 9: Workflow Step Validation**
    - **Validates: Requirements 8.3**

- [x] 3. Checkpoint - Core infrastructure complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Phase B3: New Workflow Variants
  - Implement summarize_url, search_qa_fast, and search_qa_strict_citations workflows
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 4.1 Implement summarize_url workflow
    - Create workflow graph: FetchUrlStep → SummarizeStep
    - Register in Workflow Registry
    - Ensure output includes summary and source URL in citations
    - _Requirements: 9.1, 9.4_

  - [ ]* 4.2 Write property test for summarize_url
    - **Property 25: Summarize URL Output Fields**
    - **Validates: Requirements 9.4**

  - [x] 4.3 Implement search_qa_fast workflow
    - Create workflow graph: WebSearchStep → FetchTopNStep(n=2) → AnswerGenerateStep
    - Skip summarization step for faster execution
    - Register in Workflow Registry
    - _Requirements: 9.2_

  - [x] 4.4 Implement search_qa_strict_citations workflow
    - Create workflow graph: WebSearchStep → FetchTopNStep → SummarizeStep → AnswerGenerateStep → CitationValidationStep
    - Implement CitationValidationStep to check citation format
    - Ensure each key point has [N] citation reference
    - Register in Workflow Registry
    - _Requirements: 9.3, 9.5_

  - [ ]* 4.5 Write property test for strict citations
    - **Property 26: Strict Citations Validation**
    - **Validates: Requirements 9.5**

  - [ ]* 4.6 Write unit tests for new workflows
    - Test summarize_url with sample URLs
    - Test search_qa_fast execution time vs. standard search_qa
    - Test search_qa_strict_citations validation logic
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 5. Phase B4: Observability and Evaluation
  - Implement comprehensive tracing, quality signals, cost statistics, and regression testing
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3, 13.4, 23.1, 23.2, 23.3, 23.4_

  - [x] 5.1 Implement Tracer component
    - Create `mm_orch/observability/tracer.py`
    - Implement StepTrace and WorkflowTrace dataclasses
    - Implement start_step and end_step methods
    - Implement write_workflow_trace with JSONL format
    - Ensure traces append without overwriting
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 23.1, 23.2_

  - [ ]* 5.2 Write property tests for tracing
    - **Property 15: Trace Completeness**
    - **Property 16: Trace Append Behavior**
    - **Property 17: Exception Capture Completeness**
    - **Validates: Requirements 10.2, 10.3, 10.4, 23.2, 23.3**

  - [x] 5.3 Implement Quality Signals calculation
    - Create `mm_orch/observability/quality_signals.py`
    - Implement QualitySignals dataclass
    - Implement from_trace class method
    - Calculate citation_count, answer_length, has_search, has_citations, has_structure, failure_occurred
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 5.4 Write property tests for quality signals
    - **Property 18: Quality Signal Calculation**
    - **Property 19: Failure Rate Tracking**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**

  - [x] 5.5 Implement Cost Statistics aggregation
    - Create `mm_orch/observability/cost_stats.py`
    - Implement WorkflowCostStats dataclass
    - Implement update method with incremental averaging
    - Implement persistence to JSON file
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 5.6 Implement trace query tools
    - Create `mm_orch/observability/trace_query.py`
    - Implement functions to filter traces by workflow, date, success status
    - Implement aggregation functions for statistics
    - _Requirements: 23.4_

  - [x] 5.7 Implement regression test harness
    - Create `scripts/run_regression_tests.py`
    - Load test dataset from JSONL file
    - Execute workflows and compare outputs
    - Generate summary report with pass/fail status
    - Include quality metrics and cost statistics in report
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ]* 5.8 Write unit tests for regression harness
    - Test harness execution with sample dataset
    - Test comparison logic
    - Test report generation
    - _Requirements: 13.2, 13.3, 13.4_

- [x] 6. Checkpoint - Observability complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Phase C: Trainable Router Implementation
  - Implement Router v1 (rules), v2 (classifier), v3 (cost-aware), and training pipeline
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 15.1, 15.2, 15.3, 15.4, 16.1, 16.2, 16.3, 16.4, 17.1, 17.2, 17.3, 17.4_

  - [x] 7.1 Implement Router v1 (rule-based)
    - Create `mm_orch/routing/router_v1.py`
    - Implement keyword-based rules for workflow selection
    - Return workflow name, confidence score, and ranked candidates
    - Log router decisions with matched rules
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [ ]* 7.2 Write property test for router candidate ranking
    - **Property 20: Router Candidate Ranking**
    - **Validates: Requirements 14.3**

  - [x] 7.3 Implement Router v2 (classifier-based)
    - Create `mm_orch/routing/router_v2.py`
    - Implement loading of trained vectorizer and classifier
    - Implement route method using TF-IDF + LogisticRegression
    - Return probability distributions over workflows
    - Implement model persistence (save/load)
    - _Requirements: 15.1, 15.3, 15.4_

  - [ ]* 7.4 Write property test for probability distributions
    - **Property 21: Probability Distribution Validity**
    - **Validates: Requirements 15.3**

  - [x] 7.5 Implement Router v2 training script
    - Create `scripts/train_router_v2.py`
    - Read execution traces from JSONL files
    - Extract questions and chosen workflows
    - Use TF-IDF vectorization on question text
    - Train LogisticRegression classifier
    - Save vectorizer and model to disk
    - _Requirements: 15.2, 17.1, 17.2, 17.4_

  - [x] 7.6 Implement Router v3 (cost-aware)
    - Create `mm_orch/routing/router_v3.py`
    - Load classifier, vectorizer, and cost statistics
    - Implement mode feature extraction and one-hot encoding
    - Implement cost-aware scoring: quality - lambda * cost
    - Return ranked candidates by cost-aware score
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [ ]* 7.7 Write property tests for Router v3
    - **Property 22: Cost-Aware Scoring Formula**
    - **Property 23: Mode Feature Encoding**
    - **Property 24: Mode-Specific Preference**
    - **Validates: Requirements 16.2, 16.3, 16.4**

  - [x] 7.8 Implement Router v3 training script
    - Create `scripts/train_router_v3.py`
    - Read traces and extract mode features
    - Create one-hot encoding for mode_chat
    - Concatenate text features with mode features
    - Use best_reward labeling strategy
    - Train classifier and save all artifacts
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 21.3_

  - [ ]* 7.9 Write unit tests for training pipeline
    - Test feature extraction from traces
    - Test best_reward labeling logic
    - Test model training and persistence
    - _Requirements: 17.2, 17.3, 17.4_

- [x] 8. Phase D: Structured Lesson Pack
  - Implement JSON-structured lesson output with validation
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 19.1, 19.2, 19.3, 19.4_

  - [x] 8.1 Implement LessonSection and StructuredLesson dataclasses
    - Create `mm_orch/workflows/lesson_structure.py`
    - Implement LessonSection with all required fields
    - Implement StructuredLesson with to_json and from_json methods
    - Implement validate method with completeness checks
    - _Requirements: 18.2, 19.1, 19.2, 19.4_

  - [ ]* 8.2 Write property tests for lesson structure
    - **Property 27: Lesson JSON Structure**
    - **Property 28: Lesson Minimum Sections**
    - **Property 29: Lesson Content Requirement**
    - **Property 30: Lesson Completeness Score**
    - **Validates: Requirements 18.1, 18.2, 18.3, 19.1, 19.2, 19.4**

  - [x] 8.3 Update LessonExplainStep to output JSON
    - Modify prompt to request JSON output
    - Parse JSON response and create StructuredLesson
    - Store in State.lesson_explain_structured
    - Implement fallback to plain text on parse error
    - Log parse errors in trace
    - _Requirements: 18.1, 18.3, 18.4_

  - [x] 8.4 Update lesson_pack workflow validation
    - Add validation step after LessonExplainStep
    - Check minimum sections requirement
    - Check content requirement (examples or questions)
    - Record validation errors in trace
    - Calculate completeness score
    - _Requirements: 19.1, 19.2, 19.3, 19.4_

  - [ ]* 8.5 Write unit tests for lesson validation
    - Test validation with valid lessons
    - Test validation with insufficient sections
    - Test validation with missing content
    - Test completeness score calculation
    - _Requirements: 19.1, 19.2, 19.3, 19.4_

  - [x] 8.6 Update CLI display for structured lessons
    - Modify `mm_orch/main.py` lesson_pack output
    - Display topic and grade prominently
    - Format each section with clear headers
    - List examples with numbering
    - Display key points as bullet points
    - _Requirements: 20.3, 20.4_

  - [ ]* 8.7 Write property tests for CLI formatting
    - **Property: Examples are numbered in output**
    - **Property: Key points have bullet points in output**
    - **Validates: Requirements 20.3, 20.4**

- [x] 9. Phase E: Mode-Aware Routing Integration
  - Integrate mode features throughout the routing pipeline
  - _Requirements: 21.1, 21.2, 21.3, 21.4_

  - [x] 9.1 Update State to include mode in meta
    - Ensure State.meta["mode"] is set in conversation manager
    - Set mode="chat" for chat interactions
    - Set mode="default" for CLI single-shot queries
    - _Requirements: 21.1_

  - [x] 9.2 Update Router v3 to use mode features
    - Extract mode from State.meta in route method
    - Encode mode as one-hot feature
    - Concatenate with text features
    - Use mode features in prediction
    - _Requirements: 21.1, 21.2, 21.4_

  - [ ]* 9.3 Write property tests for mode-aware routing
    - **Property 21: Mode Feature Encoding** (already covered in 7.7)
    - **Property: Mode features included in routing**
    - **Validates: Requirements 21.1, 21.2, 21.4**

  - [x] 9.4 Update training script to include mode features
    - Extract mode from trace metadata
    - Create mode_is_chat feature
    - Include in training data
    - Document mode feature in model metadata
    - _Requirements: 21.3_

- [x] 10. Backward Compatibility and Integration
  - Ensure Phase B maintains compatibility with Phase A
  - _Requirements: 22.1, 22.2, 22.3, 22.4_

  - [x] 10.1 Create compatibility layer for legacy workflows
    - Implement adapters for Phase A workflow interfaces
    - Ensure existing workflow calls work without modification
    - Test all Phase A workflows in Phase B environment
    - _Requirements: 22.1, 22.2_

  - [ ]* 10.2 Write property tests for backward compatibility
    - **Property 33: Legacy Workflow Execution**
    - **Property 34: API Contract Stability**
    - **Property 35: Configuration Fallback**
    - **Validates: Requirements 22.1, 22.2, 22.3, 22.4**

  - [x] 10.3 Implement configuration fallback logic
    - Check for Phase B configuration files
    - Fall back to Phase A behavior if missing
    - Log fallback decisions
    - _Requirements: 22.4_

  - [x] 10.4 Update main orchestrator to use new components
    - Integrate Graph Executor into main execution flow
    - Use Workflow Registry for workflow lookup
    - Use Router v3 (with fallback to v2, v1)
    - Ensure Tracer is called for all executions
    - _Requirements: 22.1, 22.2, 23.1_

- [x] 11. Serialization and Persistence
  - Implement robust State serialization for trace logging
  - _Requirements: 24.1, 24.2, 24.3, 24.4_

  - [x] 11.1 Implement State serialization utilities
    - Create `mm_orch/orchestration/serialization.py`
    - Implement state_to_json function
    - Implement json_to_state function
    - Handle nested structures and optional fields
    - Implement descriptive error messages
    - _Requirements: 24.1, 24.2, 24.3, 24.4_

  - [ ]* 11.2 Write property tests for serialization
    - **Property 31: State Serialization Round Trip**
    - **Property 32: Nested Structure Preservation**
    - **Validates: Requirements 24.1, 24.2, 24.3**

  - [ ]* 11.3 Write unit tests for serialization edge cases
    - Test with empty State
    - Test with None values
    - Test with deeply nested structures
    - Test error handling for non-serializable objects
    - _Requirements: 24.3, 24.4_

- [x] 12. Final Integration and Testing
  - Complete end-to-end integration and comprehensive testing

  - [x] 12.1 Create end-to-end integration tests
    - Test complete workflow execution from user input to output
    - Test router selection with different question types
    - Test trace generation and persistence
    - Test cost statistics accumulation
    - Test quality signal calculation

  - [x] 12.2 Create performance benchmarks
    - Benchmark workflow execution times
    - Benchmark model loading and unloading
    - Benchmark router prediction times
    - Compare Phase B performance to Phase A baseline

  - [x] 12.3 Update documentation
    - Document new Step API and how to create steps
    - Document registry usage and registration patterns
    - Document router training process
    - Document trace format and query tools
    - Update architecture diagrams

  - [x] 12.4 Create migration guide
    - Document steps to migrate from Phase A to Phase B
    - Provide examples of converting old workflows to new format
    - Document configuration changes
    - Provide troubleshooting guide

- [x] 13. Final checkpoint - Phase B complete
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples and edge cases
- The implementation follows a bottom-up approach: infrastructure → workflows → routing → integration
- Backward compatibility is maintained throughout to ensure Phase A functionality continues working