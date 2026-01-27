# Requirements Document: Extensible Orchestration Phase B

## Introduction

This document specifies the requirements for Phase B of the MuAI Multi-Model Orchestration System, which transforms the existing fixed pipeline architecture into an extensible, observable, and intelligent orchestration framework with trainable routing capabilities. Phase B builds upon the completed muai-orchestration-system to enable graph-based workflow execution, registry-based component management, comprehensive observability, and machine learning-driven routing decisions.

## Glossary

- **System**: The MuAI Multi-Model Orchestration System Phase B
- **Step**: An atomic execution unit with defined input/output keys that performs a specific operation
- **State**: A TypedDict or dataclass containing all data flowing through workflow execution
- **Graph_Executor**: The component responsible for executing workflow graphs with support for linear chains and branching
- **Tool_Registry**: A registry managing external tools (search, fetch, calculator, translator)
- **Model_Registry**: A registry managing ML models with metadata about capabilities and resource requirements
- **Workflow_Registry**: A registry mapping workflow names to graph definitions
- **Router**: The component responsible for selecting appropriate workflows based on user requests
- **Trace**: A structured log record capturing execution details for observability
- **Cost**: A measure combining latency, VRAM usage, and model load count
- **Mode**: The execution context (chat, default) affecting routing decisions
- **Lesson_Section**: A structured component of teaching content with specific fields

## Requirements

### Requirement 1: Unified Step API

**User Story:** As a system architect, I want a standardized Step interface, so that all workflow components follow consistent patterns and can be composed flexibly.

#### Acceptance Criteria

1. THE System SHALL define a Step interface with name, input_keys, and output_keys attributes
2. WHEN a Step executes, THE System SHALL accept State and Runtime parameters
3. WHEN a Step completes, THE System SHALL return an updated State object
4. THE System SHALL support both function-based and class-based Step implementations

### Requirement 2: State-Driven Execution

**User Story:** As a developer, I want workflow data to flow through a unified State object, so that all steps can access and modify shared data consistently.

#### Acceptance Criteria

1. THE System SHALL implement State as a TypedDict or dataclass
2. THE State SHALL contain fields for question, search_results, docs, summaries, final_answer, and citations
3. WHEN State is modified by a Step, THE System SHALL preserve all existing fields not explicitly updated
4. THE State SHALL include a meta dictionary for execution metadata

### Requirement 3: Graph-Based Workflow Execution

**User Story:** As a workflow designer, I want to define workflows as graphs of steps, so that I can create both linear chains and branching logic.

#### Acceptance Criteria

1. THE System SHALL implement a Graph_Executor that executes workflow graphs
2. THE Graph_Executor SHALL support linear step chains
3. THE Graph_Executor SHALL support conditional branching based on State values
4. WHEN a Step executes, THE Graph_Executor SHALL record trace information before and after execution

### Requirement 4: Tool Registry

**User Story:** As a system integrator, I want to register and manage external tools centrally, so that workflows can discover and use tools dynamically.

#### Acceptance Criteria

1. THE System SHALL implement a Tool_Registry for managing tools
2. THE Tool_Registry SHALL support registration of search, fetch, calculator, and translator tools
3. WHEN a tool is registered, THE System SHALL store tool metadata including name and capabilities
4. THE System SHALL provide methods to retrieve tools by name or capability

### Requirement 5: Model Registry with Metadata

**User Story:** As a resource manager, I want detailed metadata about each model, so that the system can make informed loading and scheduling decisions.

#### Acceptance Criteria

1. THE System SHALL implement a Model_Registry for managing ML models
2. WHEN a model is registered, THE System SHALL store capabilities, expected_vram_mb, supports_quant, and preferred_device_policy
3. THE Model_Registry SHALL support querying models by capability tags
4. THE System SHALL validate that all required metadata fields are present during registration

### Requirement 6: Enhanced Model Management

**User Story:** As a performance optimizer, I want intelligent model lifecycle management, so that GPU memory is used efficiently without constant loading/unloading.

#### Acceptance Criteria

1. WHEN a model is loaded, THE System SHALL increment its usage counter
2. WHEN a model is not used for 30 seconds, THE System SHALL mark it eligible for unloading
3. THE System SHALL support 8-bit and 4-bit quantization using bitsandbytes
4. WHEN multiple models compete for GPU memory, THE System SHALL prioritize based on usage patterns and device policy

### Requirement 7: Cost Tracking Per Step

**User Story:** As a system analyst, I want detailed cost metrics for each step execution, so that I can identify performance bottlenecks and optimize workflows.

#### Acceptance Criteria

1. WHEN a Step executes, THE System SHALL record latency in milliseconds
2. WHEN a Step loads a model, THE System SHALL record VRAM peak usage in megabytes
3. WHEN a Step completes, THE System SHALL record the number of model loads performed
4. THE System SHALL calculate normalized cost as a weighted combination of latency, VRAM, and load count

### Requirement 8: Workflow Registry

**User Story:** As a workflow developer, I want to register workflow definitions centrally, so that the router can discover and select appropriate workflows.

#### Acceptance Criteria

1. THE System SHALL implement a Workflow_Registry mapping workflow names to graph definitions
2. THE Workflow_Registry SHALL support registration of search_qa, rag_qa, lesson_pack, and new workflow variants
3. WHEN a workflow is registered, THE System SHALL validate that all referenced steps exist
4. THE System SHALL provide methods to retrieve workflow definitions by name

### Requirement 9: New Workflow Variants

**User Story:** As a product manager, I want multiple variants of search-based workflows, so that users can choose between speed, thoroughness, and citation quality.

#### Acceptance Criteria

1. THE System SHALL implement a summarize_url workflow that fetches and summarizes a single URL
2. THE System SHALL implement a search_qa_fast workflow with reduced summarization for faster responses
3. THE System SHALL implement a search_qa_strict_citations workflow with enforced citation formatting
4. WHEN summarize_url executes, THE System SHALL output summary and source URL
5. WHEN search_qa_strict_citations executes, THE System SHALL validate that each key point includes a citation reference

### Requirement 10: Trace and Dataset Logging

**User Story:** As a data scientist, I want comprehensive execution traces in JSONL format, so that I can analyze system behavior and train improved routers.

#### Acceptance Criteria

1. WHEN a workflow executes, THE System SHALL write a trace record in JSONL format
2. THE trace SHALL include request_id, question, chosen_workflow, steps_trace, urls_used, final_answer, and quality_signals
3. WHEN a step executes, THE System SHALL record step name, latency, VRAM peak, and any exceptions
4. THE System SHALL append traces to a file without overwriting existing records

### Requirement 11: Automatic Quality Signals

**User Story:** As a machine learning engineer, I want automatic quality metrics for each execution, so that I can use them as training signals for the router.

#### Acceptance Criteria

1. WHEN a workflow completes, THE System SHALL calculate citation count
2. WHEN a workflow completes, THE System SHALL measure answer length in characters
3. WHEN a workflow fails, THE System SHALL record failure rate
4. THE System SHALL include quality signals in the trace record

### Requirement 12: Cost Statistics Collection

**User Story:** As a system administrator, I want aggregated cost statistics per workflow, so that I can understand resource usage patterns.

#### Acceptance Criteria

1. THE System SHALL maintain running statistics for average latency per workflow
2. THE System SHALL maintain running statistics for average VRAM usage per workflow
3. THE System SHALL maintain running statistics for average model load count per workflow
4. THE System SHALL persist cost statistics to a JSON file

### Requirement 13: Regression Test Harness

**User Story:** As a quality assurance engineer, I want an automated test harness for workflows, so that I can detect regressions when making changes.

#### Acceptance Criteria

1. THE System SHALL provide a test harness that executes workflows against a test dataset
2. WHEN tests execute, THE System SHALL compare outputs against expected results
3. THE System SHALL report pass/fail status for each test case
4. THE System SHALL generate a summary report with quality metrics and cost statistics

### Requirement 14: Rule-Based Router v1

**User Story:** As a system designer, I want a baseline rule-based router, so that I have a foundation for comparison with learned routers.

#### Acceptance Criteria

1. THE System SHALL implement Router v1 using keyword-based rules
2. WHEN Router v1 evaluates a question, THE System SHALL return a workflow name and confidence score
3. WHEN multiple rules match, THE System SHALL return a ranked list of candidate workflows
4. THE System SHALL log router decisions including matched rules and confidence scores

### Requirement 15: Lightweight Classifier Router v2

**User Story:** As a machine learning engineer, I want a trained classifier for intent classification, so that routing decisions improve beyond simple rules.

#### Acceptance Criteria

1. THE System SHALL implement Router v2 using a MiniLM or DistilBERT classifier
2. WHEN Router v2 is trained, THE System SHALL use execution traces as training data
3. WHEN Router v2 predicts, THE System SHALL output probability distributions over workflows
4. THE System SHALL save trained Router v2 models to disk for reuse

### Requirement 16: Cost-Aware Router v3

**User Story:** As a system optimizer, I want routing decisions to consider both quality and cost, so that the system balances performance and resource usage.

#### Acceptance Criteria

1. THE System SHALL implement Router v3 that incorporates cost statistics
2. WHEN Router v3 evaluates candidates, THE System SHALL score workflows using quality minus lambda times cost
3. THE System SHALL support mode features as one-hot encoded inputs
4. WHEN mode is chat, THE System SHALL increase preference for chat_generate and lesson_pack workflows

### Requirement 17: Router Training Pipeline

**User Story:** As a machine learning engineer, I want an automated training pipeline for routers, so that I can continuously improve routing decisions from execution data.

#### Acceptance Criteria

1. THE System SHALL provide a training script that reads trace JSONL files
2. WHEN training, THE System SHALL extract features from questions and metadata
3. WHEN training, THE System SHALL use best_reward labeling to select optimal workflows
4. THE System SHALL output trained models and vectorizers for deployment

### Requirement 18: Structured Lesson Pack Output

**User Story:** As an educator, I want lesson plans in structured JSON format, so that I can programmatically validate and display teaching content.

#### Acceptance Criteria

1. WHEN lesson_pack executes, THE System SHALL output JSON with topic, grade, and sections fields
2. WHEN a Lesson_Section is created, THE System SHALL include name, teacher_say, student_may_say, examples, questions, and key_points fields
3. THE System SHALL validate that lesson JSON contains at least three sections
4. WHEN lesson JSON is invalid, THE System SHALL fall back to plain text format and log a parse error

### Requirement 19: Lesson Structure Validation

**User Story:** As a quality assurance engineer, I want automatic validation of lesson structure, so that generated lessons meet minimum quality standards.

#### Acceptance Criteria

1. WHEN lesson_pack completes, THE System SHALL verify that sections array is not empty
2. WHEN lesson_pack completes, THE System SHALL verify that at least one section contains examples or questions
3. WHEN validation fails, THE System SHALL record validation errors in the trace
4. THE System SHALL calculate a completeness score based on presence of required fields

### Requirement 20: Enhanced CLI Display for Lessons

**User Story:** As a user, I want clear and organized display of structured lessons, so that I can easily read and use the generated content.

#### Acceptance Criteria

1. WHEN lesson_pack outputs to CLI, THE System SHALL display topic and grade prominently
2. WHEN displaying sections, THE System SHALL format each section with clear headers
3. WHEN a section contains examples, THE System SHALL list them with numbering
4. WHEN a section contains key points, THE System SHALL display them as bullet points

### Requirement 21: Mode-Aware Routing

**User Story:** As a conversation designer, I want the router to consider execution mode, so that chat interactions receive appropriate workflow selection.

#### Acceptance Criteria

1. WHEN State contains mode metadata, THE System SHALL include mode in routing features
2. WHEN mode is chat, THE Router SHALL encode this as a one-hot feature
3. WHEN Router v3 trains, THE System SHALL include mode features in the training data
4. WHEN Router v3 predicts, THE System SHALL use mode features from the current State

### Requirement 22: Backward Compatibility

**User Story:** As a system maintainer, I want Phase B to maintain compatibility with existing workflows, so that current functionality continues to work without modification.

#### Acceptance Criteria

1. WHEN Phase B is deployed, THE System SHALL continue to support all existing workflow types
2. WHEN legacy code calls workflows, THE System SHALL execute them successfully
3. THE System SHALL maintain existing API contracts for external integrations
4. WHEN configuration is missing, THE System SHALL fall back to Phase A behavior

### Requirement 23: Observability for All Executions

**User Story:** As a system operator, I want complete visibility into every execution, so that I can debug issues and understand system behavior.

#### Acceptance Criteria

1. WHEN any workflow executes, THE System SHALL generate a trace record
2. THE trace SHALL include timestamps for start and end of each step
3. WHEN an exception occurs, THE System SHALL capture exception type, message, and stack trace in the trace
4. THE System SHALL provide tools to query and analyze trace files

### Requirement 24: Serialization Round Trip

**User Story:** As a developer, I want State objects to serialize and deserialize correctly, so that I can persist and restore execution state.

#### Acceptance Criteria

1. WHEN State is serialized to JSON, THE System SHALL preserve all field values
2. WHEN JSON is deserialized to State, THE System SHALL produce an equivalent State object
3. THE System SHALL handle nested structures and optional fields correctly
4. WHEN serialization fails, THE System SHALL raise a descriptive error

