# Lesson Pack Structured Output - Requirements

**Feature Name**: lesson-pack-structured-output  
**Created**: 2026-01-30  
**Status**: Draft  
**Priority**: High

---

## Overview

Enhance the LessonPack workflow to generate structured JSON output with teaching sections (导入、新授、练习、小结) instead of a single block of explanation text. This enables better programmatic access, automated validation, and future UI rendering.

---

## User Stories

### Story 1: Structured Teaching Sections
**As a** teacher using the lesson pack workflow  
**I want** the explanation content organized into structured teaching sections  
**So that** I can easily navigate different parts of the lesson and use them independently

**Acceptance Criteria**:
1.1. The lesson explanation is divided into named sections (e.g., "导入", "新授", "练习", "小结")  
1.2. Each section contains structured fields appropriate to its type  
1.3. The output is valid JSON that can be parsed programmatically  
1.4. The CLI can display both JSON and human-readable formats

### Story 2: Section-Specific Content
**As a** developer building on the lesson pack system  
**I want** each teaching section to have type-specific fields  
**So that** I can validate and process different section types appropriately

**Acceptance Criteria**:
2.1. "导入" sections include teacher_say, student_may_say, and tips fields  
2.2. "新授" sections include teacher_say and examples fields  
2.3. "练习" sections include teacher_say and questions fields  
2.4. "小结" sections include teacher_say and key_points fields  
2.5. All fields are optional except name and teacher_say

### Story 3: Backward Compatibility
**As a** system maintainer  
**I want** the new structured format to coexist with the old format  
**So that** existing code continues to work while new features are adopted

**Acceptance Criteria**:
3.1. The workflow still generates the old "explanation" field for backward compatibility  
3.2. A new "lesson_explain_structured" field contains the structured JSON  
3.3. Existing tests continue to pass without modification  
3.4. New tests validate the structured output format

### Story 4: Automated Validation
**As a** quality assurance engineer  
**I want** to automatically validate lesson pack outputs  
**So that** I can ensure quality standards are met consistently

**Acceptance Criteria**:
4.1. Validation can check for minimum number of sections (e.g., ≥ 3)  
4.2. Validation can verify specific section names exist (e.g., "练习")  
4.3. Validation can count examples and questions across all sections  
4.4. Validation results are included in workflow metadata

---

## Functional Requirements

### FR-1: State Extension
**Priority**: High  
**Description**: Extend the workflow state to store structured lesson explanation

**Details**:
- Add `lesson_explain_structured` field to LessonPackContext
- Field type: Dict[str, Any] containing the structured JSON
- Field is populated during the explanation generation step
- Field is included in the workflow result metadata

### FR-2: Structured JSON Generation
**Priority**: High  
**Description**: Generate lesson explanation as structured JSON with sections

**Details**:
- Modify the explanation generation prompt to request JSON output
- Prompt specifies the required JSON schema with sections array
- Each section has: name, teacher_say, and optional type-specific fields
- JSON parsing with error handling and fallback to plain text

**JSON Schema**:
```json
{
  "topic": "string",
  "grade": "string (optional)",
  "sections": [
    {
      "name": "string (required)",
      "teacher_say": "string (required)",
      "student_may_say": "string (optional)",
      "examples": ["string"] (optional),
      "questions": ["string"] (optional),
      "key_points": ["string"] (optional),
      "tips": "string (optional)"
    }
  ]
}
```

### FR-3: JSON Parsing and Validation
**Priority**: High  
**Description**: Parse and validate generated JSON, with fallback handling

**Details**:
- Attempt to parse generated text as JSON using json.loads()
- Clean text before parsing (remove markdown code blocks, extra whitespace)
- Validate required fields exist (topic, sections array)
- Validate each section has name and teacher_say
- On parse failure, log error and fall back to plain text explanation

### FR-4: Dual Output Format
**Priority**: High  
**Description**: Generate both structured JSON and plain text explanation

**Details**:
- Store structured JSON in `lesson_explain_structured` field
- Render structured JSON to plain text for `explanation` field
- Plain text rendering: concatenate sections with headers
- Maintain backward compatibility with existing code expecting `explanation`

### FR-5: CLI Display Options
**Priority**: Medium  
**Description**: Provide CLI options to display structured or plain text format

**Details**:
- Default: Display plain text explanation (backward compatible)
- `--structured` flag: Display structured JSON format
- `--format json`: Output raw JSON to stdout
- `--format text`: Output formatted text (default)

### FR-6: Validation Utilities
**Priority**: Medium  
**Description**: Provide utilities to validate structured lesson output

**Details**:
- Function to check minimum number of sections
- Function to verify specific section names exist
- Function to count examples and questions
- Validation results included in workflow metadata
- Validation can be run in tests or evaluation scripts

---

## Non-Functional Requirements

### NFR-1: Performance
- JSON parsing should add < 50ms overhead
- Fallback to plain text should be fast (< 100ms)
- No impact on model inference time

### NFR-2: Reliability
- JSON parsing errors should not crash the workflow
- Fallback to plain text should always succeed
- Invalid JSON should be logged for debugging

### NFR-3: Maintainability
- Structured format should be documented in code comments
- JSON schema should be defined as a constant
- Parsing logic should be in a separate method for testability

### NFR-4: Compatibility
- Existing tests should pass without modification
- Existing API contracts should remain unchanged
- New features should be opt-in where possible

---

## Technical Constraints

1. **Model Limitations**: LLM may not always generate valid JSON
2. **Prompt Engineering**: Prompt must be carefully crafted to encourage JSON output
3. **Language Support**: Must work for both Chinese and English
4. **Backward Compatibility**: Cannot break existing workflows or tests

---

## Dependencies

- Existing LessonPackWorkflow implementation
- JSON parsing library (Python standard library)
- LessonPackContext dataclass
- WorkflowResult schema

---

## Success Metrics

1. **JSON Parse Success Rate**: > 90% of generations produce valid JSON
2. **Backward Compatibility**: 100% of existing tests pass
3. **Validation Coverage**: All new structured outputs pass validation
4. **Performance**: < 50ms overhead for JSON parsing

---

## Out of Scope

- Web UI for rendering structured lessons (future work)
- Export to external formats (PDF, DOCX) (future work)
- Interactive lesson editing (future work)
- Multi-language translation of sections (future work)

---

## References

- 需求文档迭代8.md - Original requirements document
- mm_orch/workflows/lesson_pack.py - Current implementation
- mm_orch/schemas.py - Data models
