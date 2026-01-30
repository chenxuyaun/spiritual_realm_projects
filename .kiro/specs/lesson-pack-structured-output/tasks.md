# Lesson Pack Structured Output - Implementation Tasks

**Feature**: lesson-pack-structured-output  
**Created**: 2026-01-30  
**Status**: Not Started

---

## Task List

### Phase 1: Core Implementation

- [ ] 1. Extend LessonPackContext with structured field
  - [ ] 1.1 Add `lesson_explain_structured: Dict[str, Any]` field to LessonPackContext
  - [ ] 1.2 Update `__init__` to initialize field as None
  - [ ] 1.3 Add type hints and docstring
  - [ ] 1.4 Write unit test for context creation

- [ ] 2. Implement JSON parsing utilities
  - [ ] 2.1 Create `_parse_structured_explanation()` method
  - [ ] 2.2 Implement text cleaning (remove markdown code blocks)
  - [ ] 2.3 Implement JSON validation (required fields check)
  - [ ] 2.4 Add error handling and logging
  - [ ] 2.5 Write unit tests for parsing (valid/invalid JSON)
  - [ ] 2.6 Write property test: valid JSON always parses

- [ ] 3. Implement text rendering utilities
  - [ ] 3.1 Create `_render_structured_to_text()` method
  - [ ] 3.2 Implement section rendering logic
  - [ ] 3.3 Handle all optional fields (examples, questions, key_points, tips)
  - [ ] 3.4 Format output with proper headers and spacing
  - [ ] 3.5 Write unit tests for rendering
  - [ ] 3.6 Write property test: render produces non-empty text

- [ ] 4. Create structured prompt templates
  - [ ] 4.1 Define `LESSON_EXPLANATION_STRUCTURED_PROMPT_ZH`
  - [ ] 4.2 Define `LESSON_EXPLANATION_STRUCTURED_PROMPT_EN`
  - [ ] 4.3 Include JSON schema in prompts
  - [ ] 4.4 Add examples to prompts (optional)
  - [ ] 4.5 Test prompts with real models

### Phase 2: Workflow Integration

- [ ] 5. Modify explanation generation step
  - [ ] 5.1 Update `_step_generate_explanation()` to use structured prompts
  - [ ] 5.2 Call `_parse_structured_explanation()` on generated text
  - [ ] 5.3 Store parsed JSON in `ctx.lesson_explain_structured`
  - [ ] 5.4 Call `_render_structured_to_text()` to generate plain text
  - [ ] 5.5 Store rendered text in `ctx.explanation` (backward compat)
  - [ ] 5.6 Handle parse failures with fallback to plain text
  - [ ] 5.7 Add logging for success/failure cases

- [ ] 6. Update workflow result creation
  - [ ] 6.1 Include `lesson_explain_structured` in metadata
  - [ ] 6.2 Ensure `explanation` field still populated
  - [ ] 6.3 Add validation stats to metadata
  - [ ] 6.4 Write integration test for full workflow

- [ ] 7. Implement validation utilities
  - [ ] 7.1 Create `validate_structured_lesson()` function
  - [ ] 7.2 Check minimum number of sections (≥3)
  - [ ] 7.3 Check for key section names (导入, 新授, 练习, 小结)
  - [ ] 7.4 Count examples and questions
  - [ ] 7.5 Generate validation report with stats
  - [ ] 7.6 Write unit tests for validation
  - [ ] 7.7 Write property test: validation is consistent

### Phase 3: CLI Integration

- [ ] 8. Add CLI options for structured output
  - [ ] 8.1 Add `--structured` flag to argparse
  - [ ] 8.2 Add `--format` option (text/json)
  - [ ] 8.3 Update help text and documentation
  - [ ] 8.4 Write CLI tests for new options

- [ ] 9. Implement CLI display logic
  - [ ] 9.1 Create `display_lesson_pack_result()` function
  - [ ] 9.2 Implement JSON format output
  - [ ] 9.3 Implement structured view output
  - [ ] 9.4 Implement default text output (backward compat)
  - [ ] 9.5 Test all display modes

### Phase 4: Testing and Validation

- [ ] 10. Write unit tests
  - [ ] 10.1 Test JSON parsing with valid inputs
  - [ ] 10.2 Test JSON parsing with invalid inputs
  - [ ] 10.3 Test text rendering with all field types
  - [ ] 10.4 Test validation with valid structures
  - [ ] 10.5 Test validation with invalid structures
  - [ ] 10.6 Ensure 100% code coverage for new functions

- [ ] 11. Write integration tests
  - [ ] 11.1 Test end-to-end workflow with structured output
  - [ ] 11.2 Test fallback to plain text on parse failure
  - [ ] 11.3 Test backward compatibility (old tests still pass)
  - [ ] 11.4 Test CLI with different options
  - [ ] 11.5 Test with real models (GPT-2, Qwen)

- [ ] 12. Write property-based tests
  - [ ] 12.1 Property: JSON structure validity
  - [ ] 12.2 Property: Backward compatibility
  - [ ] 12.3 Property: Dual format consistency
  - [ ] 12.4 Property: Validation accuracy
  - [ ] 12.5 Property: Parse-render roundtrip

### Phase 5: Documentation and Polish

- [ ] 13. Update documentation
  - [ ] 13.1 Update README.md with structured output feature
  - [ ] 13.2 Add examples to docs/
  - [ ] 13.3 Document JSON schema
  - [ ] 13.4 Document CLI options
  - [ ] 13.5 Add migration guide for existing users

- [ ] 14. Code quality and polish
  - [ ] 14.1 Run flake8 and fix any issues
  - [ ] 14.2 Run black formatter
  - [ ] 14.3 Add type hints to all new functions
  - [ ] 14.4 Add comprehensive docstrings
  - [ ] 14.5 Review and refactor for clarity

- [ ] 15. Performance testing
  - [ ] 15.1 Measure JSON parsing overhead
  - [ ] 15.2 Measure text rendering overhead
  - [ ] 15.3 Ensure < 50ms total overhead
  - [ ] 15.4 Profile and optimize if needed

---

## Task Dependencies

```
1 → 2, 3, 4
2, 3, 4 → 5
5 → 6
6 → 7
7 → 8, 9
8, 9 → 10, 11, 12
10, 11, 12 → 13, 14, 15
```

---

## Estimated Timeline

- **Phase 1**: 2-3 days
- **Phase 2**: 2-3 days
- **Phase 3**: 1-2 days
- **Phase 4**: 2-3 days
- **Phase 5**: 1-2 days

**Total**: 8-13 days (1.5-2.5 weeks)

---

## Success Criteria

- [ ] All tasks completed and marked as done
- [ ] All tests passing (unit, integration, property)
- [ ] JSON parse success rate > 90% (measured with real models)
- [ ] Backward compatibility: 100% of existing tests pass
- [ ] Code quality: No flake8 errors, properly formatted
- [ ] Documentation: Complete and accurate
- [ ] Performance: < 50ms overhead for JSON parsing

---

## Notes

- Maintain backward compatibility throughout implementation
- Test with both Chinese and English prompts
- Log parse failures for analysis and improvement
- Consider adding examples to prompts if parse rate is low
- Monitor performance and optimize if needed
