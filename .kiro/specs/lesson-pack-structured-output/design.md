# Lesson Pack Structured Output - Design Document

**Feature Name**: lesson-pack-structured-output  
**Created**: 2026-01-30  
**Status**: Draft  
**Version**: 1.0

---

## Architecture Overview

This feature enhances the LessonPackWorkflow to generate structured JSON output with teaching sections. The design maintains backward compatibility while adding new structured capabilities.

### Key Components

1. **LessonPackContext**: Extended with `lesson_explain_structured` field
2. **Prompt Templates**: Modified to request JSON output
3. **JSON Parser**: New utility to parse and validate JSON
4. **Text Renderer**: Converts structured JSON to plain text
5. **CLI Display**: Options for structured vs plain text output
6. **Validation Utilities**: Check structured output quality

---

## Data Models

### Structured Lesson JSON Schema

```python
StructuredLesson = {
    "topic": str,           # Required: Lesson topic
    "grade": str,           # Optional: Grade level (e.g., "小学3年级")
    "sections": [           # Required: Array of teaching sections
        {
            "name": str,                    # Required: Section name
            "teacher_say": str,             # Required: Teacher's words
            "student_may_say": str,         # Optional: Expected student response
            "examples": List[str],          # Optional: Example problems
            "questions": List[str],         # Optional: Practice questions
            "key_points": List[str],        # Optional: Key takeaways
            "tips": str                     # Optional: Teaching tips
        }
    ]
}
```

### Extended LessonPackContext

```python
@dataclass
class LessonPackContext:
    topic: str
    difficulty: str = "intermediate"
    num_exercises: int = 3
    language: str = "zh"
    plan: str = ""
    explanation: str = ""                              # Plain text (backward compat)
    lesson_explain_structured: Dict[str, Any] = None   # NEW: Structured JSON
    exercises: List[Dict[str, str]] = field(default_factory=list)
    steps: List[LessonPackStep] = field(default_factory=list)
```

---

## Implementation Design

### 1. Prompt Engineering


#### Modified Explanation Prompt (Chinese)

```python
LESSON_EXPLANATION_STRUCTURED_PROMPT_ZH = """你是一名小学老师，请根据下面的教学设计生成一份结构化的课堂讲解方案。

【课题】
{lesson_topic}

【教学目标】
{lesson_objectives}

【教学流程】
{lesson_outline}

【板书设计】
{board_plan}

要求：
1. 输出为一个合法的 JSON 对象，不要添加任何解释文字或多余内容。
2. JSON 顶层字段：
   - "topic": string，课题名称
   - "grade": string，可以根据描述猜测，例如 "小学X年级"
   - "sections": 数组，每个元素是一个环节对象，结构如下：
     - "name": string，环节名称，例如 "导入"、"新授"、"练习"、"小结"
     - "teacher_say": string，老师在这一环节的讲解与引导用语
     - 可选字段：
       - "student_may_say": string，可能的学生回应或思考
       - "examples": 数组，例题或示例描述
       - "questions": 数组，课堂提问或练习题
       - "key_points": 数组，本环节需要强调的要点
       - "tips": string，教学提示

请只输出 JSON，不要包含反引号、注释或其它文本。
"""
```

### 2. JSON Parsing Logic


```python
def _parse_structured_explanation(self, text: str) -> Optional[Dict[str, Any]]:
    """
    Parse generated text as structured JSON.
    
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Clean text
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try to parse JSON
    try:
        data = json.loads(text)
        
        # Validate required fields
        if not isinstance(data, dict):
            return None
        if "sections" not in data or not isinstance(data["sections"], list):
            return None
        
        # Validate each section
        for section in data["sections"]:
            if not isinstance(section, dict):
                return None
            if "name" not in section or "teacher_say" not in section:
                return None
        
        return data
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
        return None
```

### 3. Text Rendering

```python
def _render_structured_to_text(self, structured: Dict[str, Any]) -> str:
    """
    Render structured JSON to plain text explanation.
    
    Args:
        structured: Structured lesson JSON
        
    Returns:
        Plain text explanation
    """
    lines = []
    
    # Add topic header
    topic = structured.get("topic", "")
    grade = structured.get("grade", "")
    if topic:
        lines.append(f"# {topic}")
        if grade:
            lines.append(f"**年级**: {grade}")
        lines.append("")
    
    # Render each section
    for i, section in enumerate(structured.get("sections", []), 1):
        name = section.get("name", f"环节{i}")
        lines.append(f"## {name}")
        lines.append("")
        
        # Teacher's words
        teacher_say = section.get("teacher_say", "")
        if teacher_say:
            lines.append(teacher_say)
            lines.append("")
        
        # Student response
        student_may_say = section.get("student_may_say")
        if student_may_say:
            lines.append(f"**学生可能的回应**: {student_may_say}")
            lines.append("")
        
        # Examples
        examples = section.get("examples", [])
        if examples:
            lines.append("**例题**:")
            for j, ex in enumerate(examples, 1):
                lines.append(f"{j}. {ex}")
            lines.append("")
        
        # Questions
        questions = section.get("questions", [])
        if questions:
            lines.append("**练习题**:")
            for j, q in enumerate(questions, 1):
                lines.append(f"{j}. {q}")
            lines.append("")
        
        # Key points
        key_points = section.get("key_points", [])
        if key_points:
            lines.append("**要点**:")
            for point in key_points:
                lines.append(f"- {point}")
            lines.append("")
        
        # Tips
        tips = section.get("tips")
        if tips:
            lines.append(f"**提示**: {tips}")
            lines.append("")
    
    return "\n".join(lines)
```

### 4. Modified Explanation Generation Step


```python
def _step_generate_explanation(
    self, ctx: LessonPackContext, include_examples: bool
) -> LessonPackContext:
    """
    Step 2: Generate detailed explanation content (structured JSON).
    """
    start_time = time.time()
    step = LessonPackStep(name="generate_explanation", success=False)
    
    try:
        logger.info(f"Step 2: Generating structured explanation for '{ctx.topic[:50]}...'")
        
        # Generate structured JSON
        raw_text = self._generate_explanation_structured(
            ctx.topic, ctx.plan, ctx.difficulty, ctx.language, include_examples
        )
        
        # Try to parse as JSON
        structured = self._parse_structured_explanation(raw_text)
        
        if structured:
            # Success: Store structured JSON and render to text
            ctx.lesson_explain_structured = structured
            ctx.explanation = self._render_structured_to_text(structured)
            step.success = True
            logger.info(f"Generated structured explanation with {len(structured.get('sections', []))} sections")
        else:
            # Fallback: Use raw text as plain explanation
            ctx.explanation = raw_text
            ctx.lesson_explain_structured = None
            step.success = bool(raw_text)
            logger.warning("Failed to parse structured JSON, using plain text")
        
    except Exception as e:
        step.error = str(e)
        logger.error(f"Explanation generation failed: {e}")
    
    step.duration = time.time() - start_time
    ctx.add_step(step)
    return ctx
```

---

## Validation Design

### Validation Functions

```python
def validate_structured_lesson(structured: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate structured lesson output.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Check required fields
    if "sections" not in structured:
        results["valid"] = False
        results["errors"].append("Missing 'sections' field")
        return results
    
    sections = structured["sections"]
    
    # Check minimum sections
    if len(sections) < 3:
        results["warnings"].append(f"Only {len(sections)} sections (recommended: ≥3)")
    
    # Count section types
    section_names = [s.get("name", "") for s in sections]
    results["stats"]["num_sections"] = len(sections)
    results["stats"]["section_names"] = section_names
    
    # Check for key section types
    key_sections = ["导入", "新授", "练习", "小结"]
    found_sections = [name for name in key_sections if name in section_names]
    results["stats"]["key_sections_found"] = found_sections
    
    if len(found_sections) < 2:
        results["warnings"].append(f"Only {len(found_sections)} key sections found")
    
    # Count examples and questions
    total_examples = sum(len(s.get("examples", [])) for s in sections)
    total_questions = sum(len(s.get("questions", [])) for s in sections)
    
    results["stats"]["total_examples"] = total_examples
    results["stats"]["total_questions"] = total_questions
    
    if total_examples == 0:
        results["warnings"].append("No examples found")
    if total_questions == 0:
        results["warnings"].append("No questions found")
    
    # Validate each section
    for i, section in enumerate(sections):
        if "name" not in section:
            results["errors"].append(f"Section {i} missing 'name'")
            results["valid"] = False
        if "teacher_say" not in section:
            results["errors"].append(f"Section {i} missing 'teacher_say'")
            results["valid"] = False
    
    return results
```

---

## CLI Integration

### Command Line Options

```python
# In mm_orch/main.py

parser.add_argument(
    '--structured',
    action='store_true',
    help='Display structured JSON output for lesson_pack'
)

parser.add_argument(
    '--format',
    choices=['text', 'json'],
    default='text',
    help='Output format (text or json)'
)
```

### Display Logic


```python
def display_lesson_pack_result(result: WorkflowResult, args):
    """Display lesson pack result based on CLI options."""
    
    if args.format == 'json':
        # Output raw JSON
        structured = result.metadata.get('lesson_explain_structured')
        if structured:
            print(json.dumps(structured, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result.result, ensure_ascii=False, indent=2))
    
    elif args.structured and result.metadata.get('lesson_explain_structured'):
        # Display structured view
        structured = result.metadata['lesson_explain_structured']
        print("=== Structured Lesson Sections ===\n")
        
        for i, section in enumerate(structured.get('sections', []), 1):
            print(f"Section {i}: {section.get('name', 'Unnamed')}")
            print(f"  Teacher: {section.get('teacher_say', '')[:100]}...")
            
            if section.get('examples'):
                print(f"  Examples: {len(section['examples'])}")
            if section.get('questions'):
                print(f"  Questions: {len(section['questions'])}")
            if section.get('key_points'):
                print(f"  Key Points: {len(section['key_points'])}")
            print()
    
    else:
        # Default: Display plain text (backward compatible)
        print("=== Lesson Content ===")
        print(result.result.get('explanation', ''))
        print("\n=== Exercises ===")
        for i, ex in enumerate(result.result.get('exercises', []), 1):
            print(f"\nQuestion {i}: {ex.get('question', '')}")
            print(f"Answer {i}: {ex.get('answer', '')}")
```

---

## Testing Strategy

### Unit Tests

1. **Test JSON Parsing**
   - Valid JSON → Successful parse
   - Invalid JSON → Returns None
   - Missing required fields → Returns None
   - Markdown code blocks → Cleaned and parsed

2. **Test Text Rendering**
   - Structured JSON → Formatted text
   - All section types → Correct rendering
   - Optional fields → Handled gracefully

3. **Test Validation**
   - Valid structure → No errors
   - Missing sections → Warnings
   - Invalid sections → Errors
   - Statistics → Correct counts

### Integration Tests

1. **Test End-to-End Workflow**
   - Generate structured lesson → Valid JSON
   - Fallback to plain text → Works correctly
   - Both formats available → Backward compatible

2. **Test CLI Display**
   - `--format json` → Raw JSON output
   - `--structured` → Structured view
   - Default → Plain text (backward compat)

### Property-Based Tests

1. **Property: JSON Parse Idempotency**
   - Parse(Render(structured)) == structured

2. **Property: Backward Compatibility**
   - Old tests pass with new implementation

3. **Property: Validation Consistency**
   - Valid JSON always passes validation

---

## Correctness Properties

### Property 1: JSON Structure Validity
**Description**: If structured JSON is generated, it must conform to the schema

**Formal Statement**:
```
∀ structured_json ∈ GeneratedOutputs:
  structured_json ≠ None ⟹
    "sections" ∈ structured_json ∧
    ∀ section ∈ structured_json["sections"]:
      "name" ∈ section ∧ "teacher_say" ∈ section
```

**Test Strategy**: Property-based test with Hypothesis
- Generate random structured JSON
- Validate against schema
- Ensure all required fields present

**Validates**: Requirements 1.3, 2.1-2.5

---

### Property 2: Backward Compatibility
**Description**: Existing code expecting plain text explanation continues to work

**Formal Statement**:
```
∀ workflow_result ∈ Results:
  "explanation" ∈ workflow_result.result ∧
  workflow_result.result["explanation"] ≠ ""
```

**Test Strategy**: Run all existing lesson_pack tests
- Verify explanation field exists
- Verify explanation is non-empty string
- Verify exercises field unchanged

**Validates**: Requirements 3.1, 3.3

---

### Property 3: Dual Format Consistency
**Description**: Structured JSON and plain text represent the same content

**Formal Statement**:
```
∀ ctx ∈ Contexts where ctx.lesson_explain_structured ≠ None:
  Render(ctx.lesson_explain_structured) ≈ ctx.explanation
```

**Test Strategy**: Compare rendered text with explanation
- Extract key information from both formats
- Verify topic, sections, content match
- Allow for formatting differences

**Validates**: Requirements 1.1, 3.2

---

### Property 4: Validation Accuracy
**Description**: Validation correctly identifies valid and invalid structures

**Formal Statement**:
```
∀ structured ∈ ValidStructures:
  Validate(structured).valid == True

∀ structured ∈ InvalidStructures:
  Validate(structured).valid == False ∨
  len(Validate(structured).warnings) > 0
```

**Test Strategy**: Test with known valid/invalid examples
- Valid structures → valid=True, no errors
- Missing fields → valid=False, errors listed
- Suboptimal structures → warnings listed

**Validates**: Requirements 4.1-4.4

---

### Property 5: Parse-Render Roundtrip
**Description**: Parsing and rendering are inverse operations

**Formal Statement**:
```
∀ structured ∈ ValidStructures:
  Parse(Render(structured)) ≈ structured
```

**Test Strategy**: Roundtrip test
- Start with valid structured JSON
- Render to text
- Parse back to JSON
- Compare with original (allowing minor differences)

**Validates**: Requirements 1.2, 1.3

---

## Migration Plan

### Phase 1: Implementation (Week 1)
1. Add `lesson_explain_structured` field to LessonPackContext
2. Implement JSON parsing and validation functions
3. Implement text rendering function
4. Modify explanation generation to use structured prompts

### Phase 2: Testing (Week 1-2)
1. Write unit tests for new functions
2. Write integration tests for workflow
3. Write property-based tests
4. Verify backward compatibility

### Phase 3: CLI Integration (Week 2)
1. Add CLI flags for structured output
2. Implement display logic
3. Update documentation
4. Test CLI with various options

### Phase 4: Validation (Week 2)
1. Implement validation utilities
2. Add validation to workflow metadata
3. Create evaluation scripts
4. Document validation criteria

---

## Risks and Mitigations

### Risk 1: Low JSON Parse Success Rate
**Impact**: High  
**Probability**: Medium  
**Mitigation**:
- Carefully engineer prompts with examples
- Test with multiple models
- Implement robust fallback to plain text
- Log parse failures for analysis

### Risk 2: Breaking Changes
**Impact**: High  
**Probability**: Low  
**Mitigation**:
- Maintain backward compatibility
- Run full test suite before merge
- Add new fields without removing old ones
- Document migration path

### Risk 3: Performance Degradation
**Impact**: Medium  
**Probability**: Low  
**Mitigation**:
- Profile JSON parsing overhead
- Optimize parsing logic
- Cache rendered text
- Monitor performance metrics

---

## Future Enhancements

1. **Web UI Rendering**: Interactive lesson viewer
2. **Export Formats**: PDF, DOCX, HTML export
3. **Section Templates**: Predefined section types
4. **Multi-language**: Automatic translation of sections
5. **Interactive Editing**: Edit structured lessons in UI

---

## References

- 需求文档迭代8.md - Original requirements
- mm_orch/workflows/lesson_pack.py - Current implementation
- Python json module documentation
- Hypothesis property-based testing framework
