# Implementation Roadmap - 2026-01-30

**Created**: 2026-01-30  
**Status**: Planning  
**Based on**: 需求文档迭代8.md

---

## Executive Summary

Based on the latest requirements (需求文档迭代8.md), this roadmap outlines the next implementation phase for the MuAI Orchestration System. The system has completed all 6 major specs and is production-ready (v1.0.0-rc1). The next phase focuses on two key enhancements:

1. **Lesson Pack Structured JSON Output** - NEW SPEC CREATED ✅
2. **Router v3 Mode Chat Feature** - ALREADY IMPLEMENTED ✅

---

## Current System Status

### Completed Work (100%)

| Spec | Status | Tasks | Tests |
|------|--------|-------|-------|
| OpenVINO Backend Integration | ✅ Complete | 15/15 | 100% |
| Real Model Integration | ✅ Complete | 12/12 | 100% |
| Consciousness System Deepening | ✅ Complete | 18/18 | 100% |
| Extensible Orchestration Phase B | ✅ Complete | 24/24 | 100% |
| MuAI Orchestration System | ✅ Complete | 20/20 | 100% |
| Advanced Optimization & Monitoring | ✅ Complete | 23/23 | 100% |

**Overall**: 122/122 tasks complete, 2,264/2,300 tests passing (98.4%)

---

## Next Implementation Phase

### Priority 1: Lesson Pack Structured JSON Output

**Status**: ✅ Spec Created  
**Location**: `.kiro/specs/lesson-pack-structured-output/`  
**Priority**: High  
**Estimated Duration**: 1.5-2.5 weeks

#### Overview

Enhance the LessonPack workflow to generate structured JSON output with teaching sections (导入、新授、练习、小结) instead of a single block of explanation text.

#### Key Features

1. **Structured JSON Format**
   - Topic and grade level
   - Array of teaching sections
   - Each section has: name, teacher_say, and optional fields (examples, questions, key_points, tips)

2. **Backward Compatibility**
   - Maintains existing `explanation` field (plain text)
   - Adds new `lesson_explain_structured` field (JSON)
   - All existing tests continue to pass

3. **CLI Enhancements**
   - `--structured` flag: Display structured view
   - `--format json`: Output raw JSON
   - `--format text`: Output formatted text (default)

4. **Validation Utilities**
   - Check minimum number of sections
   - Verify key section names exist
   - Count examples and questions
   - Include validation stats in metadata

#### Implementation Tasks

**Phase 1: Core Implementation** (2-3 days)
- Extend LessonPackContext with structured field
- Implement JSON parsing utilities
- Implement text rendering utilities
- Create structured prompt templates

**Phase 2: Workflow Integration** (2-3 days)
- Modify explanation generation step
- Update workflow result creation
- Implement validation utilities

**Phase 3: CLI Integration** (1-2 days)
- Add CLI options for structured output
- Implement CLI display logic

**Phase 4: Testing and Validation** (2-3 days)
- Write unit tests
- Write integration tests
- Write property-based tests

**Phase 5: Documentation and Polish** (1-2 days)
- Update documentation
- Code quality and polish
- Performance testing

#### Success Criteria

- ✅ JSON parse success rate > 90%
- ✅ Backward compatibility: 100% of existing tests pass
- ✅ All new tests passing
- ✅ Performance: < 50ms overhead
- ✅ Documentation complete

#### Files to Modify

- `mm_orch/workflows/lesson_pack.py` - Main implementation
- `mm_orch/main.py` - CLI integration
- `tests/unit/test_lesson_pack.py` - Unit tests
- `tests/integration/test_lesson_pack_integration.py` - Integration tests
- `tests/property/test_lesson_pack_properties.py` - Property tests
- `README.md` - Documentation

---

### Priority 2: Router v3 Mode Chat Feature

**Status**: ✅ ALREADY IMPLEMENTED  
**Location**: `mm_orch/routing/router_v3.py`, `scripts/train_router_v3.py`  
**Priority**: High  
**Action Required**: Verification and Testing

#### Current Implementation

The Router v3 already includes mode_chat feature support:

1. **Mode Feature Extraction** (Requirement 21.1)
   - Extracts `mode` from `State.meta`
   - Supports "chat" and "default" modes

2. **One-Hot Encoding** (Requirement 21.2)
   - Encodes mode as binary feature: `mode_is_chat` (1=chat, 0=default)

3. **Training Integration** (Requirement 21.3)
   - `train_router_v3.py` extracts mode from trace metadata
   - Creates `mode_is_chat` feature during training
   - Concatenates with TF-IDF text features
   - Documents mode feature in model metadata

4. **Prediction Integration** (Requirement 21.4)
   - Router uses mode features during prediction
   - Concatenates text features with mode feature
   - Makes cost-aware routing decisions

#### Verification Tasks

- [ ] 1. Verify mode extraction from State.meta
- [ ] 2. Test router with mode="chat" vs mode="default"
- [ ] 3. Verify training script handles mode features correctly
- [ ] 4. Test end-to-end: chat mode → chat_generate workflow
- [ ] 5. Test end-to-end: default mode → search_qa workflow
- [ ] 6. Measure routing accuracy with mode features
- [ ] 7. Document mode feature usage in README

#### Estimated Duration

1-2 days for verification and testing

---

## Implementation Schedule

### Week 1 (Days 1-5)
- **Days 1-2**: Router v3 verification and testing
- **Days 3-5**: Lesson Pack Phase 1 (Core Implementation)

### Week 2 (Days 6-10)
- **Days 6-7**: Lesson Pack Phase 2 (Workflow Integration)
- **Days 8-9**: Lesson Pack Phase 3 (CLI Integration)
- **Day 10**: Lesson Pack Phase 4 start (Testing)

### Week 3 (Days 11-13)
- **Days 11-12**: Lesson Pack Phase 4 complete (Testing)
- **Day 13**: Lesson Pack Phase 5 (Documentation and Polish)

**Total Duration**: 2-3 weeks

---

## Risk Assessment

### Risk 1: Low JSON Parse Success Rate
**Impact**: High  
**Probability**: Medium  
**Mitigation**:
- Carefully engineer prompts with examples
- Test with multiple models (GPT-2, Qwen)
- Implement robust fallback to plain text
- Log parse failures for analysis and improvement

### Risk 2: Breaking Changes
**Impact**: High  
**Probability**: Low  
**Mitigation**:
- Maintain backward compatibility throughout
- Run full test suite before each commit
- Add new fields without removing old ones
- Document migration path clearly

### Risk 3: Performance Degradation
**Impact**: Medium  
**Probability**: Low  
**Mitigation**:
- Profile JSON parsing overhead
- Optimize parsing logic if needed
- Cache rendered text
- Monitor performance metrics

---

## Success Metrics

### Lesson Pack Structured Output
- **JSON Parse Success Rate**: > 90%
- **Backward Compatibility**: 100% of existing tests pass
- **Test Coverage**: 100% for new code
- **Performance Overhead**: < 50ms
- **Documentation**: Complete and accurate

### Router v3 Mode Chat
- **Routing Accuracy**: Improved with mode features
- **Chat Mode Detection**: 100% accurate
- **Integration**: Seamless with existing workflows
- **Documentation**: Clear usage examples

---

## Next Steps After This Phase

### Short-term (1-2 months)
1. **Production Deployment**
   - Docker containerization
   - Kubernetes deployment
   - Monitoring and logging setup

2. **Performance Optimization**
   - Inference latency optimization
   - Memory usage optimization
   - Throughput improvements

3. **Additional Features**
   - Web UI for lesson pack rendering
   - Export to PDF/DOCX formats
   - Interactive lesson editing

### Medium-term (3-6 months)
1. **Multi-tenancy Support**
   - Tenant management
   - Resource isolation
   - Quota management

2. **Distributed Deployment**
   - Multi-node coordination
   - Distributed caching
   - Cross-region deployment

3. **Advanced AI Features**
   - Multi-modal support
   - Streaming output
   - Function calling (tool use)

---

## Resources Required

### Development
- **Time**: 2-3 weeks for lesson pack structured output
- **Personnel**: 1-2 developers
- **Hardware**: Standard development environment

### Testing
- **GPU**: For real model testing (T4 or better)
- **Time**: Included in development timeline
- **Test Data**: Existing test suite + new test cases

### Documentation
- **Time**: 1-2 days
- **Personnel**: 1 technical writer (or developer)

---

## Approval and Sign-off

This roadmap is ready for review and approval. Once approved, implementation can begin immediately.

**Recommended Action**: Start with Router v3 verification (1-2 days), then proceed with Lesson Pack Structured Output implementation (1.5-2.5 weeks).

---

## References

- **Requirements**: `docs/需求文档迭代8.md`
- **New Spec**: `.kiro/specs/lesson-pack-structured-output/`
- **Current Implementation**: `mm_orch/workflows/lesson_pack.py`, `mm_orch/routing/router_v3.py`
- **Project Status**: `docs/project_status_2026_01_28.md`
- **Previous Roadmap**: `docs/next_steps_2026_01_28.md`

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-30  
**Next Review**: After Phase 1 completion
