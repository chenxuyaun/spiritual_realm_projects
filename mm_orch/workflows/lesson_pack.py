"""
LessonPack Workflow Implementation.

This module implements the LessonPack workflow which generates
educational content packages including:
1. Teaching plan/outline
2. Detailed explanation content
3. Practice exercises with answers

The workflow ensures output conforms to the LessonPack data structure.

Supports both mock model manager and real model integration via
RealModelManager and InferenceEngine.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import time
import re

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType, LessonPack
from mm_orch.exceptions import ValidationError, WorkflowError
from mm_orch.logger import get_logger

if TYPE_CHECKING:
    from mm_orch.runtime.real_model_manager import RealModelManager
    from mm_orch.runtime.inference_engine import InferenceEngine


logger = get_logger(__name__)


# Prompt templates for LessonPack with real models
LESSON_PLAN_PROMPT_ZH = """你是一位专业的教育内容设计师。请为以下主题生成一个结构化的教学计划大纲。

主题: {topic}
难度级别: {difficulty}

请生成包含以下部分的教学计划（使用Markdown格式）:
1. 学习目标（3-5个具体目标）
2. 前置知识要求
3. 主要内容点（3-5个）
4. 教学重点和难点
5. 预计学习时间

教学计划:"""

LESSON_PLAN_PROMPT_EN = """You are a professional educational content designer. Please generate a structured teaching plan outline for the following topic.

Topic: {topic}
Difficulty Level: {difficulty}

Please generate a teaching plan including the following sections (use Markdown format):
1. Learning objectives (3-5 specific objectives)
2. Prerequisites
3. Main content points (3-5)
4. Key points and difficulties
5. Estimated learning time

Teaching Plan:"""

LESSON_EXPLANATION_PROMPT_ZH = """你是一位专业的教育内容设计师。基于以下教学计划，为主题"{topic}"生成详细的讲解内容。

教学计划:
{plan}

要求:
- 使用Markdown格式
- 内容清晰、易懂
- 适合{difficulty}水平的学习者
{example_instruction}

讲解内容:"""

LESSON_EXPLANATION_PROMPT_EN = """You are a professional educational content designer. Based on the following teaching plan, generate detailed explanation content for the topic "{topic}".

Teaching Plan:
{plan}

Requirements:
- Use Markdown format
- Content should be clear and easy to understand
- Suitable for {difficulty} level learners
{example_instruction}

Explanation:"""

LESSON_EXERCISES_PROMPT_ZH = """你是一位专业的教育内容设计师。为主题"{topic}"生成{num_exercises}道练习题，难度为{difficulty}。

每道题目需要包含:
1. 问题（清晰明确）
2. 答案（详细解答）

请按以下格式输出:
问题1: [问题内容]
答案1: [答案内容]

问题2: [问题内容]
答案2: [答案内容]

练习题:"""

LESSON_EXERCISES_PROMPT_EN = """You are a professional educational content designer. Generate {num_exercises} practice exercises for the topic "{topic}" at {difficulty} level.

Each exercise should include:
1. Question (clear and specific)
2. Answer (detailed solution)

Please output in the following format:
Question 1: [question content]
Answer 1: [answer content]

Question 2: [question content]
Answer 2: [answer content]

Exercises:"""


@dataclass
class LessonPackStep:
    """Tracks the execution of a workflow step."""

    name: str
    success: bool
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class LessonPackContext:
    """Context for LessonPack workflow execution."""

    topic: str
    difficulty: str = "intermediate"
    num_exercises: int = 3
    language: str = "zh"
    plan: str = ""
    explanation: str = ""
    exercises: List[Dict[str, str]] = field(default_factory=list)
    steps: List[LessonPackStep] = field(default_factory=list)

    def add_step(self, step: LessonPackStep) -> None:
        """Add a step to the execution history."""
        self.steps.append(step)


class LessonPackWorkflow(BaseWorkflow):
    """
    LessonPack Workflow: Topic → Plan → Explanation → Exercises

    This workflow generates educational content packages by:
    1. Generating a structured teaching plan/outline
    2. Expanding the plan into detailed explanation content
    3. Creating practice exercises with answers

    The output conforms to the LessonPack data structure with:
    - topic: The lesson topic
    - plan: Teaching plan/outline
    - explanation: Detailed explanation content
    - exercises: List of {question, answer} dictionaries

    Supports real model integration via RealModelManager and InferenceEngine.

    Attributes:
        workflow_type: WorkflowType.LESSON_PACK
        name: "LessonPack"

    Properties verified:
        - Property 6: 教学包结构完整性

    Requirements:
        - 3.1: Generate structured teaching plan
        - 3.2: Generate detailed explanation content
        - 3.3: Generate practice exercises with answers
        - 3.5: Organize content into LessonPack format
    """

    workflow_type = WorkflowType.LESSON_PACK
    name = "LessonPack"
    description = "Educational content package generation workflow"

    def __init__(
        self,
        model_manager: Optional[Any] = None,
        real_model_manager: Optional["RealModelManager"] = None,
        inference_engine: Optional["InferenceEngine"] = None,
        generator_model: str = "gpt2",
        default_num_exercises: int = 3,
        max_plan_length: int = 1000,
        max_explanation_length: int = 3000,
        use_real_models: bool = False,
    ):
        """
        Initialize the LessonPack workflow.

        Args:
            model_manager: Model manager for content generation (mock)
            real_model_manager: Real model manager for actual LLM inference
            inference_engine: Inference engine for real model generation
            generator_model: Model name for content generation
            default_num_exercises: Default number of exercises to generate
            max_plan_length: Maximum length for teaching plan
            max_explanation_length: Maximum length for explanation
            use_real_models: Whether to use real models instead of mock
        """
        super().__init__()
        self.model_manager = model_manager
        self.real_model_manager = real_model_manager
        self.inference_engine = inference_engine
        self.generator_model = generator_model
        self.default_num_exercises = default_num_exercises
        self.max_plan_length = max_plan_length
        self.max_explanation_length = max_explanation_length
        self.use_real_models = use_real_models

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for this workflow."""
        return ["topic"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        """Return optional parameters with defaults."""
        return {
            "difficulty": "intermediate",
            "num_exercises": self.default_num_exercises,
            "language": "zh",
            "include_examples": True,
        }

    def get_required_models(self) -> List[str]:
        """Return the list of models required by this workflow."""
        return [self.generator_model]

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate workflow parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If parameters are invalid
        """
        self._validate_required_parameters(parameters)

        topic = parameters.get("topic", "")
        if not topic or not topic.strip():
            raise ValidationError("Topic cannot be empty")

        num_exercises = parameters.get("num_exercises", self.default_num_exercises)
        if not isinstance(num_exercises, int) or num_exercises < 1:
            raise ValidationError("num_exercises must be a positive integer")

        difficulty = parameters.get("difficulty", "intermediate")
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        if difficulty not in valid_difficulties:
            raise ValidationError(f"difficulty must be one of {valid_difficulties}")

        return True

    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the LessonPack workflow.

        Steps:
        1. Plan: Generate teaching plan/outline
        2. Explain: Generate detailed explanation content
        3. Exercise: Generate practice exercises with answers

        Args:
            parameters: Workflow parameters including 'topic'

        Returns:
            WorkflowResult with LessonPack data and metadata
        """
        topic = parameters["topic"]
        difficulty = parameters.get("difficulty", "intermediate")
        num_exercises = parameters.get("num_exercises", self.default_num_exercises)
        language = parameters.get("language", "zh")
        include_examples = parameters.get("include_examples", True)

        # Initialize context
        ctx = LessonPackContext(
            topic=topic, difficulty=difficulty, num_exercises=num_exercises, language=language
        )

        try:
            # Step 1: Generate teaching plan
            ctx = self._step_generate_plan(ctx)

            if not ctx.plan:
                return self._create_result(
                    ctx, status="failed", error="Failed to generate teaching plan"
                )

            # Step 2: Generate explanation content
            ctx = self._step_generate_explanation(ctx, include_examples)

            if not ctx.explanation:
                return self._create_result(
                    ctx, status="partial", error="Failed to generate explanation content"
                )

            # Step 3: Generate exercises
            ctx = self._step_generate_exercises(ctx)

            if not ctx.exercises:
                return self._create_result(
                    ctx, status="partial", error="Failed to generate exercises"
                )

            return self._create_result(ctx, status="success")

        except Exception as e:
            logger.error("LessonPack workflow failed", error=str(e), topic=topic[:50])
            return self._create_result(
                ctx, status="partial" if ctx.plan else "failed", error=str(e)
            )

    def _step_generate_plan(self, ctx: LessonPackContext) -> LessonPackContext:
        """
        Step 1: Generate teaching plan/outline.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with teaching plan
        """
        start_time = time.time()
        step = LessonPackStep(name="generate_plan", success=False)

        try:
            logger.info(f"Step 1: Generating teaching plan for '{ctx.topic[:50]}...'")

            plan = self._generate_plan(ctx.topic, ctx.difficulty, ctx.language)
            ctx.plan = plan
            step.success = bool(plan)

            logger.info(f"Generated plan of length {len(plan)}")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Plan generation failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_generate_explanation(
        self, ctx: LessonPackContext, include_examples: bool
    ) -> LessonPackContext:
        """
        Step 2: Generate detailed explanation content.

        Args:
            ctx: Workflow context with plan
            include_examples: Whether to include examples

        Returns:
            Updated context with explanation
        """
        start_time = time.time()
        step = LessonPackStep(name="generate_explanation", success=False)

        try:
            logger.info(f"Step 2: Generating explanation for '{ctx.topic[:50]}...'")

            explanation = self._generate_explanation(
                ctx.topic, ctx.plan, ctx.difficulty, ctx.language, include_examples
            )
            ctx.explanation = explanation
            step.success = bool(explanation)

            logger.info(f"Generated explanation of length {len(explanation)}")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Explanation generation failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_generate_exercises(self, ctx: LessonPackContext) -> LessonPackContext:
        """
        Step 3: Generate practice exercises with answers.

        Args:
            ctx: Workflow context with plan and explanation

        Returns:
            Updated context with exercises
        """
        start_time = time.time()
        step = LessonPackStep(name="generate_exercises", success=False)

        try:
            logger.info(f"Step 3: Generating {ctx.num_exercises} exercises")

            exercises = self._generate_exercises(
                ctx.topic, ctx.plan, ctx.difficulty, ctx.num_exercises, ctx.language
            )
            ctx.exercises = exercises
            step.success = len(exercises) > 0

            logger.info(f"Generated {len(exercises)} exercises")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Exercise generation failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _generate_plan(self, topic: str, difficulty: str, language: str) -> str:
        """
        Generate a teaching plan/outline for the topic.

        Args:
            topic: The lesson topic
            difficulty: Difficulty level
            language: Output language

        Returns:
            Teaching plan text
        """
        # Use real models if available and enabled
        if self.use_real_models and self.inference_engine:
            return self._generate_plan_with_real_model(topic, difficulty, language)

        # Build prompt for plan generation
        if language == "zh":
            prompt = f"""为以下主题生成一个结构化的教学计划大纲。

主题: {topic}
难度级别: {self._translate_difficulty(difficulty, language)}

请生成包含以下部分的教学计划:
1. 学习目标
2. 前置知识要求
3. 主要内容点（3-5个）
4. 教学重点和难点
5. 预计学习时间

教学计划:"""
        else:
            prompt = f"""Generate a structured teaching plan outline for the following topic.

Topic: {topic}
Difficulty Level: {difficulty}

Please generate a teaching plan including:
1. Learning objectives
2. Prerequisites
3. Main content points (3-5)
4. Key points and difficulties
5. Estimated learning time

Teaching Plan:"""

        # Use model if available
        if self.model_manager:
            try:
                plan = self.model_manager.infer(
                    self.generator_model, prompt, max_new_tokens=500, temperature=0.7
                )
                return self._clean_generated_text(
                    plan, "教学计划:" if language == "zh" else "Teaching Plan:"
                )
            except Exception as e:
                logger.warning(f"Model plan generation failed: {e}")

        # Fallback: generate template-based plan
        return self._generate_template_plan(topic, difficulty, language)

    def _generate_plan_with_real_model(self, topic: str, difficulty: str, language: str) -> str:
        """
        Generate teaching plan using real model via InferenceEngine.

        Args:
            topic: The lesson topic
            difficulty: Difficulty level
            language: Output language

        Returns:
            Generated teaching plan
        """
        try:
            # Select prompt template based on language
            if language == "zh":
                prompt = LESSON_PLAN_PROMPT_ZH.format(
                    topic=topic, difficulty=self._translate_difficulty(difficulty, language)
                )
            else:
                prompt = LESSON_PLAN_PROMPT_EN.format(topic=topic, difficulty=difficulty)

            # Generate using inference engine
            from mm_orch.runtime.inference_engine import GenerationConfig

            config = GenerationConfig(
                max_new_tokens=800, temperature=0.7, top_p=0.9, repetition_penalty=1.1
            )

            result = self.inference_engine.generate(prompt, config=config)
            plan = result.text.strip()

            # Validate Markdown format
            plan = self._validate_markdown_format(plan)

            logger.info(
                f"Generated plan with real model: {len(plan)} chars, "
                f"{result.tokens_per_second:.1f} tokens/s"
            )
            return plan

        except Exception as e:
            logger.error(f"Real model plan generation failed: {e}")
            # Fallback to template
            return self._generate_template_plan(topic, difficulty, language)

    def _generate_template_plan(self, topic: str, difficulty: str, language: str) -> str:
        """
        Generate a template-based teaching plan when model is unavailable.

        Args:
            topic: The lesson topic
            difficulty: Difficulty level
            language: Output language

        Returns:
            Template-based teaching plan
        """
        difficulty_text = self._translate_difficulty(difficulty, language)

        if language == "zh":
            return f"""# {topic} 教学计划

## 学习目标
- 理解{topic}的基本概念和原理
- 掌握{topic}的核心知识点
- 能够应用所学知识解决相关问题

## 前置知识要求
- 具备相关领域的基础知识
- 难度级别: {difficulty_text}

## 主要内容
1. {topic}的定义和背景
2. {topic}的核心概念
3. {topic}的应用场景
4. {topic}的实践方法

## 教学重点
- {topic}的核心原理
- 关键概念的理解和应用

## 教学难点
- 复杂概念的深入理解
- 理论与实践的结合

## 预计学习时间
- 基础学习: 30-45分钟
- 练习巩固: 15-30分钟"""
        else:
            return f"""# Teaching Plan: {topic}

## Learning Objectives
- Understand the basic concepts and principles of {topic}
- Master the core knowledge points of {topic}
- Apply learned knowledge to solve related problems

## Prerequisites
- Basic knowledge in the related field
- Difficulty Level: {difficulty}

## Main Content
1. Definition and background of {topic}
2. Core concepts of {topic}
3. Application scenarios of {topic}
4. Practical methods of {topic}

## Key Points
- Core principles of {topic}
- Understanding and application of key concepts

## Difficulties
- Deep understanding of complex concepts
- Combining theory with practice

## Estimated Learning Time
- Basic learning: 30-45 minutes
- Practice and consolidation: 15-30 minutes"""

    def _generate_explanation(
        self, topic: str, plan: str, difficulty: str, language: str, include_examples: bool
    ) -> str:
        """
        Generate detailed explanation content based on the plan.

        Args:
            topic: The lesson topic
            plan: Teaching plan
            difficulty: Difficulty level
            language: Output language
            include_examples: Whether to include examples

        Returns:
            Detailed explanation text
        """
        # Use real models if available and enabled
        if self.use_real_models and self.inference_engine:
            return self._generate_explanation_with_real_model(
                topic, plan, difficulty, language, include_examples
            )

        example_instruction = ""
        if include_examples:
            if language == "zh":
                example_instruction = "请在适当的地方加入具体的例子来帮助理解。"
            else:
                example_instruction = (
                    "Please include specific examples where appropriate to aid understanding."
                )

        if language == "zh":
            prompt = f"""基于以下教学计划，为主题"{topic}"生成详细的讲解内容。

教学计划:
{plan[:500]}

{example_instruction}

请生成清晰、易懂的讲解内容，适合{self._translate_difficulty(difficulty, language)}水平的学习者。

讲解内容:"""
        else:
            prompt = f"""Based on the following teaching plan, generate detailed explanation content for the topic "{topic}".

Teaching Plan:
{plan[:500]}

{example_instruction}

Please generate clear and easy-to-understand explanation content suitable for {difficulty} level learners.

Explanation:"""

        # Use model if available
        if self.model_manager:
            try:
                explanation = self.model_manager.infer(
                    self.generator_model, prompt, max_new_tokens=1000, temperature=0.7
                )
                return self._clean_generated_text(
                    explanation, "讲解内容:" if language == "zh" else "Explanation:"
                )
            except Exception as e:
                logger.warning(f"Model explanation generation failed: {e}")

        # Fallback: generate template-based explanation
        return self._generate_template_explanation(topic, difficulty, language, include_examples)

    def _generate_explanation_with_real_model(
        self, topic: str, plan: str, difficulty: str, language: str, include_examples: bool
    ) -> str:
        """
        Generate explanation using real model via InferenceEngine.

        Args:
            topic: The lesson topic
            plan: Teaching plan
            difficulty: Difficulty level
            language: Output language
            include_examples: Whether to include examples

        Returns:
            Generated explanation
        """
        try:
            # Build example instruction
            if include_examples:
                if language == "zh":
                    example_instruction = "- 在适当的地方加入具体的例子来帮助理解"
                else:
                    example_instruction = (
                        "- Include specific examples where appropriate to aid understanding"
                    )
            else:
                example_instruction = ""

            # Select prompt template based on language
            if language == "zh":
                prompt = LESSON_EXPLANATION_PROMPT_ZH.format(
                    topic=topic,
                    plan=plan[:1000],
                    difficulty=self._translate_difficulty(difficulty, language),
                    example_instruction=example_instruction,
                )
            else:
                prompt = LESSON_EXPLANATION_PROMPT_EN.format(
                    topic=topic,
                    plan=plan[:1000],
                    difficulty=difficulty,
                    example_instruction=example_instruction,
                )

            # Generate using inference engine
            from mm_orch.runtime.inference_engine import GenerationConfig

            config = GenerationConfig(
                max_new_tokens=1500, temperature=0.7, top_p=0.9, repetition_penalty=1.1
            )

            result = self.inference_engine.generate(prompt, config=config)
            explanation = result.text.strip()

            # Validate Markdown format
            explanation = self._validate_markdown_format(explanation)

            logger.info(
                f"Generated explanation with real model: {len(explanation)} chars, "
                f"{result.tokens_per_second:.1f} tokens/s"
            )
            return explanation

        except Exception as e:
            logger.error(f"Real model explanation generation failed: {e}")
            # Fallback to template
            return self._generate_template_explanation(
                topic, difficulty, language, include_examples
            )

    def _generate_template_explanation(
        self, topic: str, difficulty: str, language: str, include_examples: bool
    ) -> str:
        """
        Generate template-based explanation when model is unavailable.

        Args:
            topic: The lesson topic
            difficulty: Difficulty level
            language: Output language
            include_examples: Whether to include examples

        Returns:
            Template-based explanation
        """
        if language == "zh":
            explanation = f"""# {topic} 详细讲解

## 1. 概述

{topic}是一个重要的概念/技术/领域。理解{topic}对于掌握相关知识至关重要。

## 2. 核心概念

### 2.1 基本定义
{topic}可以被定义为一种用于解决特定问题或实现特定目标的方法/概念/技术。

### 2.2 关键特点
- 特点一：{topic}具有独特的特性
- 特点二：{topic}在实际应用中表现出色
- 特点三：{topic}与其他相关概念有密切联系

## 3. 工作原理

{topic}的工作原理基于以下几个核心要素：
1. 基础原理的应用
2. 关键步骤的执行
3. 结果的验证和优化

## 4. 应用场景

{topic}在以下场景中有广泛应用：
- 场景一：日常应用
- 场景二：专业领域
- 场景三：高级应用"""

            if include_examples:
                explanation += f"""

## 5. 实例说明

### 示例1
假设我们需要应用{topic}来解决一个具体问题。首先，我们需要理解问题的本质，然后按照{topic}的方法论进行分析和处理。

### 示例2
在实际工作中，{topic}可以帮助我们更高效地完成任务。通过正确应用{topic}的原则，我们可以获得更好的结果。"""
        else:
            explanation = f"""# Detailed Explanation: {topic}

## 1. Overview

{topic} is an important concept/technology/field. Understanding {topic} is crucial for mastering related knowledge.

## 2. Core Concepts

### 2.1 Basic Definition
{topic} can be defined as a method/concept/technology used to solve specific problems or achieve specific goals.

### 2.2 Key Features
- Feature 1: {topic} has unique characteristics
- Feature 2: {topic} performs well in practical applications
- Feature 3: {topic} is closely related to other related concepts

## 3. Working Principles

The working principles of {topic} are based on the following core elements:
1. Application of fundamental principles
2. Execution of key steps
3. Verification and optimization of results

## 4. Application Scenarios

{topic} has wide applications in the following scenarios:
- Scenario 1: Daily applications
- Scenario 2: Professional fields
- Scenario 3: Advanced applications"""

            if include_examples:
                explanation += f"""

## 5. Examples

### Example 1
Suppose we need to apply {topic} to solve a specific problem. First, we need to understand the nature of the problem, then analyze and process it according to the methodology of {topic}.

### Example 2
In practical work, {topic} can help us complete tasks more efficiently. By correctly applying the principles of {topic}, we can achieve better results."""

        return explanation

    def _generate_exercises(
        self, topic: str, plan: str, difficulty: str, num_exercises: int, language: str
    ) -> List[Dict[str, str]]:
        """
        Generate practice exercises with answers.

        Args:
            topic: The lesson topic
            plan: Teaching plan
            difficulty: Difficulty level
            num_exercises: Number of exercises to generate
            language: Output language

        Returns:
            List of exercise dictionaries with 'question' and 'answer' keys
        """
        # Use real models if available and enabled
        if self.use_real_models and self.inference_engine:
            return self._generate_exercises_with_real_model(
                topic, difficulty, num_exercises, language
            )

        if language == "zh":
            prompt = f"""为主题"{topic}"生成{num_exercises}道练习题，难度为{self._translate_difficulty(difficulty, language)}。

每道题目需要包含:
1. 问题
2. 答案

请按以下格式输出:
问题1: [问题内容]
答案1: [答案内容]

问题2: [问题内容]
答案2: [答案内容]

练习题:"""
        else:
            prompt = f"""Generate {num_exercises} practice exercises for the topic "{topic}" at {difficulty} level.

Each exercise should include:
1. Question
2. Answer

Please output in the following format:
Question 1: [question content]
Answer 1: [answer content]

Question 2: [question content]
Answer 2: [answer content]

Exercises:"""

        # Use model if available
        if self.model_manager:
            try:
                exercises_text = self.model_manager.infer(
                    self.generator_model, prompt, max_new_tokens=800, temperature=0.8
                )
                exercises = self._parse_exercises(exercises_text, language)
                if exercises:
                    return exercises[:num_exercises]
            except Exception as e:
                logger.warning(f"Model exercise generation failed: {e}")

        # Fallback: generate template-based exercises
        return self._generate_template_exercises(topic, difficulty, num_exercises, language)

    def _generate_exercises_with_real_model(
        self, topic: str, difficulty: str, num_exercises: int, language: str
    ) -> List[Dict[str, str]]:
        """
        Generate exercises using real model via InferenceEngine.

        Args:
            topic: The lesson topic
            difficulty: Difficulty level
            num_exercises: Number of exercises
            language: Output language

        Returns:
            List of exercise dictionaries
        """
        try:
            # Select prompt template based on language
            if language == "zh":
                prompt = LESSON_EXERCISES_PROMPT_ZH.format(
                    topic=topic,
                    num_exercises=num_exercises,
                    difficulty=self._translate_difficulty(difficulty, language),
                )
            else:
                prompt = LESSON_EXERCISES_PROMPT_EN.format(
                    topic=topic, num_exercises=num_exercises, difficulty=difficulty
                )

            # Generate using inference engine
            from mm_orch.runtime.inference_engine import GenerationConfig

            config = GenerationConfig(
                max_new_tokens=1000, temperature=0.8, top_p=0.9, repetition_penalty=1.1
            )

            result = self.inference_engine.generate(prompt, config=config)
            exercises_text = result.text.strip()

            # Parse exercises
            exercises = self._parse_exercises(exercises_text, language)

            logger.info(
                f"Generated {len(exercises)} exercises with real model, "
                f"{result.tokens_per_second:.1f} tokens/s"
            )

            if exercises:
                return exercises[:num_exercises]

        except Exception as e:
            logger.error(f"Real model exercise generation failed: {e}")

        # Fallback to template
        return self._generate_template_exercises(topic, difficulty, num_exercises, language)

    def _validate_markdown_format(self, content: str) -> str:
        """
        Validate and clean up Markdown format.

        Args:
            content: Generated content

        Returns:
            Validated and cleaned content
        """
        if not content:
            return content

        # Remove any leading/trailing whitespace
        content = content.strip()

        # Ensure headers have proper spacing
        content = re.sub(r"(#{1,6})([^\s#])", r"\1 \2", content)

        # Ensure list items have proper spacing
        content = re.sub(r"^(\s*[-*+])([^\s])", r"\1 \2", content, flags=re.MULTILINE)
        content = re.sub(r"^(\s*\d+\.)([^\s])", r"\1 \2", content, flags=re.MULTILINE)

        # Remove excessive blank lines (more than 2)
        content = re.sub(r"\n{4,}", "\n\n\n", content)

        # Remove any generation artifacts
        stop_markers = [
            "\n\n---\n\n",
            "\n\n请注意",
            "\n\nNote:",
            "\n\n[End",
        ]
        for marker in stop_markers:
            if marker in content:
                content = content.split(marker)[0].strip()

        return content

    def _parse_exercises(self, text: str, language: str) -> List[Dict[str, str]]:
        """
        Parse generated exercises text into structured format.

        Args:
            text: Generated exercises text
            language: Output language

        Returns:
            List of exercise dictionaries
        """
        exercises = []

        # Try to parse question-answer pairs
        if language == "zh":
            question_pattern = r"问题\s*\d*[:：]\s*(.+?)(?=答案|问题|$)"
            answer_pattern = r"答案\s*\d*[:：]\s*(.+?)(?=问题|答案|$)"
        else:
            question_pattern = r"Question\s*\d*[:：]?\s*(.+?)(?=Answer|Question|$)"
            answer_pattern = r"Answer\s*\d*[:：]?\s*(.+?)(?=Question|Answer|$)"

        questions = re.findall(question_pattern, text, re.IGNORECASE | re.DOTALL)
        answers = re.findall(answer_pattern, text, re.IGNORECASE | re.DOTALL)

        # Pair questions with answers
        for i, (q, a) in enumerate(zip(questions, answers)):
            q = q.strip()
            a = a.strip()
            if q and a:
                exercises.append({"question": q, "answer": a})

        return exercises

    def _generate_template_exercises(
        self, topic: str, difficulty: str, num_exercises: int, language: str
    ) -> List[Dict[str, str]]:
        """
        Generate template-based exercises when model is unavailable.

        Args:
            topic: The lesson topic
            difficulty: Difficulty level
            num_exercises: Number of exercises
            language: Output language

        Returns:
            List of template-based exercises
        """
        exercises = []

        if language == "zh":
            templates = [
                {
                    "question": f"请简要说明{topic}的基本概念和定义。",
                    "answer": f"{topic}是指在特定领域中用于解决问题或实现目标的方法、概念或技术。它具有独特的特性和应用价值。",
                },
                {
                    "question": f"{topic}的主要特点有哪些？请列举至少三点。",
                    "answer": f"{topic}的主要特点包括：1) 具有明确的目标和应用场景；2) 遵循特定的原则和方法论；3) 可以与其他相关概念结合使用；4) 在实践中不断发展和完善。",
                },
                {
                    "question": f"请举例说明{topic}在实际中的应用场景。",
                    "answer": f"{topic}在实际中有广泛的应用，例如：在日常工作中可以用于提高效率；在专业领域可以用于解决复杂问题；在学习过程中可以帮助理解相关概念。",
                },
                {
                    "question": f"学习{topic}需要具备哪些前置知识？",
                    "answer": f"学习{topic}通常需要具备以下前置知识：1) 相关领域的基础概念；2) 基本的分析和思考能力；3) 一定的实践经验或案例了解。",
                },
                {
                    "question": f"请描述{topic}的工作原理或核心机制。",
                    "answer": f"{topic}的工作原理基于以下核心机制：首先，明确问题或目标；其次，应用相关的方法和技术；最后，验证结果并进行优化。整个过程需要遵循特定的步骤和原则。",
                },
            ]
        else:
            templates = [
                {
                    "question": f"Please briefly explain the basic concept and definition of {topic}.",
                    "answer": f"{topic} refers to a method, concept, or technology used to solve problems or achieve goals in a specific field. It has unique characteristics and application value.",
                },
                {
                    "question": f"What are the main features of {topic}? Please list at least three.",
                    "answer": f"The main features of {topic} include: 1) Clear goals and application scenarios; 2) Following specific principles and methodologies; 3) Can be combined with other related concepts; 4) Continuously developing and improving in practice.",
                },
                {
                    "question": f"Please give examples of practical applications of {topic}.",
                    "answer": f"{topic} has wide applications in practice, such as: improving efficiency in daily work; solving complex problems in professional fields; helping understand related concepts in the learning process.",
                },
                {
                    "question": f"What prerequisite knowledge is needed to learn {topic}?",
                    "answer": f"Learning {topic} usually requires the following prerequisite knowledge: 1) Basic concepts in related fields; 2) Basic analytical and thinking skills; 3) Some practical experience or case understanding.",
                },
                {
                    "question": f"Please describe the working principle or core mechanism of {topic}.",
                    "answer": f"The working principle of {topic} is based on the following core mechanisms: First, clarify the problem or goal; Second, apply relevant methods and techniques; Finally, verify results and optimize. The entire process needs to follow specific steps and principles.",
                },
            ]

        # Select exercises based on num_exercises
        for i in range(min(num_exercises, len(templates))):
            exercises.append(templates[i])

        return exercises

    def _translate_difficulty(self, difficulty: str, language: str) -> str:
        """
        Translate difficulty level to the target language.

        Args:
            difficulty: Difficulty level in English
            language: Target language

        Returns:
            Translated difficulty level
        """
        if language == "zh":
            translations = {"beginner": "初级", "intermediate": "中级", "advanced": "高级"}
            return translations.get(difficulty, "中级")
        return difficulty

    def _clean_generated_text(self, text: str, prefix: str) -> str:
        """
        Clean up generated text by removing prompt artifacts.

        Args:
            text: Generated text
            prefix: Prefix to remove if present

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove the prefix if it appears at the start
        if prefix and text.startswith(prefix):
            text = text[len(prefix) :]

        # Also try to find and remove the prefix anywhere
        if prefix and prefix in text:
            parts = text.split(prefix)
            if len(parts) > 1:
                text = parts[-1]

        return text.strip()

    def _create_result(
        self, ctx: LessonPackContext, status: str = "success", error: Optional[str] = None
    ) -> WorkflowResult:
        """
        Create the workflow result.

        Args:
            ctx: Workflow context
            status: Result status
            error: Error message if any

        Returns:
            WorkflowResult object
        """
        metadata = {
            "workflow": self.name,
            "topic": ctx.topic,
            "difficulty": ctx.difficulty,
            "language": ctx.language,
            "num_exercises_requested": ctx.num_exercises,
            "num_exercises_generated": len(ctx.exercises),
            "steps": [
                {"name": s.name, "success": s.success, "duration": s.duration, "error": s.error}
                for s in ctx.steps
            ],
        }

        # Build result as LessonPack-compatible dictionary
        result = None
        if ctx.plan:
            result = {
                "topic": ctx.topic,
                "plan": ctx.plan,
                "explanation": ctx.explanation,
                "exercises": ctx.exercises,
            }

            # Validate the result structure matches LessonPack requirements
            # Property 6: 教学包结构完整性
            if not ctx.explanation:
                result["explanation"] = ""
            if not ctx.exercises:
                result["exercises"] = []

        return WorkflowResult(result=result, metadata=metadata, status=status, error=error)

    def create_lesson_pack(self, ctx: LessonPackContext) -> Optional[LessonPack]:
        """
        Create a LessonPack dataclass from the context.

        This method creates a validated LessonPack object from the workflow
        context. It ensures the output conforms to the LessonPack data structure.

        Args:
            ctx: Workflow context with generated content

        Returns:
            LessonPack object if valid, None otherwise
        """
        if not ctx.plan or not ctx.explanation or not ctx.exercises:
            return None

        try:
            return LessonPack(
                topic=ctx.topic,
                plan=ctx.plan,
                explanation=ctx.explanation,
                exercises=ctx.exercises,
                metadata={"difficulty": ctx.difficulty, "language": ctx.language},
            )
        except ValueError as e:
            logger.error(f"Failed to create LessonPack: {e}")
            return None
