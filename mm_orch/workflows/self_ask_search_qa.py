"""
SelfAskSearchQA Workflow Implementation.

This module implements the SelfAskSearchQA workflow which performs:
1. Question decomposition - break complex questions into sub-questions
2. Iterative search - search for answers to each sub-question
3. Answer synthesis - combine sub-answers into a final comprehensive answer

This workflow is designed for complex questions that require multiple
pieces of information to answer comprehensively.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import re

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.exceptions import ValidationError, WorkflowError
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class SubQuestion:
    """Represents a decomposed sub-question."""

    question: str
    order: int
    depends_on: List[int] = field(default_factory=list)
    answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class SelfAskStep:
    """Tracks the execution of a workflow step."""

    name: str
    success: bool
    duration: float = 0.0
    input_count: int = 0
    output_count: int = 0
    error: Optional[str] = None


@dataclass
class SelfAskContext:
    """Context for SelfAskSearchQA workflow execution."""

    original_query: str
    sub_questions: List[SubQuestion] = field(default_factory=list)
    sub_answers: List[str] = field(default_factory=list)
    final_answer: str = ""
    steps: List[SelfAskStep] = field(default_factory=list)
    max_sub_questions: int = 5

    def add_step(self, step: SelfAskStep) -> None:
        """Add a step to the execution history."""
        self.steps.append(step)


class QuestionDecomposer:
    """
    Decomposes complex questions into simpler sub-questions.

    Uses pattern matching and heuristics to identify when a question
    can be broken down, and generates appropriate sub-questions.
    """

    # Patterns that indicate a complex question
    COMPLEX_PATTERNS = [
        r"\b(and|also|as well as|in addition)\b",
        r"\b(compare|contrast|difference|similar)\b",
        r"\b(how|why|what|when|where|who)\b.*\b(how|why|what|when|where|who)\b",
        r"\b(first|then|after|before|finally)\b",
        r"\b(relationship|connection|between)\b",
        r"\b(pros and cons|advantages and disadvantages)\b",
    ]

    def __init__(self, model_manager: Optional[Any] = None):
        """
        Initialize the question decomposer.

        Args:
            model_manager: Optional model manager for ML-based decomposition
        """
        self.model_manager = model_manager

    def is_complex_question(self, query: str) -> bool:
        """
        Determine if a question is complex enough to decompose.

        Args:
            query: The question to analyze

        Returns:
            True if the question should be decomposed
        """
        query_lower = query.lower()

        # Check for complex patterns
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                return True

        # Check for multiple question marks
        if query.count("?") > 1:
            return True

        # Check for long questions (likely complex)
        word_count = len(query.split())
        if word_count > 20:
            return True

        return False

    def decompose(self, query: str, max_questions: int = 5) -> List[SubQuestion]:
        """
        Decompose a complex question into sub-questions.

        Args:
            query: The complex question to decompose
            max_questions: Maximum number of sub-questions to generate

        Returns:
            List of SubQuestion objects
        """
        # Try model-based decomposition first
        if self.model_manager:
            try:
                return self._model_decompose(query, max_questions)
            except Exception as e:
                logger.warning(f"Model decomposition failed: {e}, using rule-based")

        # Fall back to rule-based decomposition
        return self._rule_based_decompose(query, max_questions)

    def _model_decompose(self, query: str, max_questions: int) -> List[SubQuestion]:
        """
        Use a language model to decompose the question.

        Args:
            query: The question to decompose
            max_questions: Maximum sub-questions

        Returns:
            List of SubQuestion objects
        """
        prompt = f"""Break down the following complex question into simpler sub-questions that can be answered independently. 
Each sub-question should focus on one specific aspect.
Return only the sub-questions, one per line, numbered.

Question: {query}

Sub-questions:"""

        response = self.model_manager.infer("gpt2", prompt, max_new_tokens=200, temperature=0.7)

        # Parse the response
        sub_questions = []
        lines = response.strip().split("\n")

        for i, line in enumerate(lines[:max_questions]):
            # Clean up the line
            line = line.strip()
            # Remove numbering if present
            line = re.sub(r"^[\d]+[\.\)]\s*", "", line)

            if line and len(line) > 5:  # Minimum question length
                sub_questions.append(SubQuestion(question=line, order=i))

        return sub_questions if sub_questions else self._rule_based_decompose(query, max_questions)

    def _rule_based_decompose(self, query: str, max_questions: int) -> List[SubQuestion]:
        """
        Use rule-based heuristics to decompose the question.

        Args:
            query: The question to decompose
            max_questions: Maximum sub-questions

        Returns:
            List of SubQuestion objects
        """
        sub_questions = []
        query_lower = query.lower()

        # Pattern 1: Questions with "and" or "also"
        if re.search(r"\b(and|also|as well as)\b", query_lower):
            parts = re.split(r"\b(?:and|also|as well as)\b", query, flags=re.IGNORECASE)
            for i, part in enumerate(parts[:max_questions]):
                part = part.strip()
                if part and len(part) > 5:
                    # Ensure it's a proper question
                    if not part.endswith("?"):
                        part = part.rstrip(".") + "?"
                    sub_questions.append(SubQuestion(question=part, order=i))

        # Pattern 2: Compare/contrast questions
        elif re.search(r"\b(compare|contrast|difference|similar)\b", query_lower):
            # Extract entities being compared
            match = re.search(r"between\s+(.+?)\s+and\s+(.+?)[\?\.]", query, re.IGNORECASE)
            if match:
                entity1, entity2 = match.groups()
                sub_questions.append(SubQuestion(question=f"What is {entity1.strip()}?", order=0))
                sub_questions.append(SubQuestion(question=f"What is {entity2.strip()}?", order=1))
                sub_questions.append(
                    SubQuestion(
                        question=f"What are the key characteristics of {entity1.strip()}?",
                        order=2,
                        depends_on=[0],
                    )
                )
                sub_questions.append(
                    SubQuestion(
                        question=f"What are the key characteristics of {entity2.strip()}?",
                        order=3,
                        depends_on=[1],
                    )
                )

        # Pattern 3: Pros and cons
        elif re.search(r"\b(pros and cons|advantages and disadvantages)\b", query_lower):
            # Extract the topic
            match = re.search(
                r"(?:pros and cons|advantages and disadvantages)\s+of\s+(.+?)[\?\.]",
                query,
                re.IGNORECASE,
            )
            topic = match.group(1).strip() if match else query
            sub_questions.append(
                SubQuestion(question=f"What are the advantages of {topic}?", order=0)
            )
            sub_questions.append(
                SubQuestion(question=f"What are the disadvantages of {topic}?", order=1)
            )

        # Pattern 4: Multiple question words
        elif re.search(
            r"\b(how|why|what|when|where|who)\b.*\b(how|why|what|when|where|who)\b", query_lower
        ):
            # Split by question words
            parts = re.split(r"[,;]|\band\b", query, flags=re.IGNORECASE)
            for i, part in enumerate(parts[:max_questions]):
                part = part.strip()
                if part and len(part) > 5:
                    if not part.endswith("?"):
                        part = part.rstrip(".") + "?"
                    sub_questions.append(SubQuestion(question=part, order=i))

        # Pattern 5: Relationship questions
        elif re.search(r"\b(relationship|connection|between)\b", query_lower):
            match = re.search(r"between\s+(.+?)\s+and\s+(.+?)[\?\.]", query, re.IGNORECASE)
            if match:
                entity1, entity2 = match.groups()
                sub_questions.append(SubQuestion(question=f"What is {entity1.strip()}?", order=0))
                sub_questions.append(SubQuestion(question=f"What is {entity2.strip()}?", order=1))
                sub_questions.append(
                    SubQuestion(
                        question=f"How does {entity1.strip()} relate to {entity2.strip()}?",
                        order=2,
                        depends_on=[0, 1],
                    )
                )

        # If no patterns matched, create a single sub-question
        if not sub_questions:
            sub_questions.append(SubQuestion(question=query, order=0))

        return sub_questions[:max_questions]


class AnswerSynthesizer:
    """
    Synthesizes multiple sub-answers into a comprehensive final answer.
    """

    def __init__(self, model_manager: Optional[Any] = None):
        """
        Initialize the answer synthesizer.

        Args:
            model_manager: Optional model manager for ML-based synthesis
        """
        self.model_manager = model_manager

    def synthesize(
        self, original_query: str, sub_questions: List[SubQuestion], sub_answers: List[str]
    ) -> str:
        """
        Synthesize sub-answers into a final comprehensive answer.

        Args:
            original_query: The original complex question
            sub_questions: List of sub-questions
            sub_answers: List of answers to sub-questions

        Returns:
            Synthesized final answer
        """
        # Try model-based synthesis first
        if self.model_manager:
            try:
                return self._model_synthesize(original_query, sub_questions, sub_answers)
            except Exception as e:
                logger.warning(f"Model synthesis failed: {e}, using rule-based")

        # Fall back to rule-based synthesis
        return self._rule_based_synthesize(original_query, sub_questions, sub_answers)

    def _model_synthesize(
        self, original_query: str, sub_questions: List[SubQuestion], sub_answers: List[str]
    ) -> str:
        """
        Use a language model to synthesize the answer.
        """
        # Build context from sub-questions and answers
        qa_pairs = []
        for sq, answer in zip(sub_questions, sub_answers):
            if sq.success and answer:
                qa_pairs.append(f"Q: {sq.question}\nA: {answer}")

        context = "\n\n".join(qa_pairs)

        prompt = f"""Based on the following information, provide a comprehensive answer to the original question.

Information gathered:
{context}

Original question: {original_query}

Comprehensive answer:"""

        response = self.model_manager.infer("gpt2", prompt, max_new_tokens=400, temperature=0.7)

        # Clean up response
        if "Comprehensive answer:" in response:
            response = response.split("Comprehensive answer:")[-1].strip()

        return response

    def _rule_based_synthesize(
        self, original_query: str, sub_questions: List[SubQuestion], sub_answers: List[str]
    ) -> str:
        """
        Use rule-based approach to synthesize the answer.
        """
        if not sub_answers:
            return "I couldn't find enough information to answer your question."

        # Filter successful answers
        successful_answers = []
        for sq, answer in zip(sub_questions, sub_answers):
            if sq.success and answer and answer.strip():
                successful_answers.append({"question": sq.question, "answer": answer.strip()})

        if not successful_answers:
            return "I couldn't find enough information to answer your question."

        # Build synthesized answer
        if len(successful_answers) == 1:
            return successful_answers[0]["answer"]

        # Multiple answers - combine them
        parts = []

        # Add introduction
        parts.append(
            f"To answer your question about '{original_query[:100]}...', here's what I found:\n"
        )

        # Add each sub-answer
        for i, item in enumerate(successful_answers, 1):
            # Truncate long answers
            answer = item["answer"]
            if len(answer) > 500:
                answer = answer[:500] + "..."

            parts.append(f"\n{i}. Regarding '{item['question'][:50]}...':\n{answer}")

        # Add conclusion
        parts.append("\n\nIn summary, " + self._generate_summary(successful_answers))

        return "".join(parts)

    def _generate_summary(self, answers: List[Dict[str, str]]) -> str:
        """Generate a brief summary of the answers."""
        if len(answers) == 1:
            return (
                answers[0]["answer"][:200] + "..."
                if len(answers[0]["answer"]) > 200
                else answers[0]["answer"]
            )

        # Extract key points from each answer
        key_points = []
        for item in answers[:3]:  # Limit to first 3
            answer = item["answer"]
            # Get first sentence
            first_sentence = answer.split(".")[0] if "." in answer else answer[:100]
            key_points.append(first_sentence.strip())

        return "; ".join(key_points) + "."


class SelfAskSearchQAWorkflow(BaseWorkflow):
    """
    SelfAskSearchQA Workflow: Decompose → Search → Synthesize

    This workflow handles complex questions by:
    1. Decomposing the question into simpler sub-questions
    2. Searching for answers to each sub-question using SearchQA
    3. Synthesizing all sub-answers into a comprehensive final answer

    This is particularly useful for:
    - Questions requiring multiple pieces of information
    - Comparison questions
    - Questions about relationships between concepts
    - Multi-part questions

    Attributes:
        workflow_type: WorkflowType.SELF_ASK_SEARCH_QA
        name: "SelfAskSearchQA"

    Requirements verified:
        - Requirement 1.5: Support self_ask_search_qa workflow
    """

    workflow_type = WorkflowType.SELF_ASK_SEARCH_QA
    name = "SelfAskSearchQA"
    description = "Self-ask search-based question answering for complex questions"

    def __init__(
        self,
        search_qa_workflow: Optional[SearchQAWorkflow] = None,
        decomposer: Optional[QuestionDecomposer] = None,
        synthesizer: Optional[AnswerSynthesizer] = None,
        model_manager: Optional[Any] = None,
        max_sub_questions: int = 5,
        max_search_results_per_question: int = 3,
    ):
        """
        Initialize the SelfAskSearchQA workflow.

        Args:
            search_qa_workflow: SearchQA workflow instance for sub-question searches
            decomposer: Question decomposer instance
            synthesizer: Answer synthesizer instance
            model_manager: Model manager for ML operations
            max_sub_questions: Maximum number of sub-questions to generate
            max_search_results_per_question: Max search results per sub-question
        """
        super().__init__()
        self.model_manager = model_manager
        self.max_sub_questions = max_sub_questions
        self.max_search_results_per_question = max_search_results_per_question

        # Initialize components
        self.search_qa_workflow = search_qa_workflow or SearchQAWorkflow(
            model_manager=model_manager, max_search_results=max_search_results_per_question
        )
        self.decomposer = decomposer or QuestionDecomposer(model_manager=model_manager)
        self.synthesizer = synthesizer or AnswerSynthesizer(model_manager=model_manager)

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for this workflow."""
        return ["query"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        """Return optional parameters with defaults."""
        return {
            "max_sub_questions": self.max_sub_questions,
            "max_search_results": self.max_search_results_per_question,
            "include_sub_answers": True,
            "force_decompose": False,
        }

    def get_required_models(self) -> List[str]:
        """Return the list of models required by this workflow."""
        return self.search_qa_workflow.get_required_models()

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

        query = parameters.get("query", "")
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        max_sub = parameters.get("max_sub_questions", self.max_sub_questions)
        if not isinstance(max_sub, int) or max_sub < 1:
            raise ValidationError("max_sub_questions must be a positive integer")

        return True

    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the SelfAskSearchQA workflow.

        Steps:
        1. Decompose: Break the complex question into sub-questions
        2. Search: Execute SearchQA for each sub-question
        3. Synthesize: Combine sub-answers into final answer

        Args:
            parameters: Workflow parameters including 'query'

        Returns:
            WorkflowResult with synthesized answer and metadata
        """
        query = parameters["query"]
        max_sub_questions = parameters.get("max_sub_questions", self.max_sub_questions)
        include_sub_answers = parameters.get("include_sub_answers", True)
        force_decompose = parameters.get("force_decompose", False)

        # Initialize context
        ctx = SelfAskContext(original_query=query, max_sub_questions=max_sub_questions)

        try:
            # Step 1: Check if decomposition is needed
            should_decompose = force_decompose or self.decomposer.is_complex_question(query)

            if not should_decompose:
                # Simple question - just use SearchQA directly
                logger.info("Question is simple, using direct SearchQA")
                return self._execute_simple_search(query, parameters)

            # Step 2: Decompose the question
            ctx = self._step_decompose(ctx)

            if not ctx.sub_questions:
                # Decomposition failed - fall back to simple search
                logger.warning("Decomposition produced no sub-questions, using direct SearchQA")
                return self._execute_simple_search(query, parameters)

            # Step 3: Search for each sub-question
            ctx = self._step_search_sub_questions(ctx, parameters)

            # Step 4: Synthesize the final answer
            ctx = self._step_synthesize(ctx)

            # Determine status
            successful_answers = sum(1 for sq in ctx.sub_questions if sq.success)
            if successful_answers == 0:
                status = "failed"
            elif successful_answers < len(ctx.sub_questions):
                status = "partial"
            else:
                status = "success"

            return self._create_result(ctx, status=status, include_sub_answers=include_sub_answers)

        except Exception as e:
            logger.error("SelfAskSearchQA workflow failed", error=str(e), query=query[:50])
            return self._create_result(
                ctx, status="failed", error=str(e), include_sub_answers=include_sub_answers
            )

    def _execute_simple_search(self, query: str, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute a simple SearchQA for non-complex questions.

        Args:
            query: The question
            parameters: Original parameters

        Returns:
            WorkflowResult from SearchQA
        """
        search_params = {
            "query": query,
            "max_results": parameters.get(
                "max_search_results", self.max_search_results_per_question
            ),
        }

        result = self.search_qa_workflow.execute(search_params)

        # Add metadata indicating this was a simple search
        result.metadata["workflow"] = self.name
        result.metadata["decomposed"] = False
        result.metadata["sub_questions_count"] = 0

        return result

    def _step_decompose(self, ctx: SelfAskContext) -> SelfAskContext:
        """
        Step 1: Decompose the question into sub-questions.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with sub-questions
        """
        start_time = time.time()
        step = SelfAskStep(name="decompose", success=False, input_count=1)

        try:
            logger.info(f"Step 1: Decomposing question '{ctx.original_query[:50]}...'")

            sub_questions = self.decomposer.decompose(
                ctx.original_query, max_questions=ctx.max_sub_questions
            )

            ctx.sub_questions = sub_questions
            step.success = len(sub_questions) > 0
            step.output_count = len(sub_questions)

            logger.info(f"Decomposed into {len(sub_questions)} sub-questions")

            for i, sq in enumerate(sub_questions):
                logger.debug(f"  Sub-question {i+1}: {sq.question[:50]}...")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Decomposition failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_search_sub_questions(
        self, ctx: SelfAskContext, parameters: Dict[str, Any]
    ) -> SelfAskContext:
        """
        Step 2: Search for answers to each sub-question.

        Args:
            ctx: Workflow context with sub-questions
            parameters: Original workflow parameters

        Returns:
            Updated context with sub-answers
        """
        start_time = time.time()
        step = SelfAskStep(name="search", success=False, input_count=len(ctx.sub_questions))

        try:
            logger.info(f"Step 2: Searching for {len(ctx.sub_questions)} sub-questions")

            max_results = parameters.get("max_search_results", self.max_search_results_per_question)

            for i, sub_q in enumerate(ctx.sub_questions):
                logger.info(
                    f"  Searching sub-question {i+1}/{len(ctx.sub_questions)}: {sub_q.question[:40]}..."
                )

                try:
                    # Execute SearchQA for this sub-question
                    search_result = self.search_qa_workflow.execute(
                        {"query": sub_q.question, "max_results": max_results}
                    )

                    if search_result.status in ["success", "partial"] and search_result.result:
                        sub_q.answer = search_result.result
                        sub_q.success = True
                        ctx.sub_answers.append(search_result.result)
                        logger.info(f"    Got answer of length {len(search_result.result)}")
                    else:
                        sub_q.success = False
                        sub_q.error = search_result.error or "No result"
                        ctx.sub_answers.append("")
                        logger.warning(f"    No answer found: {sub_q.error}")

                except Exception as e:
                    sub_q.success = False
                    sub_q.error = str(e)
                    ctx.sub_answers.append("")
                    logger.error(f"    Search failed: {e}")

            successful = sum(1 for sq in ctx.sub_questions if sq.success)
            step.output_count = successful
            step.success = successful > 0

            if successful < len(ctx.sub_questions):
                step.error = f"Only {successful}/{len(ctx.sub_questions)} sub-questions answered"

            logger.info(f"Answered {successful}/{len(ctx.sub_questions)} sub-questions")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Search step failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_synthesize(self, ctx: SelfAskContext) -> SelfAskContext:
        """
        Step 3: Synthesize sub-answers into final answer.

        Args:
            ctx: Workflow context with sub-answers

        Returns:
            Updated context with final answer
        """
        start_time = time.time()
        successful_answers = [sq for sq in ctx.sub_questions if sq.success]
        step = SelfAskStep(name="synthesize", success=False, input_count=len(successful_answers))

        try:
            if not successful_answers:
                step.error = "No successful sub-answers to synthesize"
                ctx.final_answer = "I couldn't find enough information to answer your question."
                ctx.add_step(step)
                return ctx

            logger.info(f"Step 3: Synthesizing {len(successful_answers)} sub-answers")

            final_answer = self.synthesizer.synthesize(
                ctx.original_query, ctx.sub_questions, ctx.sub_answers
            )

            ctx.final_answer = final_answer
            step.success = bool(final_answer)
            step.output_count = 1 if final_answer else 0

            logger.info(f"Generated final answer of length {len(final_answer)}")

        except Exception as e:
            step.error = str(e)
            # Fallback: concatenate successful answers
            ctx.final_answer = self._fallback_synthesis(ctx)
            logger.warning(f"Synthesis failed, using fallback: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _fallback_synthesis(self, ctx: SelfAskContext) -> str:
        """
        Fallback synthesis when model-based synthesis fails.

        Args:
            ctx: Workflow context

        Returns:
            Simple concatenated answer
        """
        successful = [
            (sq, ans) for sq, ans in zip(ctx.sub_questions, ctx.sub_answers) if sq.success and ans
        ]

        if not successful:
            return "I couldn't find enough information to answer your question."

        parts = [f"Based on my research:\n"]
        for sq, ans in successful:
            parts.append(f"\n• {ans[:300]}..." if len(ans) > 300 else f"\n• {ans}")

        return "".join(parts)

    def _create_result(
        self,
        ctx: SelfAskContext,
        status: str = "success",
        error: Optional[str] = None,
        include_sub_answers: bool = True,
    ) -> WorkflowResult:
        """
        Create the workflow result.

        Args:
            ctx: Workflow context
            status: Result status
            error: Error message if any
            include_sub_answers: Whether to include sub-question details

        Returns:
            WorkflowResult object
        """
        metadata = {
            "workflow": self.name,
            "original_query": ctx.original_query,
            "decomposed": len(ctx.sub_questions) > 0,
            "sub_questions_count": len(ctx.sub_questions),
            "successful_answers_count": sum(1 for sq in ctx.sub_questions if sq.success),
            "steps": [
                {
                    "name": s.name,
                    "success": s.success,
                    "duration": s.duration,
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "error": s.error,
                }
                for s in ctx.steps
            ],
        }

        if include_sub_answers:
            metadata["sub_questions"] = [
                {
                    "question": sq.question,
                    "order": sq.order,
                    "success": sq.success,
                    "answer": sq.answer[:500] if sq.answer else None,
                    "error": sq.error,
                }
                for sq in ctx.sub_questions
            ]

        return WorkflowResult(
            result=ctx.final_answer if ctx.final_answer else None,
            metadata=metadata,
            status=status,
            error=error,
        )
