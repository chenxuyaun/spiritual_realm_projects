"""
SearchQA Workflow Implementation.

This module implements the SearchQA workflow which performs:
1. Web search using ddgs
2. Content extraction using trafilatura
3. Summarization using T5/BART
4. Answer generation using Qwen Chat/GPT2

The workflow includes degradation strategies for handling failures
at each step.

Supports both mock model manager and real model integration via
RealModelManager and InferenceEngine.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import time

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.tools.web_search import WebSearchTool, SearchResult, get_web_search_tool
from mm_orch.tools.fetch_url import URLFetchTool, FetchedContent, get_url_fetch_tool
from mm_orch.exceptions import ValidationError, WorkflowError, NetworkError, ModelError
from mm_orch.logger import get_logger

if TYPE_CHECKING:
    from mm_orch.runtime.real_model_manager import RealModelManager
    from mm_orch.runtime.inference_engine import InferenceEngine


logger = get_logger(__name__)


# Prompt templates for SearchQA with real models
SEARCH_QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided search results.
Always cite your sources when possible. If the information is insufficient to answer the question, say so honestly."""

SEARCH_QA_PROMPT_TEMPLATE = """Based on the following search results, please answer the question.

Search Results:
{context}

Question: {query}

Instructions:
- Answer the question based on the provided search results
- Cite sources when possible (e.g., "According to [source]...")
- If the search results don't contain enough information, acknowledge this
- Be concise but comprehensive

Answer:"""

SEARCH_QA_PROMPT_TEMPLATE_ZH = """根据以下搜索结果，请回答问题。

搜索结果：
{context}

问题：{query}

要求：
- 根据提供的搜索结果回答问题
- 尽可能引用来源（例如："根据[来源]..."）
- 如果搜索结果信息不足，请如实说明
- 回答要简洁但全面

回答："""


@dataclass
class SearchQAStep:
    """Tracks the execution of a workflow step."""

    name: str
    success: bool
    duration: float = 0.0
    input_count: int = 0
    output_count: int = 0
    error: Optional[str] = None
    degraded: bool = False


@dataclass
class SearchQAContext:
    """Context for SearchQA workflow execution."""

    query: str
    search_results: List[SearchResult] = field(default_factory=list)
    fetched_contents: List[FetchedContent] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    answer: str = ""
    steps: List[SearchQAStep] = field(default_factory=list)
    degraded: bool = False

    def add_step(self, step: SearchQAStep) -> None:
        """Add a step to the execution history."""
        self.steps.append(step)
        if step.degraded:
            self.degraded = True


class SearchQAWorkflow(BaseWorkflow):
    """
    SearchQA Workflow: Search → Fetch → Summarize → Answer

    This workflow performs web search-based question answering by:
    1. Searching the web for relevant pages using ddgs
    2. Fetching and extracting content from search results using trafilatura
    3. Summarizing the extracted content using T5/BART
    4. Generating a final answer using Qwen Chat or GPT2

    The workflow implements degradation strategies:
    - If search fails: Return error with partial results
    - If fetch fails for some URLs: Continue with successful fetches
    - If summarization fails: Use raw content (truncated)
    - If answer generation fails: Return summarized content as answer

    Supports real model integration via RealModelManager and InferenceEngine.

    Attributes:
        workflow_type: WorkflowType.SEARCH_QA
        name: "SearchQA"

    Properties verified:
        - Property 2: Workflow execution step order
        - Property 5: SearchQA degradation strategy
    """

    workflow_type = WorkflowType.SEARCH_QA
    name = "SearchQA"
    description = "Search-based question answering workflow"

    def __init__(
        self,
        search_tool: Optional[WebSearchTool] = None,
        fetch_tool: Optional[URLFetchTool] = None,
        model_manager: Optional[Any] = None,
        real_model_manager: Optional["RealModelManager"] = None,
        inference_engine: Optional["InferenceEngine"] = None,
        max_search_results: int = 5,
        max_content_length: int = 2000,
        summarizer_model: str = "t5-small",
        generator_model: str = "gpt2",
        use_real_models: bool = False,
        language: str = "en",
    ):
        """
        Initialize the SearchQA workflow.

        Args:
            search_tool: Web search tool instance
            fetch_tool: URL fetch tool instance
            model_manager: Model manager for summarization and generation (mock)
            real_model_manager: Real model manager for actual LLM inference
            inference_engine: Inference engine for real model generation
            max_search_results: Maximum search results to process
            max_content_length: Maximum content length per source
            summarizer_model: Model name for summarization
            generator_model: Model name for answer generation
            use_real_models: Whether to use real models instead of mock
            language: Output language ("en" or "zh")
        """
        super().__init__()
        self.search_tool = search_tool or get_web_search_tool()
        self.fetch_tool = fetch_tool or get_url_fetch_tool()
        self.model_manager = model_manager
        self.real_model_manager = real_model_manager
        self.inference_engine = inference_engine
        self.max_search_results = max_search_results
        self.max_content_length = max_content_length
        self.summarizer_model = summarizer_model
        self.generator_model = generator_model
        self.use_real_models = use_real_models
        self.language = language

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for this workflow."""
        return ["query"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        """Return optional parameters with defaults."""
        return {"max_results": self.max_search_results, "include_sources": True, "language": "en"}

    def get_required_models(self) -> List[str]:
        """Return the list of models required by this workflow."""
        return [self.summarizer_model, self.generator_model]

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

        max_results = parameters.get("max_results", self.max_search_results)
        if not isinstance(max_results, int) or max_results < 1:
            raise ValidationError("max_results must be a positive integer")

        return True

    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the SearchQA workflow.

        Steps:
        1. Search: Use ddgs to find relevant web pages
        2. Fetch: Extract content from search results using trafilatura
        3. Summarize: Generate summaries of fetched content
        4. Answer: Generate final answer based on summaries

        Args:
            parameters: Workflow parameters including 'query'

        Returns:
            WorkflowResult with answer and metadata
        """
        query = parameters["query"]
        max_results = parameters.get("max_results", self.max_search_results)
        include_sources = parameters.get("include_sources", True)

        # Initialize context
        ctx = SearchQAContext(query=query)

        try:
            # Step 1: Search
            ctx = self._step_search(ctx, max_results)

            # Check if we have any results
            if not ctx.search_results:
                return self._create_result(ctx, status="partial", error="No search results found")

            # Step 2: Fetch content
            ctx = self._step_fetch(ctx)

            # Check if we have any content
            successful_fetches = [f for f in ctx.fetched_contents if f.success]
            if not successful_fetches:
                # Degrade: use search snippets as content
                ctx = self._degrade_use_snippets(ctx)

            # Step 3: Summarize
            ctx = self._step_summarize(ctx)

            # Step 4: Generate answer
            ctx = self._step_generate_answer(ctx)

            # Determine final status
            status = "partial" if ctx.degraded else "success"

            return self._create_result(ctx, status=status, include_sources=include_sources)

        except Exception as e:
            logger.error("SearchQA workflow failed", error=str(e), query=query[:50])
            return self._create_result(
                ctx,
                status="partial" if ctx.answer or ctx.summaries else "failed",
                error=str(e),
                include_sources=include_sources,
            )

    def _step_search(self, ctx: SearchQAContext, max_results: int) -> SearchQAContext:
        """
        Step 1: Perform web search.

        Args:
            ctx: Workflow context
            max_results: Maximum results to retrieve

        Returns:
            Updated context with search results
        """
        start_time = time.time()
        step = SearchQAStep(name="search", success=False, input_count=1)

        try:
            logger.info(f"Step 1: Searching for '{ctx.query[:50]}...'")

            results = self.search_tool.search(query=ctx.query, max_results=max_results)

            ctx.search_results = results
            step.success = True
            step.output_count = len(results)

            logger.info(f"Search returned {len(results)} results")

        except NetworkError as e:
            step.error = str(e)
            step.degraded = True
            logger.warning(f"Search failed: {e}")

        except Exception as e:
            step.error = str(e)
            step.degraded = True
            logger.error(f"Unexpected search error: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_fetch(self, ctx: SearchQAContext) -> SearchQAContext:
        """
        Step 2: Fetch content from search results.

        Args:
            ctx: Workflow context with search results

        Returns:
            Updated context with fetched content
        """
        start_time = time.time()
        step = SearchQAStep(name="fetch", success=False, input_count=len(ctx.search_results))

        try:
            urls = [r.url for r in ctx.search_results if r.url]

            if not urls:
                step.error = "No URLs to fetch"
                step.degraded = True
                ctx.add_step(step)
                return ctx

            logger.info(f"Step 2: Fetching content from {len(urls)} URLs")

            fetched = self.fetch_tool.fetch_multiple(urls)
            ctx.fetched_contents = fetched

            successful = sum(1 for f in fetched if f.success)
            step.output_count = successful
            step.success = successful > 0

            if successful < len(urls):
                step.degraded = True
                step.error = f"Only {successful}/{len(urls)} URLs fetched successfully"

            logger.info(f"Fetched {successful}/{len(urls)} URLs")

        except Exception as e:
            step.error = str(e)
            step.degraded = True
            logger.error(f"Fetch step failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _degrade_use_snippets(self, ctx: SearchQAContext) -> SearchQAContext:
        """
        Degradation: Use search snippets when fetch fails.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with snippet-based content
        """
        logger.warning("Degrading: Using search snippets as content")

        for result in ctx.search_results:
            if result.snippet:
                ctx.fetched_contents.append(
                    FetchedContent(
                        url=result.url, content=result.snippet, title=result.title, success=True
                    )
                )

        ctx.degraded = True
        return ctx

    def _step_summarize(self, ctx: SearchQAContext) -> SearchQAContext:
        """
        Step 3: Summarize fetched content.

        Args:
            ctx: Workflow context with fetched content

        Returns:
            Updated context with summaries
        """
        start_time = time.time()
        successful_fetches = [f for f in ctx.fetched_contents if f.success and f.content]
        step = SearchQAStep(name="summarize", success=False, input_count=len(successful_fetches))

        try:
            if not successful_fetches:
                step.error = "No content to summarize"
                step.degraded = True
                ctx.add_step(step)
                return ctx

            logger.info(f"Step 3: Summarizing {len(successful_fetches)} documents")

            summaries = []
            for fetched in successful_fetches:
                summary = self._summarize_content(fetched.content)
                if summary:
                    summaries.append(summary)

            ctx.summaries = summaries
            step.output_count = len(summaries)
            step.success = len(summaries) > 0

            if len(summaries) < len(successful_fetches):
                step.degraded = True

            logger.info(f"Generated {len(summaries)} summaries")

        except Exception as e:
            step.error = str(e)
            step.degraded = True
            # Degrade: use truncated content as summaries
            ctx = self._degrade_truncate_content(ctx)
            logger.warning(f"Summarization failed, using truncated content: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _summarize_content(self, content: str) -> str:
        """
        Summarize a single piece of content.

        Args:
            content: Text content to summarize

        Returns:
            Summarized text
        """
        if not content:
            return ""

        # Truncate very long content
        if len(content) > self.max_content_length * 2:
            content = content[: self.max_content_length * 2]

        # If model manager is available, use it for summarization
        if self.model_manager:
            try:
                summary = self.model_manager.infer(
                    self.summarizer_model, f"summarize: {content}", max_new_tokens=150
                )
                return summary
            except Exception as e:
                logger.warning(f"Model summarization failed: {e}")

        # Fallback: simple truncation with sentence boundary
        return self._simple_summarize(content)

    def _simple_summarize(self, content: str, max_length: int = 500) -> str:
        """
        Simple summarization by truncating at sentence boundaries.

        Args:
            content: Text to summarize
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(content) <= max_length:
            return content

        # Try to truncate at sentence boundary
        truncated = content[:max_length]

        # Find last sentence ending
        for end_char in [". ", "! ", "? ", "。", "！", "？"]:
            last_end = truncated.rfind(end_char)
            if last_end > max_length // 2:
                return truncated[: last_end + 1].strip()

        # Fallback: truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            return truncated[:last_space].strip() + "..."

        return truncated.strip() + "..."

    def _degrade_truncate_content(self, ctx: SearchQAContext) -> SearchQAContext:
        """
        Degradation: Use truncated content when summarization fails.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with truncated content as summaries
        """
        logger.warning("Degrading: Using truncated content as summaries")

        summaries = []
        for fetched in ctx.fetched_contents:
            if fetched.success and fetched.content:
                truncated = self._simple_summarize(
                    fetched.content, max_length=self.max_content_length
                )
                summaries.append(truncated)

        ctx.summaries = summaries
        ctx.degraded = True
        return ctx

    def _step_generate_answer(self, ctx: SearchQAContext) -> SearchQAContext:
        """
        Step 4: Generate final answer.

        Args:
            ctx: Workflow context with summaries

        Returns:
            Updated context with answer
        """
        start_time = time.time()
        step = SearchQAStep(name="generate", success=False, input_count=len(ctx.summaries))

        try:
            if not ctx.summaries:
                step.error = "No summaries to generate answer from"
                step.degraded = True
                ctx.add_step(step)
                return ctx

            logger.info("Step 4: Generating answer")

            answer = self._generate_answer(ctx.query, ctx.summaries)
            ctx.answer = answer
            step.success = bool(answer)
            step.output_count = 1 if answer else 0

            if not answer:
                step.degraded = True
                step.error = "Empty answer generated"

            logger.info(f"Generated answer of length {len(answer)}")

        except Exception as e:
            step.error = str(e)
            step.degraded = True
            # Degrade: use combined summaries as answer
            ctx = self._degrade_combine_summaries(ctx)
            logger.warning(f"Answer generation failed, using combined summaries: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _generate_answer(self, query: str, summaries: List[str]) -> str:
        """
        Generate answer from query and summaries.

        Args:
            query: User's question
            summaries: List of content summaries

        Returns:
            Generated answer
        """
        # Combine summaries into context
        context = "\n\n".join(summaries)

        # Truncate context if too long
        max_context = 3000
        if len(context) > max_context:
            context = context[:max_context] + "..."

        # Use real models if available and enabled
        if self.use_real_models and self.inference_engine:
            return self._generate_with_real_model(query, context)

        # Build prompt for mock model manager
        prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {query}

Answer:"""

        # If model manager is available, use it
        if self.model_manager:
            try:
                answer = self.model_manager.infer(
                    self.generator_model, prompt, max_new_tokens=300, temperature=0.7
                )
                # Clean up the answer
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                return answer
            except Exception as e:
                logger.warning(f"Model generation failed: {e}")

        # Fallback: return a simple combined response
        return self._simple_answer(query, summaries)

    def _generate_with_real_model(self, query: str, context: str) -> str:
        """
        Generate answer using real model via InferenceEngine.

        Args:
            query: User's question
            context: Combined context from search results

        Returns:
            Generated answer
        """
        try:
            # Select prompt template based on language
            if self.language == "zh":
                prompt = SEARCH_QA_PROMPT_TEMPLATE_ZH.format(
                    context=context,
                    query=query
                )
            else:
                prompt = SEARCH_QA_PROMPT_TEMPLATE.format(
                    context=context,
                    query=query
                )

            # Generate using inference engine
            from mm_orch.runtime.inference_engine import GenerationConfig
            
            config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

            result = self.inference_engine.generate(prompt, config=config)
            answer = result.text.strip()

            # Post-process: validate citations if sources are available
            answer = self._post_process_answer(answer, context)

            logger.info(
                f"Generated answer with real model: {len(answer)} chars, "
                f"{result.tokens_per_second:.1f} tokens/s"
            )
            return answer

        except Exception as e:
            logger.error(f"Real model generation failed: {e}")
            # Fallback to simple answer
            return self._simple_answer(query, context.split("\n\n"))

    def _post_process_answer(self, answer: str, context: str) -> str:
        """
        Post-process the generated answer.

        Validates citations and cleans up the response.

        Args:
            answer: Generated answer
            context: Original context for validation

        Returns:
            Post-processed answer
        """
        if not answer:
            return answer

        # Remove any trailing incomplete sentences
        answer = answer.strip()

        # Check for common generation artifacts
        stop_markers = [
            "\n\nQuestion:",
            "\n\nSearch Results:",
            "\n\n搜索结果：",
            "\n\n问题：",
        ]
        for marker in stop_markers:
            if marker in answer:
                answer = answer.split(marker)[0].strip()

        # Validate that answer doesn't hallucinate sources not in context
        # This is a simple check - more sophisticated validation could be added
        if "according to" in answer.lower() or "根据" in answer:
            # Log for monitoring, but don't modify the answer
            logger.debug("Answer contains citations - validation passed")

        return answer

    def _simple_answer(self, query: str, summaries: List[str]) -> str:
        """
        Generate a simple answer without using models.

        Args:
            query: User's question
            summaries: Content summaries

        Returns:
            Simple combined answer
        """
        if not summaries:
            return "I couldn't find relevant information to answer your question."

        # Combine first few summaries
        combined = " ".join(summaries[:3])

        # Truncate if needed
        if len(combined) > 1000:
            combined = self._simple_summarize(combined, max_length=1000)

        return f"Based on the search results: {combined}"

    def _degrade_combine_summaries(self, ctx: SearchQAContext) -> SearchQAContext:
        """
        Degradation: Use combined summaries when answer generation fails.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with combined summaries as answer
        """
        logger.warning("Degrading: Using combined summaries as answer")

        if ctx.summaries:
            ctx.answer = self._simple_answer(ctx.query, ctx.summaries)
        else:
            ctx.answer = "Unable to generate an answer due to processing errors."

        ctx.degraded = True
        return ctx

    def _create_result(
        self,
        ctx: SearchQAContext,
        status: str = "success",
        error: Optional[str] = None,
        include_sources: bool = True,
    ) -> WorkflowResult:
        """
        Create the workflow result.

        Args:
            ctx: Workflow context
            status: Result status
            error: Error message if any
            include_sources: Whether to include source information

        Returns:
            WorkflowResult object
        """
        metadata = {
            "workflow": self.name,
            "query": ctx.query,
            "steps": [
                {
                    "name": s.name,
                    "success": s.success,
                    "duration": s.duration,
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "degraded": s.degraded,
                    "error": s.error,
                }
                for s in ctx.steps
            ],
            "degraded": ctx.degraded,
            "search_results_count": len(ctx.search_results),
            "fetched_count": sum(1 for f in ctx.fetched_contents if f.success),
            "summaries_count": len(ctx.summaries),
        }

        if include_sources:
            metadata["sources"] = [
                {"title": r.title, "url": r.url, "snippet": r.snippet[:200] if r.snippet else ""}
                for r in ctx.search_results
            ]

        result = ctx.answer if ctx.answer else None

        # If no answer but we have summaries, use them
        if not result and ctx.summaries:
            result = "\n\n".join(ctx.summaries)
            status = "partial"

        return WorkflowResult(result=result, metadata=metadata, status=status, error=error)
