"""
Refactored workflow steps using the new Step API.

This module provides Step implementations that wrap existing workflow
functionality to work with the new graph-based execution model.
"""

from typing import Any, Dict, List
import time

from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration.state import State
from mm_orch.tools.web_search import WebSearchTool, get_web_search_tool
from mm_orch.tools.fetch_url import URLFetchTool, get_url_fetch_tool
from mm_orch.logger import get_logger


logger = get_logger(__name__)


class WebSearchStep(BaseStep):
    """
    Web search step using ddgs.
    
    Input Keys:
        - question: User's query
        - meta: Optional metadata (for max_results)
    
    Output Keys:
        - search_results: List of search result dicts
    """
    
    name = "web_search"
    input_keys = ["question"]
    output_keys = ["search_results"]
    
    def __init__(self, search_tool: WebSearchTool = None, max_results: int = 5):
        """
        Initialize web search step.
        
        Args:
            search_tool: Web search tool instance
            max_results: Maximum search results to return
        """
        self.search_tool = search_tool or get_web_search_tool()
        self.max_results = max_results
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Execute web search.
        
        Args:
            state: Current state with question
            runtime: Runtime context
        
        Returns:
            Dictionary with search_results key
        """
        question = state["question"]
        max_results = state.get("meta", {}).get("max_results", self.max_results)
        
        logger.info(f"WebSearchStep: Searching for '{question[:50]}...'")
        
        try:
            results = self.search_tool.search(query=question, max_results=max_results)
            
            # Convert SearchResult objects to dicts for State
            search_results = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet
                }
                for r in results
            ]
            
            logger.info(f"WebSearchStep: Found {len(search_results)} results")
            
            return {"search_results": search_results}
        
        except Exception as e:
            logger.error(f"WebSearchStep failed: {e}")
            # Return empty results on failure
            return {"search_results": []}


class FetchUrlStep(BaseStep):
    """
    Fetch and extract content from URLs.
    
    Input Keys:
        - search_results: List of search result dicts with 'url' field
    
    Output Keys:
        - docs: Dict mapping URL to extracted content
    """
    
    name = "fetch_url"
    input_keys = ["search_results"]
    output_keys = ["docs"]
    
    def __init__(self, fetch_tool: URLFetchTool = None, max_content_length: int = 2000):
        """
        Initialize fetch URL step.
        
        Args:
            fetch_tool: URL fetch tool instance
            max_content_length: Maximum content length per URL
        """
        self.fetch_tool = fetch_tool or get_url_fetch_tool()
        self.max_content_length = max_content_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Fetch content from URLs in search results.
        
        Args:
            state: Current state with search_results
            runtime: Runtime context
        
        Returns:
            Dictionary with docs key (url -> content mapping)
        """
        search_results = state["search_results"]
        
        # Extract URLs
        urls = [r["url"] for r in search_results if "url" in r]
        
        if not urls:
            logger.warning("FetchUrlStep: No URLs to fetch")
            return {"docs": {}}
        
        logger.info(f"FetchUrlStep: Fetching {len(urls)} URLs")
        
        try:
            fetched = self.fetch_tool.fetch_multiple(urls)
            
            # Build docs dict
            docs = {}
            for fetch_result in fetched:
                if fetch_result.success and fetch_result.content:
                    # Truncate content if needed
                    content = fetch_result.content
                    if len(content) > self.max_content_length:
                        content = content[:self.max_content_length]
                    
                    docs[fetch_result.url] = content
            
            logger.info(f"FetchUrlStep: Successfully fetched {len(docs)}/{len(urls)} URLs")
            
            return {"docs": docs}
        
        except Exception as e:
            logger.error(f"FetchUrlStep failed: {e}")
            return {"docs": {}}


class SummarizeStep(BaseStep):
    """
    Summarize fetched documents.
    
    Input Keys:
        - docs: Dict mapping URL to content
    
    Output Keys:
        - summaries: Dict mapping URL to summary
    """
    
    name = "summarize"
    input_keys = ["docs"]
    output_keys = ["summaries"]
    
    def __init__(self, model_name: str = "t5-small", max_summary_length: int = 500):
        """
        Initialize summarize step.
        
        Args:
            model_name: Name of summarization model
            max_summary_length: Maximum summary length
        """
        self.model_name = model_name
        self.max_summary_length = max_summary_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Summarize documents.
        
        Args:
            state: Current state with docs
            runtime: Runtime context (should have model_manager)
        
        Returns:
            Dictionary with summaries key (url -> summary mapping)
        """
        docs = state["docs"]
        
        if not docs:
            logger.warning("SummarizeStep: No documents to summarize")
            return {"summaries": {}}
        
        logger.info(f"SummarizeStep: Summarizing {len(docs)} documents")
        
        summaries = {}
        
        for url, content in docs.items():
            try:
                # Try to use model manager if available
                if hasattr(runtime, "model_manager") and runtime.model_manager:
                    summary = runtime.model_manager.infer(
                        self.model_name,
                        f"summarize: {content}",
                        max_new_tokens=150
                    )
                else:
                    # Fallback: simple truncation
                    summary = self._simple_summarize(content)
                
                summaries[url] = summary
            
            except Exception as e:
                logger.warning(f"SummarizeStep: Failed to summarize {url}: {e}")
                # Use truncated content as fallback
                summaries[url] = self._simple_summarize(content)
        
        logger.info(f"SummarizeStep: Generated {len(summaries)} summaries")
        
        return {"summaries": summaries}
    
    def _simple_summarize(self, content: str) -> str:
        """
        Simple summarization by truncating at sentence boundaries.
        
        Args:
            content: Text to summarize
        
        Returns:
            Truncated text
        """
        if len(content) <= self.max_summary_length:
            return content
        
        # Try to truncate at sentence boundary
        truncated = content[:self.max_summary_length]
        
        # Find last sentence ending
        for end_char in [". ", "! ", "? ", "。", "！", "？"]:
            last_end = truncated.rfind(end_char)
            if last_end > self.max_summary_length // 2:
                return truncated[:last_end + 1].strip()
        
        # Fallback: truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > self.max_summary_length // 2:
            return truncated[:last_space].strip() + "..."
        
        return truncated.strip() + "..."


class FetchSingleUrlStep(BaseStep):
    """
    Fetch and extract content from a single URL.
    
    Input Keys:
        - question: URL to fetch (passed as question field)
    
    Output Keys:
        - docs: Dict mapping URL to extracted content
        - citations: List containing the source URL
    """
    
    name = "fetch_single_url"
    input_keys = ["question"]
    output_keys = ["docs", "citations"]
    
    def __init__(self, fetch_tool: URLFetchTool = None, max_content_length: int = 5000):
        """
        Initialize fetch single URL step.
        
        Args:
            fetch_tool: URL fetch tool instance
            max_content_length: Maximum content length
        """
        self.fetch_tool = fetch_tool or get_url_fetch_tool()
        self.max_content_length = max_content_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Fetch content from a single URL.
        
        Args:
            state: Current state with question (URL)
            runtime: Runtime context
        
        Returns:
            Dictionary with docs and citations keys
        """
        url = state["question"]
        
        logger.info(f"FetchSingleUrlStep: Fetching URL '{url}'")
        
        try:
            fetch_result = self.fetch_tool.fetch(url)
            
            if fetch_result.success and fetch_result.content:
                # Truncate content if needed
                content = fetch_result.content
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length]
                
                docs = {url: content}
                citations = [url]
                
                logger.info(f"FetchSingleUrlStep: Successfully fetched {len(content)} characters")
                
                return {
                    "docs": docs,
                    "citations": citations
                }
            else:
                logger.warning(f"FetchSingleUrlStep: Failed to fetch URL: {fetch_result.error}")
                return {
                    "docs": {},
                    "citations": []
                }
        
        except Exception as e:
            logger.error(f"FetchSingleUrlStep failed: {e}")
            return {
                "docs": {},
                "citations": []
            }


class FetchTopNStep(BaseStep):
    """
    Fetch and extract content from top N URLs in search results.
    
    Input Keys:
        - search_results: List of search result dicts with 'url' field
    
    Output Keys:
        - docs: Dict mapping URL to extracted content
    """
    
    name = "fetch_top_n"
    input_keys = ["search_results"]
    output_keys = ["docs"]
    
    def __init__(self, fetch_tool: URLFetchTool = None, n: int = 2, max_content_length: int = 2000):
        """
        Initialize fetch top N step.
        
        Args:
            fetch_tool: URL fetch tool instance
            n: Number of top URLs to fetch
            max_content_length: Maximum content length per URL
        """
        self.fetch_tool = fetch_tool or get_url_fetch_tool()
        self.n = n
        self.max_content_length = max_content_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Fetch content from top N URLs in search results.
        
        Args:
            state: Current state with search_results
            runtime: Runtime context
        
        Returns:
            Dictionary with docs key (url -> content mapping)
        """
        search_results = state["search_results"]
        
        # Extract top N URLs
        urls = [r["url"] for r in search_results[:self.n] if "url" in r]
        
        if not urls:
            logger.warning("FetchTopNStep: No URLs to fetch")
            return {"docs": {}}
        
        logger.info(f"FetchTopNStep: Fetching top {len(urls)} URLs")
        
        try:
            fetched = self.fetch_tool.fetch_multiple(urls)
            
            # Build docs dict
            docs = {}
            for fetch_result in fetched:
                if fetch_result.success and fetch_result.content:
                    # Truncate content if needed
                    content = fetch_result.content
                    if len(content) > self.max_content_length:
                        content = content[:self.max_content_length]
                    
                    docs[fetch_result.url] = content
            
            logger.info(f"FetchTopNStep: Successfully fetched {len(docs)}/{len(urls)} URLs")
            
            return {"docs": docs}
        
        except Exception as e:
            logger.error(f"FetchTopNStep failed: {e}")
            return {"docs": {}}


class CitationValidationStep(BaseStep):
    """
    Validate that answer contains proper citation references.
    
    Input Keys:
        - final_answer: Generated answer text
        - citations: List of source URLs
    
    Output Keys:
        - validation_passed: Boolean indicating if validation passed
        - validation_errors: List of validation error messages
    """
    
    name = "citation_validation"
    input_keys = ["final_answer", "citations"]
    output_keys = ["validation_passed", "validation_errors"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Validate citation format in answer.
        
        Args:
            state: Current state with final_answer and citations
            runtime: Runtime context
        
        Returns:
            Dictionary with validation_passed and validation_errors keys
        """
        final_answer = state.get("final_answer", "")
        citations = state.get("citations", [])
        
        logger.info("CitationValidationStep: Validating citations")
        
        errors = []
        
        # Check if answer has content
        if not final_answer or len(final_answer.strip()) == 0:
            errors.append("Answer is empty")
            return {
                "validation_passed": False,
                "validation_errors": errors
            }
        
        # Check if citations exist
        if not citations:
            errors.append("No citations provided")
            return {
                "validation_passed": False,
                "validation_errors": errors
            }
        
        # Split answer into key points (sentences or paragraphs)
        # Look for citation references in format [N]
        import re
        
        # Find all citation references [N]
        citation_refs = re.findall(r'\[(\d+)\]', final_answer)
        
        if not citation_refs:
            errors.append("No citation references found in answer (expected format: [N])")
        else:
            # Check that citation numbers are valid
            max_citation_num = len(citations)
            for ref in citation_refs:
                ref_num = int(ref)
                if ref_num < 1 or ref_num > max_citation_num:
                    errors.append(f"Invalid citation reference [{ref}] (valid range: 1-{max_citation_num})")
        
        # Split into sentences to check if key points have citations
        sentences = re.split(r'[.!?。！？]\s+', final_answer)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
        
        if key_sentences:
            # Check if at least some key sentences have citations
            sentences_with_citations = sum(1 for s in key_sentences if re.search(r'\[\d+\]', s))
            citation_ratio = sentences_with_citations / len(key_sentences)
            
            if citation_ratio < 0.3:  # At least 30% of key sentences should have citations
                errors.append(f"Insufficient citations: only {sentences_with_citations}/{len(key_sentences)} key points have citations")
        
        validation_passed = len(errors) == 0
        
        if validation_passed:
            logger.info("CitationValidationStep: Validation passed")
        else:
            logger.warning(f"CitationValidationStep: Validation failed with {len(errors)} errors")
        
        return {
            "validation_passed": validation_passed,
            "validation_errors": errors
        }


class AnswerGenerateStep(BaseStep):
    """
    Generate final answer from summaries.
    
    Input Keys:
        - question: User's query
        - summaries: Dict mapping URL to summary
        - search_results: List of search results (for citations)
    
    Output Keys:
        - final_answer: Generated answer
        - citations: List of source URLs
    """
    
    name = "answer_generate"
    input_keys = ["question", "summaries"]
    output_keys = ["final_answer", "citations"]
    
    def __init__(self, model_name: str = "gpt2", max_context_length: int = 3000):
        """
        Initialize answer generation step.
        
        Args:
            model_name: Name of generation model
            max_context_length: Maximum context length
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Generate answer from summaries.
        
        Args:
            state: Current state with question and summaries
            runtime: Runtime context (should have model_manager)
        
        Returns:
            Dictionary with final_answer and citations keys
        """
        question = state["question"]
        summaries = state["summaries"]
        
        if not summaries:
            logger.warning("AnswerGenerateStep: No summaries available")
            return {
                "final_answer": "I couldn't find relevant information to answer your question.",
                "citations": []
            }
        
        logger.info(f"AnswerGenerateStep: Generating answer for '{question[:50]}...'")
        
        # Combine summaries into context
        context_parts = []
        citations = []
        
        for url, summary in summaries.items():
            context_parts.append(summary)
            citations.append(url)
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        # Build prompt
        prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {question}

Answer:"""
        
        try:
            # Try to use model manager if available
            if hasattr(runtime, "model_manager") and runtime.model_manager:
                answer = runtime.model_manager.infer(
                    self.model_name,
                    prompt,
                    max_new_tokens=300,
                    temperature=0.7
                )
                
                # Clean up the answer
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
            else:
                # Fallback: simple combined response
                answer = self._simple_answer(question, list(summaries.values()))
            
            logger.info(f"AnswerGenerateStep: Generated answer of length {len(answer)}")
            
            return {
                "final_answer": answer,
                "citations": citations
            }
        
        except Exception as e:
            logger.error(f"AnswerGenerateStep failed: {e}")
            # Fallback to simple answer
            answer = self._simple_answer(question, list(summaries.values()))
            return {
                "final_answer": answer,
                "citations": citations
            }
    
    def _simple_answer(self, question: str, summaries: List[str]) -> str:
        """
        Generate a simple answer without using models.
        
        Args:
            question: User's question
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
            combined = combined[:1000] + "..."
        
        return f"Based on the search results: {combined}"



class SummarizeToAnswerStep(BaseStep):
    """
    Summarize documents and output as final answer.
    
    This step is used for the summarize_url workflow where the summary
    itself is the final answer.
    
    Input Keys:
        - docs: Dict mapping URL to content
        - citations: List of source URLs
    
    Output Keys:
        - final_answer: Summary text
        - citations: List of source URLs (passed through)
    """
    
    name = "summarize_to_answer"
    input_keys = ["docs", "citations"]
    output_keys = ["final_answer", "citations"]
    
    def __init__(self, model_name: str = "t5-small", max_summary_length: int = 800):
        """
        Initialize summarize to answer step.
        
        Args:
            model_name: Name of summarization model
            max_summary_length: Maximum summary length
        """
        self.model_name = model_name
        self.max_summary_length = max_summary_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Summarize documents and output as final answer.
        
        Args:
            state: Current state with docs and citations
            runtime: Runtime context (should have model_manager)
        
        Returns:
            Dictionary with final_answer and citations keys
        """
        docs = state["docs"]
        citations = state.get("citations", [])
        
        if not docs:
            logger.warning("SummarizeToAnswerStep: No documents to summarize")
            return {
                "final_answer": "Could not fetch content from the provided URL.",
                "citations": citations
            }
        
        logger.info(f"SummarizeToAnswerStep: Summarizing {len(docs)} documents")
        
        # Get the first (and typically only) document
        url, content = next(iter(docs.items()))
        
        try:
            # Try to use model manager if available
            if hasattr(runtime, "model_manager") and runtime.model_manager:
                summary = runtime.model_manager.infer(
                    self.model_name,
                    f"summarize: {content}",
                    max_new_tokens=200
                )
            else:
                # Fallback: simple truncation
                summary = self._simple_summarize(content)
            
            logger.info(f"SummarizeToAnswerStep: Generated summary of length {len(summary)}")
            
            return {
                "final_answer": summary,
                "citations": citations
            }
        
        except Exception as e:
            logger.error(f"SummarizeToAnswerStep failed: {e}")
            # Fallback to simple summary
            summary = self._simple_summarize(content)
            return {
                "final_answer": summary,
                "citations": citations
            }
    
    def _simple_summarize(self, content: str) -> str:
        """
        Simple summarization by truncating at sentence boundaries.
        
        Args:
            content: Text to summarize
        
        Returns:
            Truncated text
        """
        if len(content) <= self.max_summary_length:
            return content
        
        # Try to truncate at sentence boundary
        truncated = content[:self.max_summary_length]
        
        # Find last sentence ending
        for end_char in [". ", "! ", "? ", "。", "！", "？"]:
            last_end = truncated.rfind(end_char)
            if last_end > self.max_summary_length // 2:
                return truncated[:last_end + 1].strip()
        
        # Fallback: truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > self.max_summary_length // 2:
            return truncated[:last_space].strip() + "..."
        
        return truncated.strip() + "..."



class AnswerGenerateFromDocsStep(BaseStep):
    """
    Generate final answer directly from docs (without summarization).
    
    This step is used for fast workflows that skip the summarization step.
    
    Input Keys:
        - question: User's query
        - docs: Dict mapping URL to content
    
    Output Keys:
        - final_answer: Generated answer
        - citations: List of source URLs
    """
    
    name = "answer_generate_from_docs"
    input_keys = ["question", "docs"]
    output_keys = ["final_answer", "citations"]
    
    def __init__(self, model_name: str = "gpt2", max_context_length: int = 2000):
        """
        Initialize answer generation from docs step.
        
        Args:
            model_name: Name of generation model
            max_context_length: Maximum context length
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Generate answer directly from docs.
        
        Args:
            state: Current state with question and docs
            runtime: Runtime context (should have model_manager)
        
        Returns:
            Dictionary with final_answer and citations keys
        """
        question = state["question"]
        docs = state["docs"]
        
        if not docs:
            logger.warning("AnswerGenerateFromDocsStep: No docs available")
            return {
                "final_answer": "I couldn't find relevant information to answer your question.",
                "citations": []
            }
        
        logger.info(f"AnswerGenerateFromDocsStep: Generating answer for '{question[:50]}...'")
        
        # Combine docs into context
        context_parts = []
        citations = []
        
        for url, content in docs.items():
            # Truncate each doc to reasonable length
            truncated = content[:500] if len(content) > 500 else content
            context_parts.append(truncated)
            citations.append(url)
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        # Build prompt
        prompt = f"""Based on the following information, answer the question.

Information:
{context}

Question: {question}

Answer:"""
        
        try:
            # Try to use model manager if available
            if hasattr(runtime, "model_manager") and runtime.model_manager:
                answer = runtime.model_manager.infer(
                    self.model_name,
                    prompt,
                    max_new_tokens=300,
                    temperature=0.7
                )
                
                # Clean up the answer
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
            else:
                # Fallback: simple combined response
                answer = self._simple_answer(question, list(docs.values()))
            
            logger.info(f"AnswerGenerateFromDocsStep: Generated answer of length {len(answer)}")
            
            return {
                "final_answer": answer,
                "citations": citations
            }
        
        except Exception as e:
            logger.error(f"AnswerGenerateFromDocsStep failed: {e}")
            # Fallback to simple answer
            answer = self._simple_answer(question, list(docs.values()))
            return {
                "final_answer": answer,
                "citations": citations
            }
    
    def _simple_answer(self, question: str, docs: List[str]) -> str:
        """
        Generate a simple answer without using models.
        
        Args:
            question: User's question
            docs: Document contents
        
        Returns:
            Simple combined answer
        """
        if not docs:
            return "I couldn't find relevant information to answer your question."
        
        # Combine first few docs
        combined = " ".join(doc[:300] for doc in docs[:2])
        
        # Truncate if needed
        if len(combined) > 800:
            combined = combined[:800] + "..."
        
        return f"Based on the search results: {combined}"


class LessonExplainStep(BaseStep):
    """
    Generate structured lesson explanation with JSON output.
    
    This step generates detailed lesson content in structured JSON format
    with sections, examples, questions, and key points.
    
    Input Keys:
        - lesson_topic: The lesson topic
        - lesson_objectives: Optional learning objectives
        - lesson_outline: Optional lesson outline
        - meta: Optional metadata (difficulty, language)
    
    Output Keys:
        - lesson_explain_structured: StructuredLesson object (if JSON parse succeeds)
        - teaching_text: Plain text fallback (if JSON parse fails)
        - meta: Updated with parse_error if JSON parsing fails
    
    Requirements:
        - 18.1: JSON-structured lesson output
        - 18.3: Parse JSON response and create StructuredLesson
        - 18.4: Fallback to plain text on parse error
    """
    
    name = "lesson_explain"
    input_keys = ["lesson_topic"]
    output_keys = ["lesson_explain_structured", "teaching_text"]
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_new_tokens: int = 1500,
        temperature: float = 0.7
    ):
        """
        Initialize lesson explain step.
        
        Args:
            model_name: Model to use for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Execute lesson explanation generation with JSON output.
        
        Args:
            state: Current workflow state
            runtime: Runtime context with model_manager
        
        Returns:
            Dictionary with lesson_explain_structured or teaching_text
        """
        import json
        from mm_orch.workflows.lesson_structure import StructuredLesson
        
        topic = state["lesson_topic"]
        objectives = state.get("lesson_objectives", [])
        outline = state.get("lesson_outline", [])
        meta = state.get("meta", {})
        
        difficulty = meta.get("difficulty", "intermediate")
        language = meta.get("language", "zh")
        
        logger.info(f"LessonExplainStep: Generating structured lesson for '{topic[:50]}...'")
        
        # Build prompt requesting JSON output
        prompt = self._build_json_prompt(topic, objectives, outline, difficulty, language)
        
        try:
            # Try to use model manager if available
            if hasattr(runtime, "model_manager") and runtime.model_manager:
                response = runtime.model_manager.infer(
                    self.model_name,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
            else:
                # Fallback: generate template-based JSON
                response = self._generate_template_json(topic, difficulty, language)
            
            # Try to parse JSON response
            structured_lesson = self._parse_json_response(response, topic, difficulty)
            
            if structured_lesson:
                logger.info(
                    f"LessonExplainStep: Successfully parsed JSON with "
                    f"{len(structured_lesson.sections)} sections"
                )
                return {
                    "lesson_explain_structured": structured_lesson.to_json()
                }
            else:
                # JSON parsing failed, use plain text fallback
                logger.warning("LessonExplainStep: JSON parsing failed, using plain text fallback")
                
                # Update meta with parse error
                updated_meta = meta.copy()
                updated_meta["parse_error"] = "Failed to parse JSON response"
                
                return {
                    "teaching_text": response,
                    "meta": updated_meta
                }
        
        except Exception as e:
            logger.error(f"LessonExplainStep failed: {e}")
            
            # Update meta with error
            updated_meta = meta.copy()
            updated_meta["parse_error"] = str(e)
            
            # Return minimal fallback
            fallback_text = self._generate_fallback_text(topic, language)
            return {
                "teaching_text": fallback_text,
                "meta": updated_meta
            }
    
    def _build_json_prompt(
        self,
        topic: str,
        objectives: List[str],
        outline: List[str],
        difficulty: str,
        language: str
    ) -> str:
        """
        Build prompt requesting JSON-structured lesson output.
        
        Args:
            topic: Lesson topic
            objectives: Learning objectives
            outline: Lesson outline
            difficulty: Difficulty level
            language: Output language
        
        Returns:
            Prompt string requesting JSON output
        """
        if language == "zh":
            prompt = f"""请为主题"{topic}"生成一个结构化的教学内容，以JSON格式输出。

难度级别: {self._translate_difficulty(difficulty, language)}

JSON格式要求:
{{
  "topic": "{topic}",
  "grade": "适合的年级或难度",
  "sections": [
    {{
      "name": "章节名称（如：导入、新授、练习、小结）",
      "teacher_say": "教师讲解内容",
      "student_may_say": "学生可能的回答（可选）",
      "examples": ["示例1", "示例2"],
      "questions": ["问题1", "问题2"],
      "key_points": ["要点1", "要点2"],
      "tips": "教学提示（可选）"
    }}
  ]
}}

要求:
- 至少包含3个章节
- 每个章节必须有name和teacher_say
- 至少一个章节要包含examples或questions
- 内容要清晰、易懂

请直接输出JSON，不要添加其他说明文字:"""
        else:
            prompt = f"""Please generate structured teaching content for the topic "{topic}" in JSON format.

Difficulty Level: {difficulty}

JSON format requirements:
{{
  "topic": "{topic}",
  "grade": "appropriate grade or difficulty level",
  "sections": [
    {{
      "name": "section name (e.g., Introduction, Main Content, Practice, Summary)",
      "teacher_say": "teacher's explanation content",
      "student_may_say": "expected student responses (optional)",
      "examples": ["example 1", "example 2"],
      "questions": ["question 1", "question 2"],
      "key_points": ["key point 1", "key point 2"],
      "tips": "teaching tips (optional)"
    }}
  ]
}}

Requirements:
- At least 3 sections
- Each section must have name and teacher_say
- At least one section must contain examples or questions
- Content should be clear and easy to understand

Please output JSON directly without additional explanatory text:"""
        
        return prompt
    
    def _parse_json_response(
        self,
        response: str,
        topic: str,
        difficulty: str
    ) -> "StructuredLesson":
        """
        Parse JSON response into StructuredLesson.
        
        Args:
            response: Model response text
            topic: Lesson topic (for fallback)
            difficulty: Difficulty level (for fallback)
        
        Returns:
            StructuredLesson object if parsing succeeds, None otherwise
        """
        import json
        from mm_orch.workflows.lesson_structure import StructuredLesson
        
        try:
            # Try to extract JSON from response
            # Look for JSON object markers
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON object found in response")
                return None
            
            json_str = response[start_idx:end_idx + 1]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Create StructuredLesson
            lesson = StructuredLesson.from_json(data)
            
            # Validate the lesson
            is_valid, errors = lesson.validate()
            if not is_valid:
                logger.warning(f"Lesson validation failed: {errors}")
                # Still return the lesson even if validation fails
                # The validation step will handle this
            
            return lesson
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return None
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Lesson structure error: {e}")
            return None
    
    def _generate_template_json(
        self,
        topic: str,
        difficulty: str,
        language: str
    ) -> str:
        """
        Generate template-based JSON when model is unavailable.
        
        Args:
            topic: Lesson topic
            difficulty: Difficulty level
            language: Output language
        
        Returns:
            JSON string with template lesson
        """
        import json
        from mm_orch.workflows.lesson_structure import StructuredLesson, LessonSection
        
        if language == "zh":
            sections = [
                LessonSection(
                    name="导入",
                    teacher_say=f"今天我们要学习{topic}。这是一个重要的概念，让我们一起来探索。",
                    student_may_say="好的，老师！",
                    key_points=[f"{topic}的重要性", "学习目标"]
                ),
                LessonSection(
                    name="新授",
                    teacher_say=f"{topic}的核心内容包括以下几个方面...",
                    examples=[f"{topic}的示例1", f"{topic}的示例2"],
                    key_points=[f"{topic}的定义", f"{topic}的特点"]
                ),
                LessonSection(
                    name="练习",
                    teacher_say="现在让我们通过一些练习来巩固所学内容。",
                    questions=[f"请解释{topic}的含义", f"请举例说明{topic}的应用"],
                    key_points=["实践应用", "知识巩固"]
                ),
                LessonSection(
                    name="小结",
                    teacher_say=f"今天我们学习了{topic}，掌握了其核心概念和应用方法。",
                    key_points=["总结要点", "课后思考"]
                )
            ]
            grade = self._translate_difficulty(difficulty, language)
        else:
            sections = [
                LessonSection(
                    name="Introduction",
                    teacher_say=f"Today we will learn about {topic}. This is an important concept, let's explore it together.",
                    student_may_say="Okay, teacher!",
                    key_points=[f"Importance of {topic}", "Learning objectives"]
                ),
                LessonSection(
                    name="Main Content",
                    teacher_say=f"The core content of {topic} includes the following aspects...",
                    examples=[f"Example 1 of {topic}", f"Example 2 of {topic}"],
                    key_points=[f"Definition of {topic}", f"Characteristics of {topic}"]
                ),
                LessonSection(
                    name="Practice",
                    teacher_say="Now let's consolidate what we've learned through some exercises.",
                    questions=[f"Please explain the meaning of {topic}", f"Please give examples of {topic} applications"],
                    key_points=["Practical application", "Knowledge consolidation"]
                ),
                LessonSection(
                    name="Summary",
                    teacher_say=f"Today we learned about {topic}, mastering its core concepts and application methods.",
                    key_points=["Summary points", "After-class reflection"]
                )
            ]
            grade = difficulty
        
        lesson = StructuredLesson(topic=topic, grade=grade, sections=sections)
        return lesson.to_json_string()
    
    def _generate_fallback_text(self, topic: str, language: str) -> str:
        """
        Generate plain text fallback when all else fails.
        
        Args:
            topic: Lesson topic
            language: Output language
        
        Returns:
            Plain text lesson content
        """
        if language == "zh":
            return f"""# {topic} 教学内容

## 导入
今天我们要学习{topic}。

## 主要内容
{topic}是一个重要的概念，包含以下要点：
- 基本定义
- 核心特点
- 应用场景

## 练习
请思考：
1. {topic}的含义是什么？
2. {topic}有哪些应用？

## 小结
通过本节课的学习，我们了解了{topic}的基本概念和应用。"""
        else:
            return f"""# Teaching Content: {topic}

## Introduction
Today we will learn about {topic}.

## Main Content
{topic} is an important concept that includes the following points:
- Basic definition
- Core characteristics
- Application scenarios

## Practice
Please think about:
1. What is the meaning of {topic}?
2. What are the applications of {topic}?

## Summary
Through this lesson, we learned about the basic concepts and applications of {topic}."""
    
    def _translate_difficulty(self, difficulty: str, language: str) -> str:
        """
        Translate difficulty level to target language.
        
        Args:
            difficulty: Difficulty level in English
            language: Target language
        
        Returns:
            Translated difficulty level
        """
        if language == "zh":
            translations = {
                "beginner": "初级",
                "intermediate": "中级",
                "advanced": "高级"
            }
            return translations.get(difficulty, "中级")
        return difficulty



class LessonValidationStep(BaseStep):
    """
    Validate structured lesson output.
    
    This step validates that the generated lesson meets minimum requirements:
    - At least 3 sections
    - At least one section with examples or questions
    - Calculates completeness score
    
    Input Keys:
        - lesson_explain_structured: StructuredLesson JSON dict
        - meta: Metadata dict
    
    Output Keys:
        - meta: Updated with validation_errors and completeness_score
    
    Requirements:
        - 19.1: Check minimum sections requirement
        - 19.2: Check content requirement (examples or questions)
        - 19.3: Record validation errors in trace
        - 19.4: Calculate completeness score
    """
    
    name = "lesson_validation"
    input_keys = ["lesson_explain_structured"]
    output_keys = ["meta"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Execute lesson validation.
        
        Args:
            state: Current workflow state
            runtime: Runtime context
        
        Returns:
            Dictionary with updated meta containing validation results
        """
        from mm_orch.workflows.lesson_structure import StructuredLesson
        
        lesson_data = state.get("lesson_explain_structured")
        meta = state.get("meta", {}).copy()
        
        if not lesson_data:
            logger.warning("LessonValidationStep: No structured lesson found")
            meta["validation_errors"] = ["No structured lesson data found"]
            meta["completeness_score"] = 0.0
            return {"meta": meta}
        
        try:
            # Parse lesson from JSON
            lesson = StructuredLesson.from_json(lesson_data)
            
            # Validate lesson structure
            is_valid, errors = lesson.validate()
            
            # Calculate completeness score
            completeness = lesson.completeness_score()
            
            # Update meta with validation results
            if errors:
                meta["validation_errors"] = errors
                logger.warning(f"LessonValidationStep: Validation failed with {len(errors)} errors")
            else:
                logger.info("LessonValidationStep: Validation passed")
            
            meta["completeness_score"] = completeness
            meta["validation_passed"] = is_valid
            
            logger.info(f"LessonValidationStep: Completeness score = {completeness:.2f}")
            
            return {"meta": meta}
        
        except Exception as e:
            logger.error(f"LessonValidationStep failed: {e}")
            meta["validation_errors"] = [f"Validation error: {str(e)}"]
            meta["completeness_score"] = 0.0
            return {"meta": meta}
