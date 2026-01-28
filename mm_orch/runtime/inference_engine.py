"""
推理引擎模块

提供模型推理功能，支持单次生成、流式生成和批量生成。
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch

try:
    from transformers import (
        TextIteratorStreamer,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from mm_orch.exceptions import InferenceError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """生成配置"""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def validate(self) -> None:
        """验证配置参数"""
        if self.max_new_tokens <= 0:
            raise ValidationError("max_new_tokens must be positive")
        if self.max_new_tokens > 4096:
            raise ValidationError("max_new_tokens exceeds maximum (4096)")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValidationError("top_p must be between 0.0 and 1.0")
        if self.top_k < 0:
            raise ValidationError("top_k must be non-negative")
        if self.repetition_penalty < 1.0:
            raise ValidationError("repetition_penalty must be >= 1.0")
        if self.num_beams < 1:
            raise ValidationError("num_beams must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }


@dataclass
class GenerationResult:
    """生成结果"""

    text: str
    input_tokens: int
    output_tokens: int
    total_time: float  # seconds
    tokens_per_second: float
    finish_reason: str = "stop"  # "stop", "length", "error"

    @property
    def ttft(self) -> Optional[float]:
        """首token延迟（如果可用）"""
        return getattr(self, "_ttft", None)


class InferenceEngine:
    """
    推理引擎

    提供模型推理功能：
    - 单次生成
    - 流式生成
    - 批量生成
    - 参数验证
    """

    DEFAULT_CONFIG = GenerationConfig()

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        default_config: Optional[GenerationConfig] = None,
    ):
        """
        初始化推理引擎

        Args:
            model: 预训练模型
            tokenizer: 分词器
            device: 设备
            default_config: 默认生成配置
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.default_config = default_config or self.DEFAULT_CONFIG

        # 确保tokenizer有必要的token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs
    ) -> GenerationResult:
        """
        单次生成

        Args:
            prompt: 输入提示词
            config: 生成配置
            **kwargs: 额外的生成参数

        Returns:
            GenerationResult: 生成结果

        Raises:
            InferenceError: 推理失败
        """
        config = config or self.default_config
        config.validate()

        start_time = time.time()

        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            input_tokens = input_ids.shape[1]

            # 准备生成参数
            gen_kwargs = config.to_dict()
            gen_kwargs["pad_token_id"] = config.pad_token_id or self.tokenizer.pad_token_id
            gen_kwargs["eos_token_id"] = config.eos_token_id or self.tokenizer.eos_token_id

            # 合并额外参数
            gen_kwargs.update(kwargs)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, attention_mask=attention_mask, **gen_kwargs
                )

            # 解码输出
            output_ids = outputs[0][input_tokens:]
            output_tokens = len(output_ids)
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            total_time = time.time() - start_time
            tokens_per_second = output_tokens / total_time if total_time > 0 else 0

            # 确定结束原因
            finish_reason = "stop"
            if output_tokens >= config.max_new_tokens:
                finish_reason = "length"

            return GenerationResult(
                text=generated_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_time=total_time,
                tokens_per_second=tokens_per_second,
                finish_reason=finish_reason,
            )

        except torch.cuda.OutOfMemoryError as e:
            raise InferenceError(
                f"GPU out of memory during inference: {e}", input_length=len(prompt)
            )
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}", input_length=len(prompt))

    def generate_stream(
        self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs
    ) -> Iterator[str]:
        """
        流式生成

        Args:
            prompt: 输入提示词
            config: 生成配置
            **kwargs: 额外的生成参数

        Yields:
            str: 生成的文本片段

        Raises:
            InferenceError: 推理失败
        """
        if not HAS_TRANSFORMERS:
            raise InferenceError("transformers library required for streaming")

        config = config or self.default_config
        config.validate()

        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # 创建streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # 准备生成参数
            gen_kwargs = config.to_dict()
            gen_kwargs["pad_token_id"] = config.pad_token_id or self.tokenizer.pad_token_id
            gen_kwargs["eos_token_id"] = config.eos_token_id or self.tokenizer.eos_token_id
            gen_kwargs["streamer"] = streamer
            gen_kwargs.update(kwargs)

            # 在后台线程中生成
            import threading

            def generate_thread():
                with torch.no_grad():
                    self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

            thread = threading.Thread(target=generate_thread)
            thread.start()

            # 流式输出
            for text in streamer:
                yield text

            thread.join()

        except Exception as e:
            raise InferenceError(f"Streaming inference failed: {e}", input_length=len(prompt))

    def batch_generate(
        self, prompts: List[str], config: Optional[GenerationConfig] = None, **kwargs
    ) -> List[GenerationResult]:
        """
        批量生成

        Args:
            prompts: 输入提示词列表
            config: 生成配置
            **kwargs: 额外的生成参数

        Returns:
            List[GenerationResult]: 生成结果列表

        Raises:
            InferenceError: 推理失败
        """
        if not prompts:
            return []

        config = config or self.default_config
        config.validate()

        start_time = time.time()

        try:
            # 批量编码
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            batch_size = input_ids.shape[0]
            input_lengths = attention_mask.sum(dim=1).tolist()

            # 准备生成参数
            gen_kwargs = config.to_dict()
            gen_kwargs["pad_token_id"] = config.pad_token_id or self.tokenizer.pad_token_id
            gen_kwargs["eos_token_id"] = config.eos_token_id or self.tokenizer.eos_token_id
            gen_kwargs.update(kwargs)

            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, attention_mask=attention_mask, **gen_kwargs
                )

            total_time = time.time() - start_time

            # 解码每个输出
            results = []
            for i in range(batch_size):
                input_len = input_lengths[i]
                output_ids = outputs[i][input_len:]
                output_tokens = len(output_ids)

                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                # 每个样本的时间（平均）
                sample_time = total_time / batch_size
                tokens_per_second = output_tokens / sample_time if sample_time > 0 else 0

                finish_reason = "stop"
                if output_tokens >= config.max_new_tokens:
                    finish_reason = "length"

                results.append(
                    GenerationResult(
                        text=generated_text,
                        input_tokens=input_len,
                        output_tokens=output_tokens,
                        total_time=sample_time,
                        tokens_per_second=tokens_per_second,
                        finish_reason=finish_reason,
                    )
                )

            return results

        except torch.cuda.OutOfMemoryError as e:
            raise InferenceError(
                f"GPU out of memory during batch inference: {e}",
                input_length=sum(len(p) for p in prompts),
            )
        except Exception as e:
            raise InferenceError(
                f"Batch inference failed: {e}", input_length=sum(len(p) for p in prompts)
            )

    @staticmethod
    def validate_generation_params(
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> GenerationConfig:
        """
        验证并创建生成配置

        Args:
            max_new_tokens: 最大新token数
            temperature: 温度
            top_p: Top-p采样
            top_k: Top-k采样
            repetition_penalty: 重复惩罚

        Returns:
            GenerationConfig: 验证后的配置

        Raises:
            ValidationError: 参数验证失败
        """
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        config.validate()
        return config
