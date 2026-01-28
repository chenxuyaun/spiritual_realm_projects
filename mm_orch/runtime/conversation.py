"""
对话管理器模块

提供多轮对话管理功能，支持不同模型的对话格式构建。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """对话消息"""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """对话历史"""

    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        self.messages.append(Message(role=role, content=content))

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.add_message("assistant", content)

    def get_last_user_message(self) -> Optional[str]:
        """获取最后一条用户消息"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """获取最后一条助手消息"""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def clear(self) -> None:
        """清空对话历史"""
        self.messages.clear()

    def to_list(self) -> List[Dict[str, str]]:
        """转换为消息列表"""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend([msg.to_dict() for msg in self.messages])
        return result

    def __len__(self) -> int:
        return len(self.messages)


class ConversationManager:
    """
    对话管理器

    负责管理多轮对话，支持不同模型的对话格式：
    - Qwen-Chat格式
    - GPT-2格式（提示词拼接）
    - 通用ChatML格式
    """

    # 支持的模型类型
    SUPPORTED_MODEL_TYPES = ["qwen-chat", "gpt2", "chatml", "llama"]

    def __init__(
        self,
        model_type: str = "qwen-chat",
        max_history_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化对话管理器

        Args:
            model_type: 模型类型
            max_history_tokens: 最大历史token数
            system_prompt: 系统提示词
        """
        self.model_type = model_type
        self.max_history_tokens = max_history_tokens
        self.conversation = Conversation(system_prompt=system_prompt)

        if model_type not in self.SUPPORTED_MODEL_TYPES:
            logger.warning(f"Unknown model type: {model_type}, using default format")

    def add_turn(self, role: str, content: str) -> None:
        """
        添加对话轮次

        Args:
            role: 角色 ("user" 或 "assistant")
            content: 内容
        """
        self.conversation.add_message(role, content)

    def add_user_input(self, content: str) -> None:
        """添加用户输入"""
        self.conversation.add_user_message(content)

    def add_assistant_response(self, content: str) -> None:
        """添加助手响应"""
        self.conversation.add_assistant_message(content)

    def build_prompt(
        self, system_prompt: Optional[str] = None, include_generation_prompt: bool = True
    ) -> str:
        """
        构建完整提示词

        Args:
            system_prompt: 系统提示词（覆盖默认）
            include_generation_prompt: 是否包含生成提示

        Returns:
            str: 构建的提示词
        """
        sys_prompt = system_prompt or self.conversation.system_prompt

        if self.model_type == "qwen-chat":
            return self._build_qwen_prompt(sys_prompt, include_generation_prompt)
        elif self.model_type == "gpt2":
            return self._build_gpt2_prompt(sys_prompt)
        elif self.model_type == "chatml":
            return self._build_chatml_prompt(sys_prompt, include_generation_prompt)
        elif self.model_type == "llama":
            return self._build_llama_prompt(sys_prompt, include_generation_prompt)
        else:
            # 默认使用简单格式
            return self._build_simple_prompt(sys_prompt)

    def _build_qwen_prompt(
        self, system_prompt: Optional[str], include_generation_prompt: bool
    ) -> str:
        """
        构建Qwen-Chat格式提示词

        Qwen-Chat使用特殊的对话格式：
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        """
        parts = []

        # 系统提示
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # 对话历史
        for msg in self.conversation.messages:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

        # 生成提示
        if include_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _build_gpt2_prompt(self, system_prompt: Optional[str]) -> str:
        """
        构建GPT-2格式提示词

        GPT-2使用简单的文本拼接格式：
        System: {system_prompt}

        User: {user_message}
        Assistant: {assistant_message}

        User: {user_message}
        Assistant:
        """
        parts = []

        # 系统提示
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")

        # 对话历史
        for msg in self.conversation.messages:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")

        # 添加生成提示
        if self.conversation.messages and self.conversation.messages[-1].role == "user":
            parts.append("Assistant:")

        return "\n".join(parts)

    def _build_chatml_prompt(
        self, system_prompt: Optional[str], include_generation_prompt: bool
    ) -> str:
        """
        构建ChatML格式提示词

        ChatML格式：
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
        """
        # ChatML格式与Qwen格式相同
        return self._build_qwen_prompt(system_prompt, include_generation_prompt)

    def _build_llama_prompt(
        self, system_prompt: Optional[str], include_generation_prompt: bool
    ) -> str:
        """
        构建Llama格式提示词

        Llama 2格式：
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {user_message} [/INST] {assistant_message} [INST] {user_message} [/INST]
        """
        parts = []

        # 构建第一个用户消息（包含系统提示）
        first_user_msg = None
        remaining_messages = []

        for i, msg in enumerate(self.conversation.messages):
            if msg.role == "user" and first_user_msg is None:
                first_user_msg = msg.content
            else:
                remaining_messages.append(msg)

        if first_user_msg:
            if system_prompt:
                parts.append(
                    f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{first_user_msg} [/INST]"
                )
            else:
                parts.append(f"[INST] {first_user_msg} [/INST]")

        # 添加剩余消息
        for msg in remaining_messages:
            if msg.role == "assistant":
                parts.append(f" {msg.content}")
            elif msg.role == "user":
                parts.append(f" [INST] {msg.content} [/INST]")

        return "".join(parts)

    def _build_simple_prompt(self, system_prompt: Optional[str]) -> str:
        """构建简单格式提示词"""
        parts = []

        if system_prompt:
            parts.append(f"{system_prompt}\n")

        for msg in self.conversation.messages:
            if msg.role == "user":
                parts.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"AI: {msg.content}")

        if self.conversation.messages and self.conversation.messages[-1].role == "user":
            parts.append("AI:")

        return "\n".join(parts)

    def truncate_history(self, tokenizer: Any, max_tokens: Optional[int] = None) -> int:
        """
        截断历史以适应上下文窗口

        Args:
            tokenizer: 分词器
            max_tokens: 最大token数（默认使用初始化时的值）

        Returns:
            int: 移除的消息数
        """
        max_tokens = max_tokens or self.max_history_tokens
        removed_count = 0

        while len(self.conversation.messages) > 0:
            # 构建当前提示词
            prompt = self.build_prompt()

            # 计算token数
            try:
                tokens = tokenizer.encode(prompt)
                token_count = len(tokens)
            except Exception:
                # 如果编码失败，使用字符数估算
                token_count = len(prompt) // 4

            if token_count <= max_tokens:
                break

            # 移除最早的消息对（保持对话完整性）
            if len(self.conversation.messages) >= 2:
                # 移除最早的用户-助手对
                self.conversation.messages.pop(0)
                if self.conversation.messages and self.conversation.messages[0].role == "assistant":
                    self.conversation.messages.pop(0)
                removed_count += 2
            else:
                # 只剩一条消息，移除它
                self.conversation.messages.pop(0)
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Truncated {removed_count} messages from history")

        return removed_count

    def get_history(self) -> List[Tuple[str, str]]:
        """
        获取对话历史（Qwen格式）

        Returns:
            List[Tuple[str, str]]: (用户消息, 助手回复) 列表
        """
        history = []
        user_msg = None

        for msg in self.conversation.messages:
            if msg.role == "user":
                user_msg = msg.content
            elif msg.role == "assistant" and user_msg is not None:
                history.append((user_msg, msg.content))
                user_msg = None

        return history

    def clear_history(self) -> None:
        """清空对话历史"""
        self.conversation.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.conversation.system_prompt = prompt

    def get_message_count(self) -> int:
        """获取消息数量"""
        return len(self.conversation)

    def get_mode(self) -> str:
        """
        Get execution mode for routing.

        Returns:
            "chat" for conversation mode, "default" for single-shot queries

        Requirements: 21.1
        """
        # If there are messages in the conversation, we're in chat mode
        if len(self.conversation) > 0:
            return "chat"
        return "default"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type,
            "max_history_tokens": self.max_history_tokens,
            "system_prompt": self.conversation.system_prompt,
            "messages": self.conversation.to_list(),
            "message_count": len(self.conversation),
            "mode": self.get_mode(),
        }
