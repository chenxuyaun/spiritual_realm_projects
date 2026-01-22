"""
对话管理器单元测试
"""

import pytest
from unittest.mock import Mock

from mm_orch.runtime.conversation import (
    ConversationManager,
    Conversation,
    Message,
)


class TestMessage:
    """Message测试"""
    
    def test_create_message(self):
        """测试创建消息"""
        msg = Message(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_to_dict(self):
        """测试转换为字典"""
        msg = Message(role="assistant", content="Hi there!")
        result = msg.to_dict()
        
        assert result == {"role": "assistant", "content": "Hi there!"}


class TestConversation:
    """Conversation测试"""
    
    def test_create_empty(self):
        """测试创建空对话"""
        conv = Conversation()
        
        assert len(conv.messages) == 0
        assert conv.system_prompt is None
    
    def test_create_with_system_prompt(self):
        """测试创建带系统提示的对话"""
        conv = Conversation(system_prompt="You are a helpful assistant.")
        
        assert conv.system_prompt == "You are a helpful assistant."
    
    def test_add_message(self):
        """测试添加消息"""
        conv = Conversation()
        conv.add_message("user", "Hello")
        
        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hello"
    
    def test_add_user_message(self):
        """测试添加用户消息"""
        conv = Conversation()
        conv.add_user_message("Hello")
        
        assert conv.messages[0].role == "user"
    
    def test_add_assistant_message(self):
        """测试添加助手消息"""
        conv = Conversation()
        conv.add_assistant_message("Hi!")
        
        assert conv.messages[0].role == "assistant"
    
    def test_get_last_user_message(self):
        """测试获取最后一条用户消息"""
        conv = Conversation()
        conv.add_user_message("First")
        conv.add_assistant_message("Response")
        conv.add_user_message("Second")
        
        assert conv.get_last_user_message() == "Second"
    
    def test_get_last_user_message_empty(self):
        """测试空对话获取最后一条用户消息"""
        conv = Conversation()
        assert conv.get_last_user_message() is None
    
    def test_get_last_assistant_message(self):
        """测试获取最后一条助手消息"""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        conv.add_user_message("How are you?")
        
        assert conv.get_last_assistant_message() == "Hi!"
    
    def test_clear(self):
        """测试清空对话"""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        
        conv.clear()
        
        assert len(conv.messages) == 0
    
    def test_to_list(self):
        """测试转换为列表"""
        conv = Conversation(system_prompt="System prompt")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        
        result = conv.to_list()
        
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
    
    def test_len(self):
        """测试长度"""
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        
        assert len(conv) == 2


class TestConversationManager:
    """ConversationManager测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        manager = ConversationManager()
        
        assert manager.model_type == "qwen-chat"
        assert manager.max_history_tokens == 4096
    
    def test_init_custom(self):
        """测试自定义初始化"""
        manager = ConversationManager(
            model_type="gpt2",
            max_history_tokens=2048,
            system_prompt="You are helpful."
        )
        
        assert manager.model_type == "gpt2"
        assert manager.max_history_tokens == 2048
        assert manager.conversation.system_prompt == "You are helpful."
    
    def test_add_turn(self):
        """测试添加对话轮次"""
        manager = ConversationManager()
        manager.add_turn("user", "Hello")
        manager.add_turn("assistant", "Hi!")
        
        assert manager.get_message_count() == 2
    
    def test_add_user_input(self):
        """测试添加用户输入"""
        manager = ConversationManager()
        manager.add_user_input("Hello")
        
        assert manager.conversation.messages[0].role == "user"
    
    def test_add_assistant_response(self):
        """测试添加助手响应"""
        manager = ConversationManager()
        manager.add_assistant_response("Hi!")
        
        assert manager.conversation.messages[0].role == "assistant"
    
    def test_build_qwen_prompt(self):
        """测试构建Qwen格式提示词"""
        manager = ConversationManager(model_type="qwen-chat")
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        assert "<|im_start|>user" in prompt
        assert "Hello" in prompt
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant" in prompt
    
    def test_build_qwen_prompt_with_system(self):
        """测试构建带系统提示的Qwen格式"""
        manager = ConversationManager(
            model_type="qwen-chat",
            system_prompt="You are helpful."
        )
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        assert "<|im_start|>system" in prompt
        assert "You are helpful." in prompt
    
    def test_build_gpt2_prompt(self):
        """测试构建GPT-2格式提示词"""
        manager = ConversationManager(model_type="gpt2")
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt
    
    def test_build_gpt2_prompt_with_system(self):
        """测试构建带系统提示的GPT-2格式"""
        manager = ConversationManager(
            model_type="gpt2",
            system_prompt="You are helpful."
        )
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        assert "System: You are helpful." in prompt
    
    def test_build_gpt2_prompt_multi_turn(self):
        """测试构建多轮GPT-2格式"""
        manager = ConversationManager(model_type="gpt2")
        manager.add_user_input("Hello")
        manager.add_assistant_response("Hi!")
        manager.add_user_input("How are you?")
        
        prompt = manager.build_prompt()
        
        assert "User: Hello" in prompt
        assert "Assistant: Hi!" in prompt
        assert "User: How are you?" in prompt
    
    def test_build_chatml_prompt(self):
        """测试构建ChatML格式提示词"""
        manager = ConversationManager(model_type="chatml")
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        # ChatML格式与Qwen格式相同
        assert "<|im_start|>user" in prompt
    
    def test_build_llama_prompt(self):
        """测试构建Llama格式提示词"""
        manager = ConversationManager(model_type="llama")
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        assert "[INST]" in prompt
        assert "Hello" in prompt
        assert "[/INST]" in prompt
    
    def test_build_llama_prompt_with_system(self):
        """测试构建带系统提示的Llama格式"""
        manager = ConversationManager(
            model_type="llama",
            system_prompt="You are helpful."
        )
        manager.add_user_input("Hello")
        
        prompt = manager.build_prompt()
        
        assert "<<SYS>>" in prompt
        assert "You are helpful." in prompt
        assert "<</SYS>>" in prompt
    
    def test_truncate_history(self):
        """测试截断历史"""
        manager = ConversationManager(max_history_tokens=100)
        
        # 添加很多消息
        for i in range(20):
            manager.add_user_input(f"Message {i} " * 10)
            manager.add_assistant_response(f"Response {i} " * 10)
        
        # Mock tokenizer that returns many tokens
        mock_tokenizer = Mock()
        # 返回超过max_tokens的token数，触发截断
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x) // 2))
        
        initial_count = manager.get_message_count()
        removed = manager.truncate_history(mock_tokenizer, max_tokens=100)
        
        # 应该移除了一些消息或者消息数减少了
        assert removed > 0 or manager.get_message_count() < initial_count
    
    def test_get_history(self):
        """测试获取历史（Qwen格式）"""
        manager = ConversationManager()
        manager.add_user_input("Hello")
        manager.add_assistant_response("Hi!")
        manager.add_user_input("How are you?")
        manager.add_assistant_response("I'm fine!")
        
        history = manager.get_history()
        
        assert len(history) == 2
        assert history[0] == ("Hello", "Hi!")
        assert history[1] == ("How are you?", "I'm fine!")
    
    def test_clear_history(self):
        """测试清空历史"""
        manager = ConversationManager()
        manager.add_user_input("Hello")
        manager.add_assistant_response("Hi!")
        
        manager.clear_history()
        
        assert manager.get_message_count() == 0
    
    def test_set_system_prompt(self):
        """测试设置系统提示词"""
        manager = ConversationManager()
        manager.set_system_prompt("New system prompt")
        
        assert manager.conversation.system_prompt == "New system prompt"
    
    def test_to_dict(self):
        """测试转换为字典"""
        manager = ConversationManager(
            model_type="gpt2",
            system_prompt="System"
        )
        manager.add_user_input("Hello")
        
        result = manager.to_dict()
        
        assert result["model_type"] == "gpt2"
        assert result["system_prompt"] == "System"
        assert result["message_count"] == 1
        assert len(result["messages"]) == 2  # system + user
    
    def test_unknown_model_type(self):
        """测试未知模型类型"""
        manager = ConversationManager(model_type="unknown")
        manager.add_user_input("Hello")
        
        # 应该使用简单格式
        prompt = manager.build_prompt()
        
        assert "Human: Hello" in prompt or "User: Hello" in prompt
