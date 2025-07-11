# utils/conversation_history.py
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class ConversationHistory:
    """Manage conversation history for context-aware responses"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 10
    
    def add_message(self, role: str, content: str):
        """Add a message to history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_context(self) -> str:
        """Get formatted conversation context"""
        if not self.messages:
            return ""
        
        context = "Previous conversation:\n"
        for msg in self.messages[-5:]:  # Last 5 messages
            context += f"{msg['role']}: {msg['content']}\n"
        return context
    
    def clear(self):
        """Clear all messages"""
        self.messages.clear()
    
    def get_recent_messages(self, count: int = 3) -> List[Dict[str, str]]:
        """Get recent messages"""
        return self.messages[-count:] if count <= len(self.messages) else self.messages
    
    def get_user_queries(self) -> List[str]:
        """Get all user queries from history"""
        return [msg["content"] for msg in self.messages if msg["role"] == "user"]