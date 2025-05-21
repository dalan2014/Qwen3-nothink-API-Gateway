from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str # This will be the model expected by the vLLM server
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = Field(default=1, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=False, description="If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message.")
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # To catch any other parameters clients might send
    class Config:
        extra = "allow"

class BackendChatCompletionRequest(ChatCompletionRequest):
    """
    Request model for the backend vLLM service, including Qwen-specific parameters.
    """
    chat_template_kwargs: Optional[Dict[str, Any]] = None