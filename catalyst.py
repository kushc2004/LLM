from typing import Dict, Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import torch

class Agent1:
    def __init__(self, name, description, model_name, **kwargs):
        self.name = name
        self.description = description
        self.system_prompt = "You're a helpful assistant."
        self.kwargs = kwargs
        self.model_name = model_name

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize the model with empty weights and use `init_empty_weights`
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)

        # Infer the device map
        device_map = infer_auto_device_map(self.model, max_memory={0: "12GiB", "cpu": "30GiB"})

        # Dispatch the model to the appropriate devices
        self.model = dispatch_model(self.model, device_map=device_map)

    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List] = None,
        seed: int = 10,
    ) -> str:
        # Ensure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)

        # Ensure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        # Concatenate messages into a single prompt
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Tokenize the input
        inputs = self.tokenizer(full_prompt, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200, do_sample=True, top_p=0.95, top_k=50, temperature=0.7)

        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def generate_reply(
        self,
        task: str,
        state: Dict,
        sender: "Agent1",
    ) -> Tuple[str, Dict]:
        return (
            "This is a reply from the agent. REPLY NOT IMPLEMENTED! Terminate the whole process!",
            state,
        )
    
