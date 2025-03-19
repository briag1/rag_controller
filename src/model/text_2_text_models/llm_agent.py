
import re

from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMAgent:
  def __init__(self, llm_name: str, return_full_hisory : Optional[bool] = False ):
    self.device = "cuda" # for GPU usage or "cpu" for CPU usage
    self.llm = AutoModelForCausalLM.from_pretrained(llm_name).to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
    self.return_full_history = return_full_hisory

  def query(self, query: str) -> str:
    messages = [{"role": "user", "content": query}]
    input_text=self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
    outputs = self.llm.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    response = self.tokenizer.decode(outputs[0])
    parsed_response = self.parse_output(response)
    return parsed_response

  def parse_output(self, output: str) -> str:
    matches = output.split(r"<|im_start|>")
    matches = [content.replace("<|im_end|>", "") for content in matches if len(content.replace("<|im_end|>", "")) >0]
    #matches = re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", output, re.DOTALL)

    parsed_data = {}
    for match in matches:
        lines = match.strip().split("\n", 1)
        if len(lines) == 2:
            role, message = lines
        else:
            role, message = lines[0], ""
        parsed_data[role.strip()] =message.strip()

    if self.return_full_history:
      return parsed_data
    elif "assistant" in parsed_data:
      return parsed_data["assistant"]
    else:
      print([k+ v for k,v in parsed_data.items()], "\n")