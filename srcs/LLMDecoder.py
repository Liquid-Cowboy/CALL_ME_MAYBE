from llm_sdk import Small_LLM_Model


class LLMDecoder():
    def __init__(self, model_name: str) -> None:
        self.llm = Small_LLM_Model(model_name=model_name)

    def run_prompts(self, prompts: list,
                    funcs: dict, prefix: str) -> None:
        infos = '\n\n'.join([f.get_func_info() for f in funcs.values()])
        full_prompt = prefix + infos + f'\n<|im_end|>\n\n<|im_start|>user\nprompt: ' 
        prompt_tokens = self.llm.encode(full_prompt).flatten().tolist()
        for prompt in prompts:
            prompt = prompt.prompt
            self.decode_prompt(prompt, prompt_tokens, funcs)
            


    def decode_prompt(self, prompt: str, prefix_ids: list[int], funcs: dict) -> None:
        llm = self.llm
        ids = prefix_ids.copy()
        rest = (f'"{prompt}"\n<|im_end|>\n\n'
                            '<|im_start|>assistant\n'
                            'result:\n'
                            '{')
        
        ids.extend(llm.encode(rest).flatten().tolist())
        
        