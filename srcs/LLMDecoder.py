from llm_sdk import Small_LLM_Model
import json


class LLMDecoder():
    def __init__(self, model_name: str) -> None:
        self.llm = Small_LLM_Model(model_name=model_name)
        self.bytes = self._bytelevel()
        self.merges = {}
        with open(self.llm.get_path_to_merges_file(),
                  'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.merges[tuple(line.split(' ', maxsplit=2))] = i

        with open(self.llm.get_path_to_vocab_file(),
                  'r', encoding='utf-8') as f:
            self.vocab = dict(json.load(f))
            self.rev_vocab = {x: y for y, x in self.vocab.items()}


    def run_prompts(self, prompts: list,
                    funcs: dict, prefix: str) -> None:
        infos = '\n\n'.join([f.get_func_info() for f in funcs.values()])
        full_prompt = (prefix + infos +
                       '\n<|im_end|>\n\n<|im_start|>user\nprompt: ')
        prompt_tokens = self.llm.encode(full_prompt).flatten().tolist()
        for prompt in prompts:
            prompt = prompt.prompt
            self.decode_prompt(prompt, prompt_tokens, funcs)

    def _bytelevel(self) -> dict:
        """
        Converts non-printable characters into printable
        ones that can be read by the LLM.
        """
        printable = (
            list(range(ord('!'), ord('~') + 1)) +
            list(range(ord('¡'), ord('¬') + 1)) +
            list(range(ord('®'), ord('ÿ') + 1))
        )

        non_print = printable[:]

        for byte in range(256):
            # non printable characters will be assigned
            # to special characters above 255
            if byte not in printable:
                printable.append(byte)
                non_print.append(256 + byte)

        return dict(zip(printable, map(chr, non_print)))

    def _bpe(self, input: str) -> None:
        new_in = ''
        for c in input:
            new_in += self.bytes.get(ord(c), '')
        print(new_in)

    def decode_prompt(self, prompt: str, prefix_ids: list[int], funcs: dict) -> None:
        llm = self.llm
        ids = prefix_ids.copy()
        rest = (f'"{prompt}"\n<|im_end|>\n\n'
                            '<|im_start|>assistant\n'
                            'result:\n'
                            '{')
        
        ids.extend(llm.encode(rest).flatten().tolist())
        
        