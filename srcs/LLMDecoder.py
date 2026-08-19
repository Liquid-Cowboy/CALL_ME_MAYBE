from llm_sdk import Small_LLM_Model
import json
from srcs.Trie import Trie
from typing import Any
from srcs.parsing.FuncDef import FuncDef


class LLMDecoder():
    def __init__(self, model_name: str) -> None:
        self.llm = Small_LLM_Model(model_name=model_name)
        self.bytes = self._to_unicode()
        self.merges = {}
        with open(self.llm.get_path_to_merges_file(),
                  'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                self.merges[tuple(line.split(' ', maxsplit=2))] = i

        with open(self.llm.get_path_to_vocab_file(),
                  'r', encoding='utf-8') as f:
            self.vocab = dict(json.load(f))
            self.rev_vocab = {x: y for y, x in self.vocab.items()}

    def run_prompts(self, prompts: list,
                    funcs: dict, prefix: str) -> list[dict]:
        llm = self.llm

        infos = '\n\n'.join([f.get_func_info() for f in funcs.values()])

        full_prompt = (prefix + infos +
                       '\n<|im_end|>\n\n<|im_start|>user\nprompt: ')

        prompt_tokens = llm.encode(full_prompt)
        if hasattr(prompt_tokens, 'flatten'):
            prompt_tokens = prompt_tokens.flatten()
        prompt_tokens = prompt_tokens.tolist()

        func_name_tokens = {n: llm.encode(n).tolist()[0] for n in funcs.keys()}

        res = []

        for prompt in prompts:
            prompt = prompt.prompt
            res.append(self.decode_prompt(prompt, prompt_tokens,
                                          funcs, func_name_tokens))
        return res

    def _to_unicode(self) -> dict:
        """
        Converts non-printable characters into printable
        ones that can be read by the LLM.
        """
        printable = (
            list(range(ord('!'), ord('~') + 1)) +
            list(range(ord('¡'), ord('¬') + 1)) +
            list(range(ord('®'), ord('ÿ') + 1))
        )

        all = printable[:]

        for byte in range(256):
            # non printable characters will be assigned
            # to special characters above 255
            if byte not in printable:
                printable.append(byte)
                all.append(256 + byte)

        return dict(zip(printable, map(chr, all)))

    def _bpe(self, input: str) -> None:
        new_in = []
        for c in input:
            new_in.append(self.bytes.get(ord(c), ''))

    def decode_prompt(self, prompt: str,
                      prefix_ids: list[int],
                      funcs: dict,
                      func_name_tokens: dict[str, list[int]]) -> dict:

        encode = self.llm.encode

        ids = prefix_ids.copy()
        prefix = (f'"{prompt}"\n<|im_end|>\n\n'
                  '<|im_start|>assistant\n'
                  'result:\n')

        res = ('{\n'
               '  "prompt": "' + prompt + '",\n'
               '  "name": ')
        print(res, end='', flush=True)
        prefix += res

        added_tokens = encode(prefix)
        if hasattr(added_tokens, 'flatten'):
            added_tokens = added_tokens.flatten()

        ids.extend(added_tokens.tolist())

        func_res = self.decode_func_name(ids, func_name_tokens)
        func_name = func_res.get('name', '')

        separator = ',\n  "parameters": {'
        print(separator, end='', flush=True)
        ids.extend(encode('"' + func_name + '"' + separator).tolist()[0])
        param_string = self.decode_parameters(ids)
        print('}\n', end='', flush=True)
        params = self.extract_parameters(param_string, funcs.get(func_name))
        return {
            'prompt': prompt,
            'name': func_name,
            'parameters': params
        }

    def decode_func_name(self,
                         gen_tokens: list[int],
                         name_tokens: dict[str,
                                           list[int]]) -> dict[str, Any]:
        llm = self.llm

        trie = Trie(list(name_tokens.values()))
        generated = []
        print('"', end='', flush=True)

        while trie.get_children():
            available = trie.get_children()
            logits = llm.get_logits_from_input_ids(gen_tokens + generated)
            best = max(available, key=lambda t: logits[t])
            trie.move_up(best)
            generated.append(best)
            print(llm.decode([best]), end='', flush=True)
        print('"', end='', flush=True)
        return {
            'name': llm.decode(generated),
            'tokens': generated,
        }

    def decode_parameters(self, ids: list[int],
                          max_tokens: int = 100) -> str:

        end = '}\n'
        decoded_str = ''

        generated = []

        while end not in decoded_str and max_tokens:
            logits = self.llm.get_logits_from_input_ids(ids + generated)
            best = logits.index(max(logits))
            generated.append(best)
            max_tokens -= 1
            decoded = self.llm.decode([best])
            decoded_str += decoded

            print(decoded, end='', flush=True)
        return decoded_str

    def extract_parameters(self, decoded_str: str,
                           func: FuncDef | None) -> dict[str, Any]:

        last_i = len(decoded_str) - 1 - decoded_str[::-1].index('}')

        decoded_str = decoded_str[:last_i]
        decoded_parts = [s.strip() for s in decoded_str.split(', "')]

        res = {}
        if not func:
            return {}

        for name, p_type in func.parameters.items():
            p_type = p_type.type
            for part in decoded_parts:
                if name not in part:
                    continue

                value = part.split(':', maxsplit=1)[1].strip(' "\'')

                try:
                    match p_type:

                        case 'number' | 'float':
                            res[name] = float(value)

                        case 'int' | 'integer':
                            res[name] = int(value)

                        case 'boolean':
                            res[name] = (True if value.lower() == 'true'
                                         else False)

                        case _:
                            res[name] = value

                except ValueError:
                    res[name] = value

        return res
