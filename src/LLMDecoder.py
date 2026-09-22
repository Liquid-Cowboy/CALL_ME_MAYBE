from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
import json
from src.Trie import Trie
from typing import Any
from src.parsing.FuncDef import FuncDef, FuncParam
from src.parsing.Prompt import Prompt


class LLMDecoder():
    def __init__(self, model_name: str) -> None:
        """
        Instantiates the decoder.

        Parameters:
            model_name: name of LLM model to be used

        Atributes:
            self.llm: instance of the llm obj
            self.byte_unicode: dictionary with string to unicode pairings
            self.unicode_byte: dictionary with unicode to string pairings
            self.can_decode: True if vocabulary was found and processed,
            else False
        """
        self.llm = Small_LLM_Model(model_name=model_name)
        self.byte_unicode = self._to_unicode()
        self.unicode_byte = {v: k for k, v in self.byte_unicode.items()}
        self.can_decode = False

        try:
            with open(self.llm.get_path_to_vocab_file(),
                      'r', encoding='utf-8') as f:
                self.vocab = dict(json.load(f))
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.can_decode = True
        except Exception:
            print('Couldn\'t load vocabulary. Falling back to LLM decoding...')

    @staticmethod
    def _to_unicode() -> dict[str, str]:
        """
        Translates non-printable characters to unicode representations.

        Returns: dictionary with string to unicode pairings
        """
        printable = (
                    list(range(ord("!"), ord("~") + 1))
                    + list(range(ord("¡"), ord("¬") + 1))
                    + list(range(ord("®"), ord("ÿ") + 1))
                    )

        all_bytes = printable[:]

        n = 0
        for byte in range(256):
            if byte not in printable:
                printable.append(byte)
                all_bytes.append(256 + n)
                n += 1

        return dict(zip(map(chr, printable), map(chr, all_bytes)))

    def _decode(self, tokens: list[int]) -> str:
        """
        Utility function to decode token ids.

        Parameters:
            tokens: list of ids to be decoded

        Returns: the decoded string
        """
        decoded = ''
        if self.can_decode:
            for t in tokens:
                unicode = self.inv_vocab.get(t, '')
                decoded += ''.join([str(self.unicode_byte.get(c, ''))
                                    for c in unicode])
        else:
            decoded = self.llm.decode(tokens)
        return decoded

    def run_prompts(self, prompts: list[Prompt],
                    funcs: dict[str, FuncDef],
                    prefix: str) -> list[dict[str, Any]]:
        """
        Runs a list of prompts through the LLM. It starts by
        encoding the constant/ unchanging first part of the prompt,
        corresponding to the description of the LLM's task and what
        functions are available for it to choose from. Since this
        part is shared by all prompts, we encode it only once, leaving
        the remainder to be processed at each prompt resolution.

        Parameters:
            prompts: list of prompt objs
            funcs: dictionary pairing function names with their objs
            prefix: first part of the prompt

        Returns: a list of outputs in dictionary form for json dumping
        """
        encode = self.llm.encode

        infos = '\n\n'.join([f.get_func_info() for f in funcs.values()])

        full_prompt = (prefix + infos +
                       '\n<|im_end|>\n\n')

        prompt_tokens = encode(full_prompt).tolist()[0]

        func_name_tokens = {n: encode(n).tolist()[0] for n in funcs.keys()}

        res = []

        for prompt in prompts:
            p = prompt.prompt
            res.append(self.resolve_prompt(p, prompt_tokens,
                                           funcs, func_name_tokens))
        return res

    def resolve_prompt(self, prompt: str,
                       prefix_ids: list[int],
                       funcs: dict[str, FuncDef],
                       func_name_tokens: dict[str,
                                              list[int]]) -> dict[str, Any]:
        """
        Attempts to solve a single prompt.

        Parameters:
            prompt: string with the given prompt.

            prefix_ids: generated token ids of the first part of the
            prompt, which is shared by all prompts.

            funcs: dictionary pairing the available functions
            and their names.

            func_name_tokens: dictionary pairing function names
            and their encoded counterpart.

        Returns: JSON compliant dictionary.
        """

        encode = self.llm.encode

        ids = prefix_ids.copy()
        prefix = ('<|im_start|>assistant\n'
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

        if not params and funcs.get(
            func_name, FuncDef(name='', description='',
                               parameters={},
                               returns=FuncParam(type=''))).parameters:
            return {}

        for p in params.values():
            if p == '':
                return {}

        return {
            'prompt': prompt,
            'name': func_name,
            'parameters': params
        }

    def decode_func_name(self,
                         gen_tokens: list[int],
                         name_tokens: dict[str,
                                           list[int]]) -> dict[str, Any]:
        """
        Generates a function name given a prompt and a list of available
        functions.

        Parameters:
            gen_tokens: Prompt tokens generated so far.
            name_tokens: dictionary pairing function names with their
            encoded versions.

        Returns: a dictionary with the function name and it's token ids
        """
        llm = self.llm

        trie = Trie(list(name_tokens.values()))
        generated: list[int] = []
        print('"', end='', flush=True)
        decoded = ''

        while trie.get_children():
            available = trie.get_children()
            logits = llm.get_logits_from_input_ids(gen_tokens + generated)

            best = max(available, key=lambda t: logits[t])
            trie.move_up(best)
            generated.append(best)

            decoded_token = self._decode([best])
            decoded += decoded_token
            print(decoded_token, end='', flush=True)

        print('"', end='', flush=True)
        return {
            'name': decoded,
            'tokens': generated,
        }

    def decode_parameters(self, ids: list[int],
                          max_tokens: int = 100) -> str:
        """
        Generates a string with the parameters the assigned
        function is to receive, given the submited prompt.

        Parameters:
            ids: list of token ids encoding the prompt and part
            of the generated answer.
            max_tokens: maximum amount of tokens allowed to be generated.

        Returns: the generated string.
        """

        end = '}\n'
        decoded_str = ''

        generated: list[int] = []

        while end not in decoded_str and max_tokens:
            logits = self.llm.get_logits_from_input_ids(ids + generated)
            best = logits.index(max(logits))
            generated.append(best)
            max_tokens -= 1
            decoded = self._decode([best])
            decoded_str += decoded

            print(decoded, end='', flush=True)
        return decoded_str

    def extract_parameters(self, decoded_str: str,
                           func: FuncDef | None) -> dict[str, Any]:
        """
        Extracts parameters of generated string, in order to build
        JSON compliant dictionary.

        Parameters:
            decoded_str: parameter text generated by the LLM.
            func: the assigned function to this one prompt.

        Returns: a dictionary matching parameter names with
        said parameters.
        """
        try:
            last_bracket = len(decoded_str) - 1 - decoded_str[::-1].index('}')
        except ValueError:
            return {}

        decoded_str = decoded_str[:last_bracket]
        decoded_parts = [s.strip() for s in decoded_str.split(', "')]

        res: dict[str, int | float | bool | str] = {}
        if not func:
            return {}

        for name, p_type in func.parameters.items():
            t = p_type.type
            for part in decoded_parts:
                if name not in part:
                    continue

                value = part.split(':', maxsplit=1)[1].strip(' "\'')
                value = value.replace('\\\\', '\\')

                try:
                    match t:

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
                    return {}

        return res
