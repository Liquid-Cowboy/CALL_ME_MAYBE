from llm_sdk import Small_LLM_Model
from srcs.Trie import TokenTrie
from srcs.FunctionDefinition import FunctionDefinition
import re


class LLMTranslator:
    def __init__(self, device: str | None = None):
        self.llm = Small_LLM_Model(device=device)

    def request_func_name(self, prompt: str, func_info: str,
                          trie: TokenTrie) -> str:
        llm = self.llm

        func_request = ('<|system|>\n'
                        'You are a function selector. '
                        'Given a user prompt and a list of functions, '
                        'choose the best function.'
                        '\n\nRules:\n'
                        '- Only choose from the list of available '
                        'functions or "None".\n'
                        '- Only choose "None" if none of the available '
                        'functions can reasonably satisfy the prompt\'s'
                        'requirements.\n'
                        '- Do NOT invent functions.\n'
                        '- Output ONLY the function\'s name or "None".\n'
                        '\n\n<|assistant|>\n'
                        f'Available functions: \n{func_info}\n\n'
                        '<|user|>\n'
                        f'Prompt: "{prompt}".\n'
                        '<|assistant|>\nSelected function: ')

        request_tokens = llm.encode(func_request).tolist()[0]

        generated = []

        node = trie.root

        while trie.name_found(node) is None:

            viable_tokens = trie.get_children(node)
            logits = llm.get_logits_from_input_ids(request_tokens + generated)

            # for token in viable_tokens:
            #     print(
            #             repr(llm.decode([token])),
            #             float(logits[token])
            #         )
            best = max(viable_tokens, key=lambda t: logits[t])

            generated.append(best)
            node = trie.update(node, best)
        return llm.decode(generated)

    def request_parameters(self, prompt: str, func: FunctionDefinition):
        if not func.parameters:
            return None

        viable_params = {'number': {}, 'float': {},
                         'int': {}, 'string': {}}
        
        output = {}
        for name, value in func.parameters.items():
            par_type = value.type
            match par_type:
                case 'number':
                    self.extract_number(prompt, 'number', name,
                                        func, viable_params)
                    print(viable_params['number']['strs'])
                case 'float':
                    self.extract_number(prompt, 'float', name,
                                        func, viable_params)
                case 'int':
                    self.extract_number(prompt, 'int', name,
                                        func, viable_params)
                case 'boolean':
                    pass
                case 'string':
                    pass
                    # extract_string(prompt, viable_params)

    def extract_number(self, prompt: str, type: str, name: str,
                       func: FunctionDefinition, candidates: dict):
        llm = self.llm

        if candidates.get('number'):
            matches = candidates.get('number', {})
        else:
            matches = re.findall(r'-?(?:\d+\.\d+|\d+)', prompt)
            if not matches:
                return None
            candidates['number']['strs'] = matches
            candidates['number']['tokens'] = [llm.encode(n).tolist()[0]
                                              for n in matches]
        match type:
            case 'number':
                pass

            case 'int':
                matches = [m for m in matches if '.' not in m]
                candidates['int']['strs'] = matches
                candidates['int']['tokens'] = [llm.encode(n).tolist()[0]
                                                for n in matches]

            case 'float':
                matches = [m for m in matches if '.' in m]
                candidates['float']['strs'] = matches
                candidates['float']['tokens'] = [llm.encode(n).tolist()[0]
                                                    for n in matches]

        base_prompt = ('<|system|>\n'
                       'You are a number selector.\n\n'
                       'Given:\n'
                       '- the function info,\n'
                       '- a user prompt,\n'
                       '- a parameter name,\n'
                       '- a list of available numbers,\n'
                       'select the number that best matches the '
                       'requested parameter.\n\n'
                       'Rules:\n'
                       '- Output exactly one value.\n'
                       '- The output must be one of the available '
                       'numbers or "None".\n'
                       '- Do NOT output anything else.\n'
                       '- Only output "None" if no available number matches \n'
                       'the requested parameter.\n\n'
                       'Use:\n'
                       '1. the user prompt,\n'
                       '2. the function description,\n'
                       '3. the parameter name.\n\n'
                       'Keywords like "sum", "subtract", "multiply", '
                       '"divide" or "square root" tend to appear fairly '
                       'close to the matching numbers.\n\n'
                       'Examples:\n\n'
                       'Prompt: "Return the sum of 4 and 3.32."\n'
                       'Result: 4 | 3.32\n\n'
                       'Prompt: "Calculate the square root of 25."\n'
                       'Result: 25\n\n'
                       '<|assistant|>\n\n'
                       f'Function info:\n{func.info_message()}\n\n'
                       f'Available numbers:\n{candidates[type]["strs"]}\n\n'
                       f'Requested parameter: "{name}"\n\n'
                       '<|user|>\n'
                       f'Prompt: "{prompt}".\n\n'
                       '<|assistant|>\nSelected parameter: ')

        prompt_tokens = llm.encode(base_prompt).tolist()[0]
        trie = TokenTrie()

        strings: list = candidates[type]['strs']
        tokens: list = candidates[type]['tokens']

        print(strings)
        print(tokens)

        if not strings:
            return None

        for i in range(len(strings)):
            trie.insert(strings[i], tokens[i])

        node = trie.root
        generated = []

        while not trie.name_found(node):
            viable_tokens = trie.get_children(node)
            print(f'Viable tokens: {viable_tokens}')
            logits = llm.get_logits_from_input_ids(prompt_tokens + generated)
            best = max(viable_tokens, key=lambda t: logits[t])
            print(f'Best: {best}')
            node = trie.update(node, best)
            generated.append(best)

        output = strings.pop(strings.index(trie.name_found(node)))
        tokens.pop(tokens.index(generated))
        print(strings)
        print(tokens)

        print(output)

        return output
