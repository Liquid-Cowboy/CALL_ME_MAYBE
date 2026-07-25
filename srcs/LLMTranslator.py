from llm_sdk import Small_LLM_Model


class LLMTranslator:
    def __init__(self, device: str | None = None):
        self.llm = Small_LLM_Model(device=device)

    def request_func_name(self, prompt: str, funcs: list,
                          func_tokens: dict[str, list]):
        llm = self.llm

        func_strs = '\n\n'.join(f.info_message()
                                for f in funcs)

        function_request = ('<|system|>\n'
                            'You are a function selector.'
                            'Given a user prompt and a list of functions, '
                            'choose the best function.'
                            '\n\nRules:\n'
                            '- Only choose from the list of available functions.\n'
                            '- Do NOT invent functions.\n'
                            '- Output ONLY the function\'s name.\n\n'
                            '<|assistant|>\n'
                            f'Available functions: \n{func_strs}\n\n'
                            '<|user|>\n'
                            f'Prompt: "{prompt}".\n'
                            '<|assistant|>')

        request_tokens = llm.encode(function_request).tolist()[0]

        generated = []

        while (True):
            if generated in func_tokens.values():
                break
            viable_tokens = set()
            logits = llm.get_logits_from_input_ids(request_tokens + generated)
            for token_list in func_tokens.values():
                if token_list[:len(generated)] == generated:
                    viable_tokens.add(token_list[len(generated)])
            token_score = {logits[t]: t for t in viable_tokens}
            generated.append(token_score[max(token_score.keys())])

        print(llm.decode(generated))
        # print(llm.decode(generated))


        # with open(llm.get_path_to_vocab_file(), 'r', encoding='utf-8') as f:
        #     vocab = json.load(f)

        # print(vocab)

        # logits = llm.get_logits_from_input_ids(tok_req)
        # for value in tokens.values():
            
        # # print(len(logits))
