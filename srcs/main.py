from srcs.parser import parse_args, validate_args
import json
from srcs.LLMTranslator import LLMTranslator
from srcs.Trie import TokenTrie
from srcs.FunctionDefinition import FunctionDefinition


def main():
    args = parse_args()
    try:
        data = validate_args(args)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'Error: {e}')
    except Exception as e:
        print(e)

    translator = LLMTranslator('cpu')

    # for i in list(translator.vocab.items()):
    #     print(i)

    func_info = '\n\n'.join(f.info_message()
                            for f in data['functions'])

    func_trie = TokenTrie()
    for func in data['functions']:
        tokens = translator.llm.encode(func.name).tolist()[0]
        func_trie.insert(func.name, tokens)

    func_trie.insert('None', translator.llm.encode('None').tolist()[0])

    for prompt in data['prompts']:
        if not prompt.prompt.strip():
            continue
        func_name = translator.request_func_name(prompt.prompt,
                                                 func_info,
                                                 func_trie)

        func = FunctionDefinition.get_func(data['functions'], func_name)
        if not func:
            print('Not a func.')
            break

        print(f'Prompt: {prompt.prompt}.')
        print(f'Function name: {func.name}')

        translator.request_parameters(prompt.prompt, func)


if __name__ == "__main__":
    main()
