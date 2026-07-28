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

    func_info = '\n\n'.join(f.info_message()
                            for f in data['functions'])

    trie = TokenTrie()
    for func in data['functions']:
        tokens = translator.llm.encode(func.name).tolist()[0]
        trie.insert(func.name, tokens)

    trie.insert('None', translator.llm.encode('None').tolist()[0])

    for prompt in data['prompts']:
        func_name = translator.request_func_name(prompt.prompt,
                                                 func_info,
                                                 trie)

        func = FunctionDefinition.get_func(data['functions'], func_name)
        if not func:
            print('Not a func.')
            break

        translator.request_parameters(prompt.prompt, func)


if __name__ == "__main__":
    main()
