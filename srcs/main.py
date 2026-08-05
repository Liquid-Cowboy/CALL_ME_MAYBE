from srcs.parser import parse_args, validate_args
import json
from srcs.LLMTranslator import LLMTranslator
from srcs.Trie import TokenTrie
from srcs.FunctionDefinition import FunctionDefinition
from srcs.dump_json import dump_json


def main():
    args = parse_args()

    data = {}
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

    functions = data.get('functions', [])
    prompts = [p.prompt for p in data.get('prompts', [])]

    func_trie = TokenTrie()
    for func in functions:
        tokens = translator.llm.encode(func.name).tolist()[0]
        func_trie.insert(func.name, tokens)

    func_trie.insert('None', translator.llm.encode('None').tolist()[0])

    entry = []

    for prompt in prompts:
        if not prompt.strip():
            continue
        func_name = translator.request_func_name(prompt,
                                                 func_info,
                                                 func_trie)

        func = FunctionDefinition.get_func(functions, func_name)
        if not func:
            print('Not a func.')
            break

        print(f'Prompt: {prompt}.')
        print(f'Function name: {func.name}')

        params = translator.request_parameters(prompt, func)
        res = {
            'prompt': prompt,
            'name': func.name,
            'parameters': params,
        }
        entry.append(res)
    dump_json(entry, args.get('output'))


if __name__ == "__main__":
    main()
