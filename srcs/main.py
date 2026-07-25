from srcs.parser import parse_args, validate_args
import json
from srcs.LLMTranslator import LLMTranslator


def main():
    args = parse_args()
    try:
        data = validate_args(args)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'Error: {e}')
    except Exception as e:
        print(e)

    translator = LLMTranslator('cpu')

    func_name_list = [f.name for f in data['functions']]
    func_tokens = {f: translator.llm.encode(f).tolist()[0]
                   for f in func_name_list}

    for prompt in data['prompts']:
        translator.request_func_name(prompt.prompt,
                                     data['functions'],
                                     func_tokens)


if __name__ == "__main__":
    main()
