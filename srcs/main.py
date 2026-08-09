from srcs.parsing.parser import Parser, ParsingError, JSONError
from srcs.LLMDecoder import LLMDecoder

PROMPT_PREFIX = ('<|im_start|>system\n'
               'You are a function calling JSON compliant assistant. '
               'You must take a prompt from a user and produce a valid '
               'JSON following the provided template.\n\n'
               'TEMPLATE:\n'
               '{\n'
               '  "prompt": "<prompt>",\n'
               '  "name": "<function name>",\n'
               '  "parameters": {"<parameter name>": <parameter value>, ...}\n'
               '}\n\n'
               'Available functions:\n')


def main():
    try:
        

        print('Parsing start...')
        parser = Parser()
        print('Parsing complete.')

        decoder = LLMDecoder(parser.model_name)
        import json
        with open(decoder.llm.get_path_to_vocab_file(), 'r',encoding='utf-8') as f:
            print(json.load(f))

        decoder.run_prompts(parser.prompts,
                            parser.func_defs,
                            PROMPT_PREFIX)

    except ParsingError as e:
        print(f'--PARSING ERROR--\n{e}')
    except JSONError as e:
        print(f'--JSON ERROR--\n\n{e}')


if __name__ == "__main__":
    main()
