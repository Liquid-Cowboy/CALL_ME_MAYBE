from srcs.parsing.parser import Parser, ParsingError, JSONError
from srcs.LLMDecoder import LLMDecoder
import json
from pathlib import Path

PROMPT_PREFIX = ('<|im_start|>system\n'
                 'You are a function calling JSON compliant assistant. '
                 'You must take a prompt from a user and produce a valid '
                 'JSON following the provided template.\n\n'
                 'TEMPLATE:\n'
                 '{\n'
                 '  "prompt": "<prompt>",\n'
                 '  "name": "<function name>",\n'
                 '  "parameters": {"<parameter name>": <parameter value>, ...}'
                 '\n}\n\n'
                 'Available functions:\n')


def main():
    try:
        print('Parsing start...')
        parser = Parser()
        print('Parsing complete.')

        decoder = LLMDecoder(parser.model_name)
        output = decoder.run_prompts(parser.prompts,
                                     parser.func_defs,
                                     PROMPT_PREFIX)

        decoder._bpe('Hello World, my name\'s Marcos.')

        output_path = Path(parser.output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(parser.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)

    except ParsingError as e:
        print(f'--PARSING ERROR--\n{e}')
    except JSONError as e:
        print(f'--JSON ERROR--\n\n{e}')


if __name__ == "__main__":
    main()
