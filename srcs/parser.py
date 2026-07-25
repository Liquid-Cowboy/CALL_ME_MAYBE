import argparse
from typing import Any
import json
from srcs.Prompt import Prompt
from srcs.FunctionDefinition import FunctionDefinition
from pydantic import ValidationError


def parse_args() -> dict[str, Any]:
    """Defines valid arguments to be parsed by using
    argparse.ArgumentParser()

    Returns:
        Returns a dictionary with paths to the input,
        function definitions and output files.
    """

    parser = argparse.ArgumentParser(
        description='Function calling pipeline CLI'
    )

    parser.add_argument(
        '--input',
        help='Path to the input prompts JSON file.',
        default='data/input/function_calling_tests.json'
    )

    parser.add_argument(
        '--output',
        help='Path to the output JSON file.',
        default='data/output/function_calling_tests.json'
    )

    parser.add_argument(
        '--functions_definition',
        help='Path to the function definitions JSON file.',
        default='data/input/functions_definition.json'
    )

    args = parser.parse_args()
    return {
        'input': args.input,
        'output': args.output,
        'functions': args.functions_definition
    }


def validate_args(args: dict[str, str]) -> dict[str, Any]:
    """Validates input and function definitions by using
    Pydantic BaseModels. Raises a comprehensive log of all
    errors, if any are detected.

    Parameters:
        args: dictionary with input and function definitions'
        path in order to load the proper JSONs.

    Returns: dictionary with 'prompts' and 'functions' keys,
    representing lists of validated objects.
    """
    prompts = []
    prompts_e = {}
    f_definitions = []
    f_e = {}
    for i, p in enumerate(load_json(args['input'])):
        try:
            prompts.append(Prompt.model_validate(p))
        except ValidationError as e:
            log = 'Validation error:\n'
            for er in e.errors():
                location = '.'.join(map(str, er['loc']))
                log += f'\t\t{location}: {er["msg"]}.\n'
            prompts_e[i + 1] = '\t' + log.strip()
        except (TypeError, ValueError) as e:
            prompts_e[i + 1] = '\t' + str(e)

    for i, f in enumerate(load_json(args['functions'])):
        try:
            f_definitions.append(FunctionDefinition.model_validate(f))
        except ValidationError as e:
            log = 'Validation error:\n'
            for er in e.errors():
                location = '.'.join(map(str, er['loc']))
                log += f'\t\t{location}: {er["msg"]}.\n'
            f_e[i + 1] = '\t' + log.strip()

        except (TypeError, ValueError) as e:
            f_e[i + 1] = '\t' + str(e)

    log = ''
    if prompts_e:
        log += 'Input JSON errors:\n'
        for i in prompts_e.keys():
            log += f'In prompt {i}:\n'
            log += prompts_e[i] + '\n'

    if f_e:
        log += '\n\nFunction definition errors:\n'
        for i in f_e.keys():
            log += f'In function definition {i}:\n'
            log += f_e[i] + '\n'

    if prompts_e or f_e:
        raise Exception(log.strip())

    return {
        'prompts': prompts,
        'functions': f_definitions,
    }


def load_json(json_path: str) -> Any:
    """Loads json file given the proper path
    and re-formats the messages from the errors
    that might arise.

    Parameters:
        json_path: str of the appointed json file

    Returns:
        Returns the native value of json.load()
        """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at "{json_path}".')

    except json.JSONDecodeError as e:
        raise TypeError(f'Invalid JSON format at "{json_path}" - {e}.')
