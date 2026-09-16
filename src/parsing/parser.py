from typing import Any
from src.parsing.FuncDef import FuncDef
from src.parsing.Prompt import Prompt
from pydantic import ValidationError


class ParsingError(Exception):
    """Custom class for parsing errors."""
    pass


class JSONError(Exception):
    """Custom class for JSON related errors."""
    pass


class Parser():
    """
    Utility class to handle the parsing and validation
    of CLI arguments, function definitions and prompts.
    """
    def __init__(self, args: list[str] | None = None) -> None:
        """
        Instances the class with default atributes.

        Parameters:
            args: list of possible arguments the CLI can accept

        Atributes:
            self.func_defs: mapping of function names to FuncDefs objs
            self.prompts: list of prompt objs to be ran through

        """
        self.parser(args)
        self.func_defs: dict[str, FuncDef] = {}
        self.prompts: list[Prompt] = []
        self.validator()

    def parser(self, raw_args: list[str] | None = None) -> None:
        """
        Parses the CLI arguments using argparse.

        Arguments:
            raw_args: list of arguments to be parsed
        """

        import argparse as ap

        parser = ap.ArgumentParser(
            prog='Call Me Maybe',
            description=('Function caller based on constrained '
                         'decoding of an LLM\'s output.')
        )

        parser.add_argument(
            '--functions_definition',
            help=('Path to the JSON file with the '
                  'available function definitions.'),
            default='data/input/functions_definition.json'
        )

        parser.add_argument(
            '--input',
            help='Path to the input prompter JSON file.',
            default='data/input/function_calling_tests.json',
        )

        parser.add_argument(
            '--output',
            help='Path to the output JSON file.',
            default='data/output/function_calling_results.json'
        )

        parser.add_argument(
            '--model',
            help='Choosen LLM model to run the function caller.',
            default='Qwen/Qwen3-0.6B'
        )

        args = parser.parse_args(args=raw_args)
        self.func_def_path = args.functions_definition
        self.input_path = args.input
        self.output_path = args.output
        self.model_name = args.model

    def validator(self) -> None:
        """Validates every function definition and prompt."""

        errors = [(f'Input - <{self.input_path}>\n'
                   f'Functions Definition - <{self.func_def_path}>')]

        for i, obj in enumerate(self.load_json(self.input_path), start=1):

            try:
                self.prompts.append(Prompt(**obj))

            except ValidationError as e:
                for er in e.errors():
                    loc = '.'.join(map(str, er.get('loc', [])))
                    loc = '(' + loc + ')' if loc else ''
                    errors.append(f'In Prompt {i} - '
                                  f'{er.get("msg", "")} {loc}')

            except TypeError:
                log = (f'Prompt {i} must be a '
                       f'mapping, not {type(obj).__name__}')
                errors.append(log)

        for i, obj in enumerate(self.load_json(self.func_def_path), start=1):

            try:
                func = FuncDef.model_validate(obj,
                                              context=self.func_defs.values())
                self.func_defs.update({func.name: func})
            except ValidationError as e:
                logs = []
                for er in e.errors():
                    loc = '.'.join(map(str, er.get('loc', [])))
                    loc = '(' + loc + ')' if loc else ''
                    msg = f'\t- {er.get("msg", "")} {loc}'
                    msg += '\n\tYour input: '
                    input = er.get('input')
                    msg += f'{input}'
                    logs.append(msg)
                finished_log = f'In Function Definition {i}:'
                finished_log += '\n' + '\n'.join(logs)
                errors.append(finished_log)

            except TypeError:
                log = (f'Function definition {i} must be a '
                       f'mapping, not {type(obj).__name__}')
                errors.append(log)

            except ValueError as e:
                print(e)
                errors.append(f'In Function Definition {i}: {e}')

        if len(errors) > 1:
            raise ParsingError('\n\n'.join(errors))

    @staticmethod
    def load_json(path: str) -> Any:
        """
        Utility function to load JSONs and raise
        apropriate errors.

        Parameters:
            path: path to JSON file

        Returns: Any JSON compatible obj
        """
        import json

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        except FileNotFoundError:
            raise JSONError(f'File not found at "{path}".')

        except json.JSONDecodeError as e:
            raise JSONError(f'Invalid JSON format at "{path}" - {e}.')
