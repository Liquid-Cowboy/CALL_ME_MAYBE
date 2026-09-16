import pytest

from src.LLMDecoder import LLMDecoder
from src.constants import PROMPT_PREFIX
from src.parsing.parser import Parser
from src.parsing.Prompt import Prompt


def run_test(prompt: str) -> list[dict]:
    """
    Runs a single test with the given prompt.

    Parameters:
        prompt: prompt to be ran

    Returns: list of outputs in dictionary form
    """

    parser = Parser([])
    decoder = LLMDecoder(parser.model_name)
    output = decoder.run_prompts([Prompt(prompt=prompt)],
                                 parser.func_defs, PROMPT_PREFIX)
    return output


def test_empty() -> None:
    """Tests an empty prompt case."""
    print('\n\n')
    parser = Parser([])
    decoder = LLMDecoder(parser.model_name)

    with pytest.raises(Exception):

        decoder.run_prompts(
            [Prompt(prompt='')],
            parser.func_defs,
            PROMPT_PREFIX)
        print('\n\n')


def test_large_n() -> None:
    """Tests a large number case."""
    print('\n\n')
    output = run_test('Mulitply 3984123543678245921345238945 '
                      'by 34294756219384523198.')
    print('\n\n')

    assert output != [{}]


def test_ambiguous() -> None:
    """Tests an amibuous prompt case."""
    print('\n\n')
    output = run_test('what?.')
    print('\n\n')

    assert output != [{}]
