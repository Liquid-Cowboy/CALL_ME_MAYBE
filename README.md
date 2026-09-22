*This project has been created as part of the 42 curriculum by mnogueir.*


# Call Me Maybe

## Description

**Call Me Maybe** is a function-calling tool: it turns a natural-language prompt (e.g. *"What is the sum of 2 and 3?"*) into a structured, machine-executable function call — a function `name` plus typed `parameters` — instead of a plain-text answer.

The core constraint of the project is that this structured output must be **guaranteed valid**, even though the underlying model (`Qwen/Qwen3-0.6B` by default) is a 0.6B-parameter model that is, on its own, unreliable at producing well-formed JSON. Reliability is achieved through **constrained decoding**: at generation time, the set of tokens the model is allowed to pick from is restricted so that the output cannot drift outside what is structurally and semantically valid, rather than hoping the model "gets it right" from prompting alone.

Given:
- a `functions_definition.json` file describing the available functions (name, parameter names/types, return type, description) and
- a `function_calling_tests.json` file with a list of natural-language prompts,

the program produces a JSON array where each entry contains the original `prompt`, the selected function `name`, and its `parameters` with correctly typed values.

## Instructions

**Requirements:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/).

**Repository layout expected at runtime:**
```
.
├── src/                # implementation
├── llm_sdk/             # provided SDK (workspace member, copied alongside src/)
├── data/
│   ├── input/            # functions_definition.json, function_calling_tests.json
│   └── output/           # generated results (not versioned)
├── pyproject.toml
└── uv.lock
```

**Install dependencies:**
```sh
uv sync
```
This installs `numpy`, `pydantic`, and the local `llm_sdk` workspace package (declared in `pyproject.toml`) — no `pip`/`venv` step is needed beyond this, since `llm_sdk` is registered as a `[tool.uv.workspace]` member.

**Run the program:**
```sh
uv run python -m src \
  [--functions_definition <function_definition_file>] \
  [--input <input_file>] \
  [--output <output_file>] \
  [--model <model_name>]
```

Defaults (from `src/parsing/parser.py`):
| Flag | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |
| `--model` | `Qwen/Qwen3-0.6B` |


**Other Makefile targets:**
```sh
make install    # install dependencies
make run        # uv run python -m main.py
make debug      # run under pdb
make clean      # remove __pycache__ / .mypy_cache
make lint       # flake8 . && mypy . (project's required flags)
make lint-strict # flake8 . && mypy . --strict
make tests      # executes test suite
make clean_output # cleans all caches and output files
```

## Resources

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — general background on the function-calling paradigm this project implements from scratch.
- [Hugging Face — Guiding text generation with constrained decoding](https://huggingface.co/blog) (`LogitsProcessor` docs) — conceptual reference for restricting a model's next-token distribution.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model used in this project.
- Byte-level BPE tokenization (GPT-2-style byte↔unicode mapping) — background for the manual vocabulary decoding implemented in `LLMDecoder._to_unicode`.
- [Trie (prefix tree) data structure](https://en.wikipedia.org/wiki/Trie) — background for the function-name constrained-decoding structure in `src/Trie.py`.
- Figuring out tokenization: [TOKENIZATION: How AI models turn text into numbers | Byte-Pair Encoding](https://www.youtube.com/watch?v=4A_nfXyBD08) *by Annie Sexton.*

**AI usage:** AI was used mainly to get to more practical design decisions such as the trie structure, to figure how non-printable characteres were assigned to printable ones during the pre-tokenization process and to help write out this README.

## Algorithm explanation

Constrained decoding is applied in two different ways depending on which part of the output is being produced:

1. **JSON envelope (not generated).** The surrounding structure — braces, the `"prompt"`, `"name"`, `"parameters"` keys, and separators — is written directly by `LLMDecoder.resolve_prompt` as plain Python string concatenation, then re-tokenized and fed back to the model as context. The model never has the opportunity to break this part of the JSON, because it never generates it.

2. **Function name — trie-constrained.** `decode_func_name` builds a `Trie` (`src/Trie.py`) from the tokenized form of every function name in `functions_definition.json`. At each step it asks the SDK for logits over the *entire* vocabulary (`get_logits_from_input_ids`), but only considers tokens that are valid children of the trie node representing "what has been generated of the function name so far" — i.e. every token outside that legal set is effectively excluded before the argmax is taken, which is the same "set invalid tokens to −∞" idea from the subject, implemented via trie pruning instead of literally rewriting the logits array. Generation stops once the current trie node has no children left, which happens exactly when a full, unambiguous name has been produced. **This guarantees the selected function name is always one of the names declared in `functions_definition.json`.**

3. **Parameter values — free generation + parsing.** Once the name is fixed, `decode_parameters` lets the model generate tokens greedily (plain `argmax` over the full vocabulary, no trie/schema constraint) until the decoded text contains a closing `}\n` or a 100-token budget (configurable) is exhausted. `extract_parameters` then trims the text to the last `}`, splits it on `, "`, matches each declared parameter by name, and casts its value to the type declared in `functions_definition.json` (`number`/`float` → `float`, `int`/`integer` → `int`, `boolean` → `bool`, anything else → `str`), falling back to the raw string if casting fails.

4. **Tokenization without the SDK's `decode()`.** `LLMDecoder` reads the tokenizer's vocabulary file directly via `get_path_to_vocab_file()` and builds its own `token id → string` map, plus a byte-level unicode remapping table (`_to_unicode`, mirroring the standard GPT-2/BPE byte↔unicode scheme) so that generated token ids can be turned back into text without calling any private `llm_sdk` method. If the vocab file can't be read, it transparently falls back to `llm.decode()`.

5. **Prefix caching.** The constant system prompt plus the serialized function catalogue (`FuncDef.get_func_info()` for every function) is tokenized once per program run in `run_prompts`, not once per prompt, since it is identical across all prompts.

## Design decisions

- **Pydantic-first validation.** `FuncDef`, `FuncParam`, and `Prompt` (in `src/parsing/`) validate both input files structurally *before* any LLM call is made. `FuncDef` also rejects duplicate or empty function names via a context-aware `model_validator`. All validation errors across every entry are collected and raised together as a single `ParsingError`, rather than stopping at the first bad entry — giving one readable report instead of a crash loop.
- **Split constraint strategy.** Only the function name is hard-constrained (guaranteed correct by construction via the trie); parameter values are generated freely and recovered by best-effort parsing/casting. This trades a strict per-token schema guarantee on parameter values for a much simpler implementation, at the cost of parameter extraction being only as good as the model's free-form output.
- **No dependency on private/undocumented `llm_sdk` internals.** Vocabulary is read from the public `get_path_to_vocab_file()` path and decoded manually, satisfying the project's "no private methods/attributes" constraint and covering part of the bonus (public `encode`/`decode` handling, avoiding direct `decode()` use in the main path).
- **Configurable model.** `--model` lets any model name be passed to `Small_LLM_Model`, defaulting to `Qwen/Qwen3-0.6B`, partially covering the "support other models" bonus.
- **Graceful failure.** `__main__.py` catches `ParsingError` and `JSONError` specifically and prints a clear message instead of letting the program crash with a raw traceback.

## Performance analysis

Takes on average 4'30 minutes to run through the entire default prompts. It solves almost every prompt but it has some trouble with format templates.

By construction:
- **Function selection is 100% schema-valid**: the trie constraint makes it structurally impossible to emit a function name that isn't in `functions_definition.json`.
- **Parameter extraction is best-effort, not guaranteed**: since `decode_parameters` is unconstrained and capped at 100 tokens, malformed or truncated model output can result in missing/incorrectly-typed parameters; `extract_parameters` degrades gracefully (keeps the raw string) rather than crashing, but does not guarantee schema compliance the way the function-name step does.

## Challenges faced

One of the most troublesome challenges to overcome during this project was figuring out where the constrained decoding should take place and where I should let the LLM "find it's way" more freely. The implementation of the trie structure (or something similar to it) during the function name generation part was one of the first ideas I sought to implement, so as to save up on running through the logits array and setting every invalid token to −∞. During the design process, besides thinking about the functionality of each step, I kept trying to find cost effective ways of generating the expected output while keeping the LLM's workload as minimal as possible.   
Another difficult problem was figuring out how the LLM's pre-tokenization turned non-printable characters to printable ones so as to map out the vocabulary and avoid using the built-in ```decode``` function. I then figured the non-printable characters were incrementally assigned to unicode codes above the 255 threshold and wrote ```_to_unicode``` to help solve that problem.


## Testing strategy

To test the program, I mainly used the default input data, but other inputs were tried as well such as empty strings, large numbers and ambiguous or more complex prompts. The LLM still tries to find a way to answer each and every prompt, having trouble only with template driven prompts.


## Example usage

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Given a prompt such as `"What is the sum of 2 and 3?"` and a `fn_add_numbers(a: number, b: number) -> number` definition, the pipeline is expected to produce an entry shaped like:
```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```
one such object per prompt, written as a JSON array to the `--output` path.
