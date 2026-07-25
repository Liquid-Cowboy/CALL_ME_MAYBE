from pydantic import BaseModel, model_validator
from typing import Any


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode='before')
    @classmethod
    def validate(cls, args: Any) -> Any:
        if not isinstance(args, dict):
            raise TypeError(f'Expected a dict, got an {type(args).__name__}.')
        if ['prompt'] != list(args.keys()):
            raise ValueError('Expected "prompt" as the only key.')
        return args
