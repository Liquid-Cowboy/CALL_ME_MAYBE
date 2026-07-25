from pydantic import BaseModel, model_validator
from typing import Any


class FunctionParameter(BaseModel):
    """Parameter to be taken in by the function"""
    type: str


class FunctionDefinition(BaseModel):
    """Function validation class with a comprehensive set of attributes."""
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode='before')
    @classmethod
    def pre_validate(cls, args: dict[str, Any]) -> Any:
        if not isinstance(args, dict):
            raise TypeError(f'Expected a dict, got {type(args).__name__}.')
        valid_keys = ['name', 'description', 'parameters', 'returns']
        if list(args.keys()) != valid_keys:
            raise ValueError('Invalid or missing keys. ' +
                             f'Expected keys - {valid_keys}.')
        return args

    def info_message(self) -> str:
        parameters = ''
        for par, t in self.parameters.items():
            parameters += f'"{par}"({t.type}) '

        return (f'- Function name: {self.name}\n'
                f'- Descritpion: {self.description}\n'
                f'- Parameters: {parameters}\n'
                f'- Returns: {self.returns.type}')
