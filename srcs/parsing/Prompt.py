from pydantic import BaseModel, model_validator


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode='after')
    def check_if_empty(self) -> 'Prompt':
        if not self.prompt.strip():
            raise ValueError('Empty prompt.')
        return self
