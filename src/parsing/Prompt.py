from pydantic import BaseModel, model_validator


class Prompt(BaseModel):
    """
    Typical prompt structure.

    Atributes:
        prompt: prompt in string form
    """
    prompt: str

    @model_validator(mode='after')
    def check_if_empty(self) -> 'Prompt':
        """
        Checks if the prompt is empty and raises errors if need be.

        Returns: itself
        """
        if not self.prompt.strip():
            raise ValueError('Empty prompt.')
        return self
