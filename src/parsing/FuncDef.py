from pydantic import (BaseModel, model_validator,
                      ValidationInfo)


class FuncDef(BaseModel):
    """
    Typical function definition validation.

    Atributes:
        name: Function name
        description: Function description
        parameters: Function parameters using a FuncParam obj
        returns: Function return value
    """
    name: str
    description: str
    parameters: dict[str, 'FuncParam']
    returns: 'FuncParam'

    @model_validator(mode='after')
    def check_func_name(self, info: ValidationInfo) -> 'FuncDef':
        """
        Checks wether a function with the same name has already been
        registered or not.

        Parameters:
            info: Obj with a list of names to be compared with

        Returns: itself
        """
        if info is None or info.context is None:
            return self

        names = [f.name for f in info.context]
        if self.name in names:
            raise ValueError('Repeated function name.')

        if not self.name.strip():
            raise ValueError('Empty function name.')

        return self

    def get_func_info(self) -> str:
        """
        Writes a descriptive text about the function.

        Returns: text in string form
        """
        params = []
        for k, v in self.parameters.items():
            t = v.type
            params.append(f'"{k}": {{"type": "{t}"}}')
        param_s = '{' + ', '.join(params) + '}'
        info = (f'name: "{self.name}"\n'
                f'description: "{self.description}"\n'
                f'parameters: {param_s}\n'
                f'returns: "{self.returns.type}"')
        return info


class FuncParam(BaseModel):
    """
    Typical function parameter definition for both
    parameter validation and return type validation.

    Atributes:
        type: Parameter or return value type
    """
    type: str

    @model_validator(mode='after')
    def check_empty_type(self) -> 'FuncParam':
        """
        Checks if the "type" field is an empty string.
        It will default "type" to "None".

        Returns: itself
        """
        if not self.type.strip():
            self.type = 'None'
        return self
