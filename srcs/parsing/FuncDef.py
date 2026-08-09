from pydantic import (BaseModel, model_validator,
                      ValidationInfo)


class FuncDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, 'FuncParam']
    returns: 'FuncParam'

    @model_validator(mode='after')
    def check_func_name(self, info: ValidationInfo) -> 'FuncDef':
        if info is None or info.context is None:
            return self

        names = [f.name for f in info.context]
        if self.name in names:
            raise ValueError('Repeated function name.')

        if not self.name.strip():
            raise ValueError('Empty function name.')

        return self

    def get_func_info(self) -> str:
        params = []
        for k, v in self.parameters.items():
            v = v.type
            params.append(f'"{k}": {{"type": "{v}"}}')
        param_s = '{' + ', '.join(params) + '}'
        info = (f'name: "{self.name}"\n'
                f'description: "{self.description}"\n'
                f'parameters: {param_s}\n'
                f'returns: "{self.returns.type}"')
        return info


class FuncParam(BaseModel):
    type: str

    @model_validator(mode='after')
    def check_empty_type(self) -> 'FuncParam':
        if not self.type.strip():
            self.type = 'None'
        return self
