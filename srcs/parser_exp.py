STRINGS = [
    ' "a": 3, "b": 5 }',
    ' "a": 12, "b": 4 }',
    ' "n": 4 }',
    ' "n": 7 }',
    ' "principal": 1234567.89, "rate": 0.0375, "years": 23 }',
    ' "query": "SELECT * FROM users", "database": "production" }',
    ' "query": "INSERT INTO logs VALUES (1, 2, 3)", "database": "system" }',
    ' "path": "/home/user/data.json", "encoding": "utf-8" }',
    ' "path": "C:\\Users\\john\\config.ini", "encoding": "latin-1" }',
    ' "template": "Format template: Hello {user}\'s profile!" }',
    ' "template": "Format template: Say \"hello\" to {name}" }'
]

def parse_strings(decoded_params:list[str]):
    for s in decoded_params:
        last_i = len(s) - 1 - s[::-1].index('}')
        clean_s = s[:last_i]
        params = clean_s.split(", \"")
        print('PARAMS:')
        for p in params:
            print(p)


if __name__ == '__main__':
    parse_strings(STRINGS)
