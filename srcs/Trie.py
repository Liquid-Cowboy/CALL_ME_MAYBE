class TrieNode():
    def __init__(self) -> None:
        self.children = {}


class TokenTrie():
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, name: str, tokens: list[int]) -> None:
        node = self.root

        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.children['name'] = name

    def get_children(self, node: TrieNode) -> list[int]:
        return list(node.children.keys())

    def name_found(self, node: TrieNode) -> str | None:
        return node.children.get('name')

    def update(self, node: TrieNode, token: int):
        return node.children[token]
