class TrieNode():
    def __init__(self) -> None:
        self.children: dict = {}


class Trie():

    def __init__(self, token_ars: list[list]) -> None:
        self.root = TrieNode()
        self.curr = self.root
        for ar in token_ars:
            node = self.root
            for n in ar:
                node.children[n] = TrieNode()
                node = node.children.get(n)
                if node is None:
                    break

    def get_children(self) -> list[int] | None:
        return list(self.curr.children.keys()) if self.curr.children else None

    def move_up(self, token: int) -> None:
        self.curr = self.curr.children.get(token)
