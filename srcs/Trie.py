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
                if not node.children.get(n):
                    node.children[n] = TrieNode()
                node = node.children.get(n)
                if node is None:
                    break

    def get_children(self) -> list[int] | list:
        if not self.curr:
            return []
        return list(self.curr.children.keys()) if self.curr.children else []

    def move_up(self, token: int) -> None:
        if self.curr is None:
            return
        self.curr = self.curr.children.get(token)
