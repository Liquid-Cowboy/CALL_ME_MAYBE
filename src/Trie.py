class TrieNode():
    def __init__(self) -> None:
        self.children: dict[int, 'TrieNode'] = {}


class Trie():
    """Main class for trie structure."""

    def __init__(self, token_ars: list[list[int]]) -> None:
        """
        Instantiates the trie.

        Parameters:
            token_ars: list of arrays of token ids

        Atributes:
            self.root: head of the trie structure
            self.curr: variable keeping track of the current
            node we're analysing
        """
        self.root = TrieNode()
        self.curr: TrieNode | None = self.root
        for ar in token_ars:
            node = self.root
            for n in ar:
                if not node.children.get(n):
                    node.children[n] = TrieNode()
                node = node.children[n]

    def get_children(self) -> list[int]:
        """
        Gets a list of the current node's children.

        Returns: list of current node's children
        """
        if not self.curr:
            return []
        return list(self.curr.children.keys()) if self.curr.children else []

    def move_up(self, token: int) -> None:
        """
        Replaces the current node with one of it's children.

        Parameters:
            token: token id of the current node's child to be set
        """
        if self.curr is None:
            return
        self.curr = self.curr.children.get(token)
