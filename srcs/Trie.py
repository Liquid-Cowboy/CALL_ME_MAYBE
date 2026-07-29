# from pydantic import BaseModel

class TrieNode():
    """Unit of a node in a trie structure."""
    def __init__(self) -> None:
        """
        Initiates the node along with a dictionary
        to receive a variable amount of references to
        other nodes.
        """
        self.children = {}


class TokenTrie():
    def __init__(self) -> None:
        """Initiates the structure with a node root."""
        self.root = TrieNode()

    def insert(self, name: str, tokens: list[int]) -> None:
        """
        Inserts multiple nodes in the trie structure
        following the order of a list of tokens. In the
        last node, a reference to the decoded token list
        is included.
        
        Parameters:
            name -- Decoded version of the list of tokens.
            tokens -- List of tokens.
        """
        node = self.root

        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.children['name'] = name

    def get_children(self, node: TrieNode) -> list[int]:
        """
        Looks for the possible following tokens for the
        parent (previous) token.

        Parameters:
            node -- Parent node from which to extract
            children nodes.

        Returns:
            - A list of possible child tokens.
        """
        return list(node.children.keys())

    def name_found(self, node: TrieNode) -> str | None:
        """
        Checks for a 'name' key in the given node.

        Parameters:
            node -- Node to check.
        
        Returns:
            - The string corresponding to the 'name'
            key or 'None' if no 'name' key exists.
        """
        return node.children.get('name')

    def update(self, node: TrieNode, token: int) -> TrieNode | None:
        """
        Sets the given node to one of it's children.

        Parameters:
            node -- Node to be set.
            token -- Key to the next node to be selected.
        
        Returns:
            - The trie node corresponding to the given token key
            or 'None' if none is found.
        """

        return node.children.get(token)
