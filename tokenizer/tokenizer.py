class SimpleTokenizer:

    def __init__(self, text):
        self.characters = sorted(set(text))

        self.char_to_id = {
            char: i
            for i, char in enumerate(self.characters)
        }

        self.id_to_char = {
            i: char
            for char, i in self.char_to_id.items()
        }

    def encode(self, text):
        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(self, token_ids):
        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )