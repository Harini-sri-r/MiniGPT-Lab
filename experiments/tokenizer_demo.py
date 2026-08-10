from tokenizer.tokenizer import SimpleTokenizer


text = "hello world"

tokenizer = SimpleTokenizer(text)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Original text:")
print(text)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(decoded)

print("\nCharacter to ID:")
print(tokenizer.char_to_id)