import tiktoken


def main():
    text = "Hello, world!"

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(tokens)


if __name__ == "__main__":
    main()
