import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def get_num_tokens(prompt: str, verbose: bool = False) -> int:
    num_tokens = len(encoding.encode(prompt))
    if verbose:
        print(f"Input text of len {len(prompt)} requires {num_tokens} tokens")
    return num_tokens


def split_string_with_limit(text: str, limit: int = 3084, encoding: tiktoken.Encoding = encoding, overlap:int = 1024) -> list[str]:
    """Split a string into parts of given size without breaking words.
    
    Args:
        text (str): Text to split.
        limit (int): Maximum number of tokens per part.
        encoding (tiktoken.Encoding): Encoding to use for tokenization.
        
    Returns:
        list[str]: List of text parts.
        
    """
    tokens = encoding.encode(text)
    parts = []
    text_parts = []
    current_part = []
    current_count = 0

    for token in tokens:
        current_part.append(token)
        current_count += 1

        if current_count >= limit:
            
            parts.append(current_part)
            if overlap:
                current_part = current_part[-overlap:]
                current_count = len(current_part)
            else:
                current_part = []
                current_count = 0

            
    if current_part:
        parts.append(current_part)

    # Convert the tokenized parts back to text
    for part in parts:
        text = [
            encoding.decode_single_token_bytes(token).decode("utf-8", errors="replace")
            for token in part
        ]
        text_parts.append("".join(text))

    return text_parts