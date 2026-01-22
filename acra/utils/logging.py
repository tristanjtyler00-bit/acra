import json

from rich import print


def append_to_log(log_file, messages, verbose=True):
    with open(log_file, 'a', encoding="utf-8") as f:
        if isinstance(messages, dict):
            messages = json.dumps(messages, indent=4)
        f.write(messages)
        f.write('\n')
        f.flush()  # force write to disk
    if verbose:
        print(messages)
    return True
