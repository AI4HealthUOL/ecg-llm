import json
import os


def append_to_json_file(filename, data):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[")
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("]")
    else:
        with open(filename, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            last_char = f.read(1)
            if last_char == b"]":
                f.seek(-1, os.SEEK_END)
                f.truncate()
            else:
                f.write(b",")

        with open(filename, "a", encoding="utf-8") as f:
            f.write(",")
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("]")
