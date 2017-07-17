import sys, re, json
from itertools import groupby

def clear_text(text):
    text = re.sub(r"[0-9]+", "0", text)
    return text

def main(fi):
    for is_empty, section in groupby(list(fi), key=lambda x: x.strip() == ""):
        if not is_empty:
            
            sent = []
            section = list(section)
            
            if section[0].split(" ")[0].startswith("-DOCSTART-"):
                continue
            if section[0].split() == "":
                continue
            for line in section:
                surface, pos, chunk, target = line.strip().split(" ")
                surface = clear_text(surface)
                token = {
                    "raw": surface,
                    "surface": surface.lower(),
                    "pos": pos,
                    "target": target
                }
                sent.append(token)
            print(json.dumps(sent))

if __name__ == '__main__':
    main(sys.stdin)
