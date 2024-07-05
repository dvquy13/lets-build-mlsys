import hashlib

def deterministic_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)
