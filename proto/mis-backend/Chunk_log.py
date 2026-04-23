def read_log_in_chunks(path, chunk_size=25):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        chunk = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
