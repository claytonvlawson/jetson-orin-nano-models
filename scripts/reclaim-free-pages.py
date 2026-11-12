import sys
# Touch anonymous pages to force page-cache reclaim, then release them all.
# On unified-memory Jetson this is what turns a fragmented buddy list into
# something a multi-GB cudaMalloc can succeed against.
# 2000 is NOT enough on this board: the GGUF read leaves ~5 GB of page cache for
# an 8B model, and the KV cudaMalloc still fails. 5000 drives real reclaim;
# measured result is Cached 2.6 GB -> 1.2 GB and MemFree ~5.0 GB.
target_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
chunk = 32 * 1024 * 1024
bufs = []
done = 0
try:
    while done < target_mb:
        b = bytearray(chunk)
        for i in range(0, chunk, 4096):
            b[i] = 1
        bufs.append(b)
        done += chunk // (1024 * 1024)
        print("touched %d MB" % done, flush=True)
except MemoryError:
    print("MemoryError at %d MB" % done, flush=True)
del bufs
print("released", flush=True)
