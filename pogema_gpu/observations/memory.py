"""CPU-safe admission estimates for bounded large-map distance windows."""

WINDOW_CELLS = 129 * 129


def vendor_window_state_bytes(agents, height, width, chunk_size, budget_bytes):
    """Peak adapter memory including inherited scratch and vendor temporaries.

    The vendor adapter retains the serial window allocations. Each raw call
    additionally allocates a bitmap, two frontier queues and an int16 output;
    conversion/indexing needs temporary int32 windows and coordinate vectors.
    Calls are sequential, so only one chunk's vendor workspace is live.
    """
    for value in (height, width):
        if type(value) is not int or value < 1:
            raise ValueError('map dimensions must be positive integers')
    cells = height*width
    plan = window_cache_plan(agents,cells,chunk_size,budget_bytes)
    lanes = plan['chunk_size']
    bitmap = ((cells+31)//32)*4
    frontier = 2*max(4096,2*(height+width))*4
    temporary = lanes*(bitmap+frontier+WINDOW_CELLS*6+128)
    return plan['cache_bytes']+plan['scratch_bytes']+cells+temporary+agents*128+1024


def window_cache_plan(agents, cells, chunk_size, budget_bytes):
    for name, value in (("agents",agents), ("cells",cells), ("chunk_size",chunk_size), ("budget_bytes",budget_bytes)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    cache = agents * (WINDOW_CELLS * 4 + 2 * 4) + 8  # Two device dispatcher counters.
    # One distance plane, BFS queue and overflow flag per scratch lane.
    chunk = min(agents, chunk_size, (budget_bytes - cache) // (cells * 8 + 1))
    if chunk < 1:
        raise ValueError(f"window BFS cache and one scratch lane require {cache + cells*8 + 1} bytes, exceeding budget {budget_bytes}")
    return {"chunk_size": chunk, "cache_bytes": cache,
            "scratch_bytes": chunk * (cells * 8 + 1), "budget_bytes": budget_bytes}
