def parse_structure(structure: str) -> tuple[list[int], list[int]]:
    """Parse the structure string and return the number of layers and sizes."""
    blocks = structure.split(",")
    n_layers = []
    sizes = []
    for block in blocks:
        n_l, size = block.split("x")
        n_layers.append(int(n_l))
        sizes.append(int(size))
    return n_layers, sizes
