#!/usr/bin/env python3
import torch

from VAR_Q.pack_unpack import (
    pack_last_dim_to_int32_python,
    unpack_last_dim_from_int32_python,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bits = 4
    x = torch.randint(-(2 ** (bits - 1)), 2 ** (bits - 1), (2, 3, 17), dtype=torch.int8, device=device)

    packed, meta = pack_last_dim_to_int32_python(x, bits)
    restored = unpack_last_dim_from_int32_python(packed, meta)

    if not torch.equal(restored, x):
        raise RuntimeError("pack/unpack roundtrip failed")

    print(f"pack/unpack smoke passed on {device}: {tuple(x.shape)} -> {tuple(packed.shape)}")


if __name__ == "__main__":
    main()
