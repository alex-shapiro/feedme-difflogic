import mlx.core as mx


# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |
# | 1  | A and B              | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |
#
def binary_op(a: mx.array, b: mx.array, i: int) -> mx.array:  # pyright: ignore[reportReturnType]
    assert a[0].shape == b[0].shape
    match i:
        case 0:
            return mx.zeros_like(a)
        case 1:
            return a * b
        case 2:
            return a - a * b
        case 3:
            return a
        case 4:
            return b - a * b
        case 5:
            return b
        case 6:
            return a + b - 2 * a * b
        case 7:
            return a + b - a * b
        case 8:
            return 1 - (a + b - a * b)
        case 9:
            return 1 - (a + b - 2 * a * b)
        case 10:
            return 1 - b
        case 11:
            return 1 - b + a * b
        case 12:
            return 1 - a
        case 13:
            return 1 - a + a * b
        case 14:
            return 1 - a * b
        case 15:
            return mx.ones_like(a)


def binary_ops(a: mx.array, b: mx.array, inputs) -> mx.array:
    r = mx.zeros_like(a)
    for i in range(16):
        r += inputs[..., i] * binary_op(a, b, i)
    return r


# def unique_connections(d_in: int, d_out: int, device: mx.Device):
#     assert d_out * 2 >= d_in
#     x = mx.arange(d_in, dtype=mx.int64)
#     a = x[..., ::2]  # elements @ even indices
#     b = x[..., 1::2]  # elements @ odd indices
#     if a.shape[-1] != b.shape[-1]:
#         m = min(a.shape[-1], b.shape[-1])
#         a = a[..., :m]
#         b = b[..., :m]
