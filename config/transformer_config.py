class PatternFactory:
    _input_shape: tuple[int, int, int, int] = (5, 10, 10, 256)
    _P: int = 2
    _M: int = 4

    def __init__(self, input_shape: tuple[int, int, int, int]):
        PatternFactory._input_shape = input_shape

    @staticmethod
    def axial():
        T, H, W, C = PatternFactory._input_shape
        cuboid_sizes = [(T, 1, 1), (1, H, 1), (1, 1, W)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('l', 'l', 'l')]
        shift_sizes = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_sizes, shift_sizes, strategy

    @staticmethod
    def full_attention():
        T, H, W, C = PatternFactory._input_shape
        cuboid_sizes = [(T, H, W)]
        strategy = [('l', 'l', 'l')]
        shift_sizes = [(0, 0, 0)]
        return cuboid_sizes, shift_sizes, strategy

    @staticmethod
    def video_swin():
        T, H, W, C = PatternFactory._input_shape
        P = min(PatternFactory._P, T)
        M = min(PatternFactory._M, H, W)
        cuboid_sizes = [(P, M, M), (P, M, M)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
        shift_sizes = [(0, 0, 0), (P // 2, M // 2, M // 2)]
        return cuboid_sizes, shift_sizes, strategy

    @staticmethod
    def divided_space_time():
        T, H, W, C = PatternFactory._input_shape
        cuboid_sizes = [(T, 1, 1), (1, H, W)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
        shift_sizes = [(0, 0, 0), (0, 0, 0)]
        return cuboid_sizes, shift_sizes, strategy

    @staticmethod
    def earthformer():
        T, H, W, C = PatternFactory._input_shape

        if H <= PatternFactory._M and W <= PatternFactory._M:
            cuboid_sizes = [(T, 1, 1), (1, H, W)]
            strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
            shift_sizes = [(0, 0, 0), (0, 0, 0)]
        else:
            cuboid_sizes = [(T, 1, 1), (1, PatternFactory._M, PatternFactory._M)
                , (1, PatternFactory._M, PatternFactory._M)]
            strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('d', 'd', 'd')]
            shift_sizes = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_sizes, shift_sizes, strategy