class DefaultFactory:
    def __init__(self):
        pass

    @staticmethod
    def default_block_cuboid_sizes():
        return [(4, 4, 4), (4, 4, 4)]

    @staticmethod
    def default_block_cuboid_strategy():
        return [('l', 'l', 'l'), ('d', 'd', 'd')]

    @staticmethod
    def default_cuboid_shift_size():
        return [(0, 0, 0), (0, 0, 0)]

    @staticmethod
    def default_depth():
        #return [4, 4, 4]
        return [1, 1, 1]
