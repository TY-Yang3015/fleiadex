import inspect


class StructureError(Exception):
    def __init__(self, expected_structure, actual_structure):
        super().__init__(
            f"the expected {expected_structure} does not match the actual {actual_structure}"
        )


class DimensionError(Exception):
    def __init__(self, expected_dimension, actual_dimension):
        super().__init__(
            f"the expected {expected_dimension} does not match the actual {actual_dimension}."
        )


class CheckPointError(Exception):
    def __init__(self, error):
        super().__init__(
            f"the checkpoint was not properly saved, due to the error:\n{error}"
        )


class DimensionMismatchError(Exception):
    def __init__(self, shape_var_1, shape_var_2):
        super().__init__(
            f"the shape variable {shape_var_1} does "
            f"not match the shape variable {shape_var_2}."
        )
