import re

_REPLACED_WORDS = (
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'sqrt', 'exp', 'log', 'log10',
    'pi', 'e', 'abs',
)


def parse_formula(formula_str):
    """
    Parses a user-friendly math string and converts it into a numpy-executable string.
    Example: 'sin(x**2 + y**2)' -> 'np.sin((x+1e-10)**2 + (y+1e-10)**2)'
    """
    parsed_str = formula_str

    # 1. Map standard math functions to numpy equivalents
    for word in _REPLACED_WORDS:
        pattern = r'\b' + word + r'\b'
        parsed_str = re.sub(pattern, f'np.{word}', parsed_str)

    # 2. Automatically apply a microscopic offset to x and y to prevent division-by-zero at (0,0)
    parsed_str = re.sub(r'\bx\b', '(x + 1e-10)', parsed_str)
    parsed_str = re.sub(r'\by\b', '(y + 1e-10)', parsed_str)

    return parsed_str