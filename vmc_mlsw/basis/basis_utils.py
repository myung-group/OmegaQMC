import re

def extract_basis_block(basis_text: str, element: str) -> str:

    pattern = rf'BASIS\s+"{element}\b[^"]*".*?END'
    match = re.search(pattern, basis_text, flags=re.DOTALL)

    if match:
        return match.group(0)

    pattern2 = rf'^{element}\s+[SPDFGH]\b[\s\S]*?^END'
    match2 = re.search(pattern2, basis_text, flags=re.DOTALL | re.MULTILINE)

    if match2:
        return match2.group(0)

    raise ValueError(f"Basis block for element '{element}' not found.")
