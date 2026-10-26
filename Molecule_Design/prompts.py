MOL_CAPTION = """
You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n\
Please strictly follow the format, no other information can be provided. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""

MOL_DESIGN = """
"You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES \
string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable. Wrap your response in JSON: {"answer": "<your answer>"}.\n
"""