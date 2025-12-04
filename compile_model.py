import os
import sys

import torch

with torch.no_grad():
    model_path = sys.argv[1]

    program = torch.export.load(model_path)

    output_path = torch._inductor.aoti_compile_and_package(
        program,
        package_path=os.path.join(os.getcwd(), "compiled_model.pt2")
    )

    print(output_path)