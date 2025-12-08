"""
Quickstart: SCI on a toy signal.

This script shows how to:
- build a simple decomposition Pi and interpreter psi_Theta
- wrap them in an SCI controller
- run the closed-loop system on a synthetic signal
- inspect SP(t) and DeltaSP(t)
"""

import numpy as np

from sci.controller import SCIController
from sci.decomposition import SimpleDecomposition
from sci.interpreter import SimpleInterpreter


def main() -> None:
    # 1. Time axis and synthetic signal (sine + noise + transient)
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
    x[400:450] += 2.0  # simple transient anomaly

    # 2. Construct decomposition Pi and interpreter psi_Theta
    decomp = SimpleDecomposition(config_path="configs/mitbih.yaml")
    interpreter = SimpleInterpreter(config_path="configs/mitbih.yaml")

    # 3. Wrap in SCI controller
    controller = SCIController(
        decomposition=decomp,
        interpreter=interpreter,
        target_sp=0.9,
    )

    # 4. Run closed-loop inference
    result = controller.run(x)

    print("SP(t) length:", len(result.sp_trajectory))
    print("DeltaSP(t) length:", len(result.delta_sp_trajectory))
    print("First 10 SP values:", result.sp_trajectory[:10])
    print("First 10 DeltaSP values:", result.delta_sp_trajectory[:10])


if __name__ == "__main__":
    main()
