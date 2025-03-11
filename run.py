import subprocess

for mode in ['train', 'evalaute']: # Two mode: 'train', 'evalaute'
    for stride in [125]:
        for window_size in [500, 750, 1000, 2000]:
            for sas in [1]: # Two types of security attack: 1,  2
                retval = subprocess.run(
                    [
                        "python",
                        f"{mode}.py",
                        f"--sas={sas}",
                        f"--window_size={window_size}",
                        f"--stride={stride}",
                    ]
                )
                retval.check_returncode()

