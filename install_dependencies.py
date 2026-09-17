"""Install the scientific Python dependencies used to validate the mortality scripts."""

import argparse
import subprocess
import sys


def main():
    argparse.ArgumentParser(description=__doc__, epilog="Run inside a dedicated Python 3.12+ virtual environment.").parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError("Use Python 3.12 or newer in a dedicated virtual environment")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==2.5.3", "pandas==3.0.5",
                           "scipy==1.18.1", "scikit-learn==1.9.1", "scikit-survival==0.28.0", "joblib==1.6.0"])


if __name__ == "__main__":
    main()
