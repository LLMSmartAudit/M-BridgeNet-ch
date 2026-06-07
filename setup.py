from pathlib import Path

from setuptools import setup, find_packages


ROOT = Path(__file__).parent


def read_requirements() -> list[str]:
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        return []
    return [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

setup(
    name="mbridgenet",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=8.0", "pytest-mock>=3.12"],
    },
)
