from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = ""
readme = here / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf8")

setup(
    name="streaming_vlm",
    version="0.0.0",
    description="streaming vlm package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
)
