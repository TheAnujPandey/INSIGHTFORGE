from setuptools import find_packages, setup

setup(
    name="insightforge",
    version="0.1.0",
    description="INSIGHTFORGE: AI-powered customer retention platform with churn + SHAP + LangGraph + RAG + FastAPI.",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    python_requires=">=3.10",
)
