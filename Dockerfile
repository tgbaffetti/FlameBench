# CUDA 12.6 PyTorch build retains Pascal support for GTX 1080 Ti.
FROM python:3.11-slim
WORKDIR /workspace
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
COPY pyproject.toml utils.py ./
COPY DataProcessing DataProcessing
COPY Baselines Baselines
COPY Experiments Experiments
RUN pip install --no-cache-dir .
ENV OMP_NUM_THREADS=4
CMD ["python", "-m", "Experiments.run", "--help"]
