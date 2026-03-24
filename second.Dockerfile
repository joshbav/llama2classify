FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN pip install --no-cache-dir \
  transformers==4.41.0 \
  datasets==2.19.0 \
  peft==0.11.1 \
  bitsandbytes==0.43.1 \
  accelerate==0.30.1 \
  trl==0.8.6 \
  wandb

RUN touch /iscontainer

WORKDIR /workspace

COPY detectcuda.py .

ENV HF_TOKEN=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
ENV WANDB_PROJECT=nebius
ENV WANDB_API_KEY=3205a4bccc661c762bfa483780109feed2a6824c 
