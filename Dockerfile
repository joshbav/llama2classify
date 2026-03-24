FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN pip install --no-cache-dir \
  transformers==4.41.0 \
  datasets==2.19.0 \
  peft==0.11.1 \
  bitsandbytes==0.43.1 \
  accelerate==0.30.1 \
  trl==0.8.6 \
  wandb \

RUN touch /iscontainer

WORKDIR /workspace

COPY detectcuda.py .

ENV HF_TOKEN=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
ENV WANDB_PROJECT=nebius
ENV WANDB_API_KEY=3205a4bccc661c762bfa483780109feed2a6824c 

RUN curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
RUN cp /root/.nebius/bin/nebius /bin

COPY private.pem .

RUN echo "creating nebius profile"

ENV SA_PROFILE_NAME=joshsa
ENV SA_ID=serviceaccount-e00g8k04vp6pk27xb8
ENV PUBLIC_KEY_ID=NAKI73PMY5TBBR365BCA
ENV PRIVATE_KEY_PATH=/workspace
ENV NB_PROFILE_NAME=container
ENV NB_PROJECT_ID=project-e00fp8xacz6pnzwv87n77
ENV NB_TENANT_ID=tenant-e00hnw9t8x3etx9frk

RUN nebius profile create --endpoint api.nebius.cloud --service-account-id serviceaccount-e00g8k04vp6pk27xb8 --public-key-id publickey-e00xnabh0gg7tvs684 --private-key-file /workspace/private.pem --profile joshsa --parent-id project-e00fp8xacz6pnzwv87n77

ENV models-bucket=4f4faegf-models

RUN echo "installing aws cli"
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
RUN unzip /tmp/awscliv2.zip -d /tmp
RUN bash /tmp/aws/install
RUN rm -rf /tmp/aws

ENV NB_ACCESS_KEY_AWS_ID=NAKI73PMY5TBBR365BCA
ENV NB_SECRET_ACCESS_KEY=MugO7+TeYoD+U68qfWz2jS3VXSJl4pnRwNNWkicS

RUN aws configure set region eu-north1
RUN aws configure set endpoint_url https://storage.eu-north1.nebius.cloud:443
RUN aws configure set aws_access_key_id NAKI73PMY5TBBR365BCA
RUN aws configure set aws_secret_access_key MugO7+TeYoD+U68qfWz2jS3VXSJl4pnRwNNWkicS


#NEBIUS_CLI_TOKEN="your-token" 
#NEBIUS_ENDPOINT="https://storage.xx‑region.nebius.cloud" 
