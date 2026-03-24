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

ENV HF_TOKEN=[redacted]
ENV WANDB_PROJECT=[redacted]
ENV WANDB_API_KEY=[redacted]

RUN curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
RUN cp /root/.nebius/bin/nebius /bin

COPY private.pem .

RUN echo "creating nebius profile"

ENV SA_PROFILE_NAME=joshsa
ENV SA_ID=[redacted]
ENV PUBLIC_KEY_ID=[redacted]
ENV PRIVATE_KEY_PATH=/workspace
ENV NB_PROFILE_NAME=container
ENV NB_PROJECT_ID=[redacted]
ENV NB_TENANT_ID=[redacted]

RUN nebius profile create --endpoint api.nebius.cloud --service-account-id serviceaccount-[redacted] --public-key-id publickey-[redacted]--private-key-file /workspace/private.pem --profile joshsa --parent-id project-[redacted]

ENV models-bucket=[redacted]

RUN echo "installing aws cli"
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
RUN unzip /tmp/awscliv2.zip -d /tmp
RUN bash /tmp/aws/install
RUN rm -rf /tmp/aws

ENV NB_ACCESS_KEY_AWS_ID=[redacted]
ENV NB_SECRET_ACCESS_KEY=[redacted]

RUN aws configure set region eu-north1
RUN aws configure set endpoint_url https://storage.eu-north1.nebius.cloud:443
RUN aws configure set aws_access_key_id [redacted]
RUN aws configure set aws_secret_access_key [redacted]


#NEBIUS_CLI_TOKEN="your-token" 
#NEBIUS_ENDPOINT="https://storage.xx‑region.nebius.cloud" 
