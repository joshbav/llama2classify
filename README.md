# llama2classify
This was my take-home assignment for my interview at Nebius during the summer of 2025.

I was tasked with a fine-tuning job on any existing model, and demonstrating the improvement over the base model. I was allowed to use existing training code, but decided to make my own (with help).

I chose TinyLlama since it's designed for text completion, not classification. This way I could show the improvement. I was worried that if I used a BERT model the zero-shot performance would be too good, then I wouldn't be able to demonstrate any reliable improvement. 

I used the IMDB movie sentiment dataset, since it's small enough that it hopefully would not cause catastrophic model collapse. It's too big to include in this repo. 

ChatGPT and Claude were utilized to help me with the training code, but I had to make some changes manually regardless.

Using [Nebius Soperator](https://github.com/nebius/soperator) (Slurm on Kubernetes) I ran a distributed PyTorch training job. I then used vLLM to serve the model in Kubernetes, on the same cluster as Soperator. I had to hack Soperator's configmap to scale in the worker pods, in order to free up a GPU node for vLLM. 

This took considerably more time than I had estimated. I was surprised how useful ChatGPT and Claude were for "big items" and how many mistakes they made on the "small items", such as basic data conversion of the IMDB dataset to Alpaca format, catching data formatting errors, etc. I eventually realized my context window had been filled and therefore the LLMs were forgetting earlier directions and problems.
# BIGGEST CHALLENGE
1. TinyLlama is designed for next-token prediction, not sentiment analysis, so the tokenizer is not configured for training an Alpaca dataset.
2. Causal LLMs usually do not use pad_token during training, so it's not set.
3. Hugging Face Transformers silently fall back to using eos_token as pad_token. This is a [known issue](https://discuss.huggingface.co/t/llama-pad-token/480010) that took a while to discover. The [smoking-gun.py](https://github.com/joshbav/llama2classify/blob/main/smoking-gun.py) script was made to find the problem.
   
   I smiled when I first saw the output:
   
   [INFO] Checking special tokens...
   [problem found!] eos_token is the same as pad_token: '</s>’
   [problem found!] At least two special tokens are set to the same value — fix the tokenizer config.
4. [after-smoking-gun.py](https://github.com/joshbav/llama2classify/blob/main/after-smoking-gun.py) was created to ensure the problem was fixed.
   Was ran locally, to do as much tokenizer work as possible, in order to save time since we don't need a GPU for tokenization.
   It simulated the data flow through the loss masking logic to see if any entries end up fully masked, which was the cause of a problem since loss had gone to 0 during training.
# A FEW KEY FILES
[This](https://github.com/joshbav/llama2classify/blob/main/distributed_train.py) is the custom PyTorch training script.

[This](https://github.com/joshbav/llama2classify/blob/main/train5.sbatch) is the Slurm .sbatch file for the training job.

[This](https://github.com/joshbav/llama2classify/blob/main/Dockerfile) is the Dockerfile for running the pytorch job, and doing other things.

[This](https://github.com/joshbav/llama2classify/blob/main/Nebius%20Demo.pptx) is the powerpoint for presenting my work during the interview.

This was a fun and challenging project.
