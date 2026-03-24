#!/bin/bash

echo
echo "This shows the base tiny llama model doing a bad job."
echo "If we do the grep command in the bash comment, to look in the dataset, to spot memorization, we'll see response is not there."
echo "The model is hallucinating, which is why I allowed it 40 output tokens."

# grep -i "was confusing and slow" train_alpaca.filtered.jsonl

echo "The output is clearly wrong. This is a negative review, not positive."
echo "It did not follow the the one word response directive either."
echo
echo "Correct response is \"text\": \" negative\"" 
echo

curl -s http://89.169.108.37/v1/completions \
-H "Content-Type: application/json" \
  -d '{
    "model":"basemodel",
    "prompt":"### Instruction: Classify this movie review as either \"positive\" or \"negative\". Provide a single word answer. ### Review: \"the movie was confusing and slow\"\n\n### Response:\n",
    "temperature":0.0,
    "max_tokens":40,
    "stop":["\n\n"]
  }' | jq .

