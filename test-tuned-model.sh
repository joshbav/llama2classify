#!/bin/bash

echo
echo "This shows the tuned model doing far better. Notice it also has 40 output tokens, but answers with 1 word."
echo
echo "Classify this movie review as either \"positive\" or \"negative\". Provide a single word answer." 
echo "Review: \"the movie was confusing and slow\""
echo
echo Correct response is "\"text\": \" negative\""
echo


curl -s http://89.169.109.59/v1/completions \
-H "Content-Type: application/json" \
  -d '{
    "model":"trainedmodel",
    "prompt":"### Instruction: Classify this movie review as either \"positive\" or \"negative\". Provide a single word answer. ### Review: \"the movie was confusing and slow\"\n\n### Response:\n",
    "temperature":0.0,
    "max_tokens":40,
    "stop":["\n\n"]
  }' | jq .

