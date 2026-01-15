# event-location-extraction-prompt-optimization

This repo is a PoC for a self-optimizing agentic script for finding the best prompt for event-location extraction and NER. 
I wanted to find out whether a small LLM can perform well-enough to extract the correct principal location and what actually happened from newspaper articles.

The whole idea is that a small LLM is prompted with a test prompt and a large LLM evaluates the output. Until it reaches a certain quality (9.5/10) or a maximum number of iterations, it continues to adapt the prompt. 
The JSON scheme for structured output is hard-coded but could be subject to this process too!

Currently it uses lmstudio but it could be easily rewritten to work with vllm or external APIs.

I chose to use 

STUDENT_MODEL = "liquid/lfm2.5-1.2b" 
TEACHER_MODEL = "qwen/qwen3-next-80b"

as both work really well in general. For me, they are the best in their class but as the field is so dynamic, there could be better models of course.

## Input

- test articles (3 in the example)
- json schema for structured outputs (something you need to use to get reliable and correct json)

## To Do's

### Batch inferencing

Unfortunately lmstudio does not support batch inferencing yet (sending many prompts at the same time to get around 10x more throughput).
vllm does support batching, but not on mlx. So if you're on Apple Silicon, a good way to speed up these runs would be to use mlx-lm directly or a light-weight server around it. 