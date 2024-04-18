# modal-outlines-examples

Examples of code that use [Outlines](https://github.com/outlines-dev/outlines) to enable structured text generation for LLMs running on [Modal](https://modal.com).

Is your computer only capable of running small, quantized LLMs? Would you love to have an **Nvidia H100** at home? Keep reading, because we're about to make your dream come true.

## Level 1: The Basics

**The basic idea:** We deploy a function to Modal, then we call that function locally inside a Jupyter notebook.

**The unbelievable part:** The function will run on a super powerful GPU in the cloud. Plus, we get to use free processing credits ($30 monthly).

## Level 2: Different LLMs

Yeah, I know Mistral is good, but what if I want to test different models?

## Level 3: vLLM

Coming [soon](https://github.com/vllm-project/vllm/pull/4109)

## Level 4: FastAPI

How to serve an inference endpoint for structured text generation. GPU-powered. Serverless. Automatically scales. Pay as you go.