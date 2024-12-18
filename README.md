# FarExStance: Explainable Stance Detection for Farsi

**This repo will be updated soon!**

This repository contains data and code for the paper [FarExStance: Explainable Stance Detection for Farsi](https://coling2025.org/program/main_conference_papers/#long-papers). This research will be presented at the 31st International Conference on Computational Linguistics (COLING 2025).

## Introduction
We introduced a new dataset for Farsi, _FarExStance_, **the first and largest claim-based explainable stance detection dataset in Farsi**. We conducted extensive experiments to establish baseline performance using a variety of multilingual open-source small and large language models, retrieval-augmented generation, and parameter-efficient fine-tuning approaches. We also provided insights into the strengths and limitations of these approaches using both automatic and human evaluations. Our dataset, manually curated with labeled instances and supporting evidence, has been made publicly available to foster further research in this area, e.g. experiments with the social media perspectives.

## Reproducing the Experiments

In order to reproduce the results of our experiments, you can run ``*.sh`` files with various arguments. To cite an example, by running the ``huggingface_exp.sh``, you can reproduce our zero-shot and few-shot results of the explainable stance detection task on the test set by using [Command-R-32B](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) and [Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B).

You can find the description of each argument in [rag_inference.py file](https://github.com/Zarharan/FarExStance/blob/main/rag_inference.py)
