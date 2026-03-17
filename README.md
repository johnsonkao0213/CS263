# CS263 Final Project: Identity Bias in Multi-Agent Debate - Quantifying Conformity and Obstinacy in LLM Collaboration



## Requirements Setup

(1) Setup environment

```
conda env create -f environment.yaml
conda activate venv
```

(2) Setup Huggingface API

You need to have a huggingface account with a corresponding API credentials saved under a file named ```token```. It should contain a single line of the token string.

(2b) Setup Gemini API (optional)

If you plan to run Gemini-3-Flash, set the environment variable ```GEMINI_API_KEY``` (or pass ```--gemini_api_key```). This repo accepts ```--model gemini-3-flash``` as an alias for the Gemini 3 Flash model.

(3) Setup Directories

Create an output folder:
```
mkdir out
```



## Run Experiments

The experiment commands for each dataset are provided in ```scripts/```.
For each experiment, you will need to set up a directory to load the LLM parameters and datasets. To do that, add the ```--data_dir``` and ```--model_dir``` arguments to the command with your own directories.

For example, to run the arithmetics dataset on Qwen2.5-7B-Instruct, run
```
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --data_dir [your_directory] --model_dir [your_directory]
```

To run Gemini 3 Flash, run
```
GEMINI_API_KEY=your_key python src/main.py --model gemini-3-flash --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --data_dir [your_directory]
```

To run Sparse MAD or Centralized MAD, add ```--sparse``` or ```--centralized``` to the command. To run heterogeneous agent settings, add ```--multi_persona```.

To compute Conformity vs. Obstinacy (Appendix A.3) with a single-peer debate prompt (Appendix B.1), add:
```
--single_peer --prompt_template paper --report_identity_bias
```
Metrics are saved to ```out/metrics/{run_name}_identity_bias.json```.
```
