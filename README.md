# Stealthy Attacks Against RAG

This is the evaluation code of the paper "Through the Stealth Lens: Rethinking Attacks and Defenses in RAG". 

This serves as an evaluation of existing poisoning attacks designed for Retrieval-augmented Generation (RAG) systems against the AV Filter proposed in the aforementioned paper. 

To achieve this, we have incorporated and adapted baseline implementations along with their publicily released test sets from [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) and [Certifiably Robust RAG](https://github.com/inspire-group/RobustRAG) as a foundation of our evaluation. We extend our gratitude to the contributors of those repositories for their valuable work.

## Attacks
We have evaluated two representative poisoning attacks:

1. **Poison** - A poisoned passage is crafted using LLMs (e.g., GPT-4o) to subtly embed an adversary-specified target answer for a particular query. This passage blends naturally with retrieved content and aims to bias the model’s output without direct instructions.

2. **Prompt Injection Attack (PIA)** - This attack embeds explicit instructions within the poisoned passage, directing the system to respond with an adversary-chosen target answer when a specific query is posed.

Additionally, we assess **Adaptive Attacks** using jailbreak methods such as GCG and AutoDAN, which are designed to bypass the proposed AV Filter.

## Defenses

Alongside our evaluation of the proposed **AV Filter**, we benchmark against two certified defense methods from [Certifiably Robust RAG](https://github.com/inspire-group/RobustRAG):

- **Cert. RAG-Keyword**
- **Cert. RAG-Decoding**

These serve as baseline defenses for comparison with the AV Filter under various adversarial settings.

##  Datasets

We evaluate on four open-domain question answering datasets, using both **Google Search** and **Wikipedia Corpus** as sources for retrieved context:

- **RealtimeQA-MC (RQA-MC)**
- **RealtimeQA (RQA)**
- **Natural Questions (NQ)**
- **HotpotQA**

Query-context pairs are obtained from the publicly released artifacts of [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) and [Certifiably Robust RAG](https://github.com/inspire-group/RobustRAG). Context retrieval is performed using:
- **Google Search** (from Certifiably Robust RAG)
- **Wikipedia Corpus with Contriever retriever** (from PoisonedRAG)

Each dataset is stored in a corresponding `.json` file located in `results/adv_targeted_results`.

We suggest using Anaconda for managing the environment.

## Setting up conda 
First, create a conda virtual environment with Python 3.9.21 and activate the environment.
```
conda create -n venv python=3.9.21
conda activate venv
```
Run the following command to install all the required python packages.
```
pip install -r requirements.txt
```

## Set API key

If you want to use GPT-4o, please enter your api key in **model_configs** folder.

For LLaMA-2 and Mistral-7B, the api key is your **HuggingFace Access Tokens**. 

Here is an example:

```json
"api_key_info":{
    "api_keys":[
        "Your api key here"
    ],
    "api_key_use": 0
},
```

## Usage 
Reproduce the evaluation results for AV Filter by running the following script.
```
./run_script.sh
```
To run a single a single attack against AV Filter, run the following commands with the right system arguments:
```
python main.py --eval_dataset [dataset] --model_name [model] --model_config_path [model_config_file] --attack_method [attack] --attack_repeat_times [repetions in poisoned passage] --adv_per_query [number of poisoned passage] --top_tokens [alpha for AV Filter] --name [name for results] --attention_filtering 
```
Each run will be store its evaluation result in `results/query_results/main` directory. This is the list of additional arguments.
| Argument     | Values       | Use / Description                          |
|--------------|------------------|--------------------------------------------|
| `--top_k`     | `int` | specifies the number of passages to retrieve per query with defaut value `10`|
| `--adaptive_attack_GCG`   | - | add to run adaptive attack using GCG against AV Filter |
| `--adaptive_attack_autoDAN`  | - | add to run adaptive attack using AutoDAN against AV Filter |
| `--security_game`       | - | add to evaluate security game on the successful attacked queries               |
| `--top_tokens`  | `int` | specifies the value of $\alpha$ to use top-$\alpha$ tokens for computing attention score |
| `--filtering_threshold`  | `float` | specifies the threshold of the variance of attention scores in retrieved set to enable filtering with default value `26.20` |
| `--lambda_value`  | `float` | specifies the value of $\lambda$ while running adaptive attacks with default value `0.1`|

## Logs

Since many of the evaluations — particularly the adaptive attacks — are time-consuming, we have also provided logs and results for each run.

- The logs for all runs reported in the paper can be found in the `logs/` directory.
- Their corresponding results are located in `results/query_results/main/`.
- Additionally, the optimized poisoned passages generated by the adaptive attacks are available in `results/query_results/adap_poisoned_passage/`.

## Acknowledgment
The code and datasets of the evaluation largely reuse [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) and [Certifiably Robust RAG](https://github.com/inspire-group/RobustRAG).
