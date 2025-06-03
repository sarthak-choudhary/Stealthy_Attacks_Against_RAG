#!/bin/bash

eval_datasets="realtimeqa realtimeqa_nochoices open_nq wiki_hotpotqa wiki_open_nq"
models="mistral7b llama7b gpt-4o"
top_tokens="5 10 1000000"

mkdir -p logs

for dataset in $eval_datasets; do
    for model in $models; do
        if [ "$model" = "llama7b" ]; then
            model_config_path="model_configs/llama7b_config.json"
        elif [ "$model" = "mistral7b" ]; then
            model_config_path="model_configs/mistral7b_config.json"
        elif [ "$model" = "gpt-4o" ]; then
            model_config_path="model_configs/azuregpt_config.json"        
        else
            echo "Unknown model: $model"
            continue
        fi
        
        #------- For Poison Attack ----------#
        for top_token in $top_tokens; do
            run_name="${model}_Poison_filter_top_${top_token}_${dataset}_5_seeds"
            name="${model}_Poison_filter_top_${top_token}_${dataset}"
            output_file="logs/${run_name}.log"

            echo "Running: model=$model, dataset=$dataset"
            echo "Logging to $output_file"  

            python main.py \
                --eval_dataset "$dataset" \
                --query_results_dir main \
                --model_name "$model" \
                --model_config_path "$model_config_path" \
                --top_k 10 \
                --gpu_id 0 \
                --attack_method Poison \
                --attack_repeat_times 2 \
                --adv_per_query 1 \
                --top_tokens "$top_token" \
                --name "$name" \
                --attention_filtering \
                > "$output_file" 2>&1
        done

        #-------- For PIA Attack ----------#
        # for top_token in $top_tokens; do
        #     run_name="${model}_PIA_filtering_top_${top_token}_${dataset}_5_seeds"
        #     name="${model}_PIA_filtering_top_${top_token}_${dataset}"
        #     output_file="logs/${run_name}.log"

        #     echo "Running: model=$model, dataset=$dataset"
        #     echo "Logging to $output_file"  

        #     python main.py \
        #         --eval_dataset "$dataset" \
        #         --query_results_dir main \
        #         --model_name "$model" \
        #         --model_config_path "$model_config_path" \
        #         --top_k 10 \
        #         --gpu_id 0 \
        #         --attack_method PIA \
        #         --attack_repeat_times 1 \
        #         --adv_per_query 1 \
        #         --top_tokens "$top_token" \
        #         --name "$name" \
        #         --attention_filtering \
        #         > "$output_file" 2>&1
        # done
    done
done