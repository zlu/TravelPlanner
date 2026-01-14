import argparse
import os
from datasets import load_dataset
from tqdm import tqdm
import json


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")

    args = parser.parse_args()

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'


    results = open(f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt','r').read().strip().split('\n')
    
    # Use reuse_cache_if_exists to avoid downloading script if data is already cached
    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train', download_mode='reuse_cache_if_exists')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation', download_mode='reuse_cache_if_exists')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test', download_mode='reuse_cache_if_exists')['test']

    idx_number_list = [i for i in range(1,len(query_data_list)+1)]
    for idx in tqdm(idx_number_list[:]):
        plan_path = f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'
        if not os.path.exists(plan_path):
            continue
        generated_plan = json.load(open(plan_path))
        results_key = f'{args.model_name}{suffix}_{args.mode}_results'
        if results_key not in generated_plan[-1]:
            candidates = [
                key for key in generated_plan[-1].keys()
                if key.endswith(f'{suffix}_{args.mode}_results')
            ]
            if not candidates:
                candidates = [
                    key for key in generated_plan[-1].keys()
                    if key.endswith(f'_{args.mode}_results')
                ]
            if candidates:
                results_key = candidates[0]
        if generated_plan[-1].get(results_key) not in ["", "Max Token Length Exceeded."]:
            try:
                raw_line = results[idx - 1]
                if "```json" in raw_line:
                    result = raw_line.split('```json')[1].split('```')[0]
                else:
                    result = raw_line
                    if "\t" in result:
                        result = result.split("\t", 1)[1]
            except Exception:
                print(f"{idx}:\n{results[idx-1]}\nThis plan cannot be parsed. Please check the format.")
                result = None
            try:
                if result is None:
                    raise ValueError("Empty parse result.")
                result = result.strip()
                if not result or result.isdigit():
                    raise ValueError("Non-plan placeholder.")
                if args.mode == 'two-stage':
                    generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = eval(result)
                else:
                    generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = eval(result)
            except Exception:
                print(f"{idx}:\n{result}\n This is an illegal json format. Please modify it manualy when this occurs.")
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
        else:
            if args.mode == 'two-stage':
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
            else:
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
  
        with open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json','w') as f:
            json.dump(generated_plan,f)
