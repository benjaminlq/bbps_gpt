import argparse
import os
import json
import openai
import pandas as pd

from tqdm import tqdm
from config import MAIN_DIR, DATA_DIR, SEED, ARTIFACT_DIR
from utils import get_experiment_logs, save_checkpoint, classify

from openai import OpenAI
from callbacks import TokenCounter
from prompts.prompts_utils import get_gpt4v_messages

def get_argument_parser():
    parser = argparse.ArgumentParser("Hyper Kvasir")
    parser.add_argument("--testcases", "-i", type=str, default=os.path.join(DATA_DIR, "hyper-kvasir", "testcases.txt"), help="Path to testcase file")
    parser.add_argument("--output", "-o", type=str, help="Path to save output artifacts")
    parser.add_argument("--prompts", "-p", type=str, help="Path to prompt files. Must be in json format")
    parser.add_argument("--examples", "-e", type=str, default=None, help="Path to example files")
    parser.add_argument("--checkpoint", "-k", type=str, default=None, help="Path to previous checkpoint")
    parser.add_argument("--temperature", "-t", type=float, default=0, help="Temperature setting for generative model")
    parser.add_argument("--max_tokens", "-tk", type=int, default=512, help="Maximum number of generated tokens")
    parser.add_argument("--description", "-d", type=str, default="exp", help="experiment description")
    parser.add_argument("--query_resize", "-qr", type=bool, default=True, help="Whether to resize query images") 
    parser.add_argument("--example_resize", "-er", type=bool, default=False, help="Whether to resize example images")

    args = parser.parse_args()
    return args
    
def main():
    args = get_argument_parser()
    testcase_path = args.testcases
    temperature = args.temperature
    max_tokens = args.max_tokens
    output_folder = args.output
    description = args.description
    checkpoint = args.checkpoint
    prompts_path = args.prompts
    examples_path = args.examples
    query_resize = "auto" if args.query_resize else None # Currently only support auto resizing.
    example_resize = "auto" if args.example_resize else None # Currently only support auto resizing.

    with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
        api_keys = json.load(f)
        
    openai.api_key = api_keys["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]
    
    logger = get_experiment_logs(description, log_folder=output_folder)
    
    logger.info(f"Running testcase from path {testcase_path}")
    if os.path.exists(testcase_path):
        with open(testcase_path, 'r') as fp:
            data = fp.read()
            all_file_paths = data.split("\n") 
    filenames = [path.split("/")[-1] for path in all_file_paths]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    if checkpoint:
        with open(checkpoint, "r") as f:
            ckpt_content = json.load(f)
        fs_text_responses = ckpt_content["gpt_raw_answers"]
        fs_text_gpt_scores = ckpt_content["gpt_scores"]
        start = int(checkpoint.split("/")[-2])
        logger.info(f"Loading existing checkpoint {checkpoint}")
        
    else:
        fs_text_responses = []
        fs_text_gpt_scores = []
        start = 0
        logger.info(f"Running experiment from start.")
        
    token_counter = TokenCounter()
    client = OpenAI()
    
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
        system_prompt = prompts.get("system")
        query_prompt = prompts.get("query")
        example_prompt = prompts.get("example")      
    
    if examples_path and os.path.isdir(examples_path):
        examples_path = [os.path.join(examples_path, filename) for filename in examples_path] 
    
    from utils import extract_score
    logger.info("Start Running Experiment")
    for idx, query_img_path in enumerate(
        tqdm(all_file_paths[start:], total=len(all_file_paths[start:])),
        start=start):
        try:
            messages = get_gpt4v_messages(
                query_img_path=query_img_path,
                system_prompt=system_prompt,
                query_prompt=query_prompt,
                query_resize=query_resize,
                example_images=examples_path,
                example_prompt=example_prompt,
                example_resize=example_resize
            )

            gptv_response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=SEED,
            )
            
            token_counter.update(gptv_response)
            response_str = gptv_response.choices[0].message.content
            fs_text_responses.append(response_str)
            score_dict = extract_score(response_str, client, token_counter)
            fs_text_gpt_scores.append(score_dict["Score"])        

            if (idx + 1) % 20 == 0:
                ckpt_folder = os.path.join(ARTIFACT_DIR, "hyper-kvasir", "checkpoint", str(idx+1))
                save_checkpoint(ckpt_folder, all_file_paths, fs_text_responses, fs_text_gpt_scores)
                print("Successfully saved checkpoint at folder:", ckpt_folder)
                
        except Exception as e:
            logger.warning(f"Error encountered at file idx {idx} and file {query_img_path}")
            logger.critical(e)
            ckpt_folder = os.path.join(ARTIFACT_DIR, "hyper-kvasir", "checkpoint", str(idx))
            save_checkpoint(ckpt_folder, all_file_paths, fs_text_responses, fs_text_gpt_scores)
            print("Successfully saved checkpoint at folder:", ckpt_folder)
            break
        
    exp_json = []
    for filename, fs_text_raw_answer, fs_text_score \
        in zip(filenames, fs_text_responses, fs_text_gpt_scores):
            exp_json.append(
                {
                    "filename": filename,
                    "fs_text_raw_answer": fs_text_raw_answer,
                    "fs_text_score": fs_text_score
                }
            )
            
    with open(os.path.join(output_folder, f"result.json"), "w") as f:
        json.dump(exp_json, f)
        
    pd_dict = {
        "filename": filenames,
        "fs_text_raw_answer": fs_text_responses,
        "fs_text_score": fs_text_gpt_scores
    }
    result_df = pd.DataFrame(pd_dict)

    result_df = result_df.rename(columns = {"fs_text_score": "gpt_score"})
    GT_DIR = os.path.join(DATA_DIR, "hyper-kvasir", "ground_truths")

    gt_dict = {}

    for gt_score in range(4):
        img_folder = "BBPS " + str(gt_score)
        img_files = os.listdir(os.path.join(GT_DIR, img_folder))
        for img_file in img_files:
            gt_dict[img_file] = gt_score

    result_df["gpt_score"] = result_df["gpt_score"].fillna(-1)
    result_df["gt_score"] = [gt_dict[file] for file in result_df["filename"]]

    result_df["gt_class"] = result_df["gt_score"].apply(lambda x: classify(x))
    result_df["gpt_class"] = result_df["gpt_score"].apply(lambda x: classify(x))

    result_df.to_csv(os.path.join(output_folder, "result.csv"))

    from sklearn.metrics import classification_report
    logger.info(classification_report(result_df["gt_score"], result_df["gpt_score"], digits=5))
    logger.info(classification_report(result_df["gt_class"], result_df["gpt_class"], digits=5))

if __name__ == "__main__":
    main()
    
# python3 src/scripts/hyper-kvasir_experiment.py --output artifacts/hyper-kvasir/one-shot --prompts src/prompts/criteria.json --examples data/samples/example_1.JPG --description 