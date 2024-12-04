import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
from tqdm import tqdm

from tools import Chatter




if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='0', required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-m', '--resume_path', type=str, required=True)
    parser.add_argument('-r', '--csv_read_path', type=str, required=True)
    parser.add_argument('-w', '--csv_write_path', type=str, required=True)
    args = parser.parse_args()

    # Read csv file and make write csv file
    os.makedirs(os.path.dirname(args.csv_write_path), exist_ok=True)
    questions = pd.read_csv(args.csv_read_path)['Question'].tolist()

    # Initializer model wrapper
    chatter = Chatter(args)
    
    # Pre-trained GPT-2 inference
    for question in tqdm(questions):
        question = question.strip()
        _, response = chatter.generate(question)

        data ={'prompt': question, 'response': response}
        df = pd.DataFrame([data])
        
        if os.path.exists(args.csv_write_path):
            df.to_csv(args.csv_write_path, mode='a', header=False, index=False)
        else:
            df.to_csv(args.csv_write_path, mode='w', header=True, index=False)