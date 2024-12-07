import pickle
import random
import pandas as pd



questions = pd.read_csv('data/questions.csv')['prompt'].tolist()

child_response = pd.read_csv('data/child_gt.csv')['response'].tolist()
professor_response = pd.read_csv('data/professor_gt.csv')['response'].tolist()
philosopher_response = pd.read_csv('data/philosopher_gt.csv')['response'].tolist()
not_preferred_response = pd.read_csv('data/vanilla_gpt2_results.csv')['response'].tolist()

assert len(questions) == len(child_response) == len(professor_response) == len(philosopher_response)




train_dpo, test_dpo = [], []
for i, (q, child_r, prof_r, phil_r, non_preferred_r) in enumerate(zip(questions, child_response, professor_response, philosopher_response, not_preferred_response)):
    tmp1 = {'prompt': q, 'non_preferred_response': non_preferred_r, 'preferred_response': child_r, 'style_id': 0}
    tmp2 = {'prompt': q, 'non_preferred_response': non_preferred_r, 'preferred_response': prof_r, 'style_id': 1}
    tmp3 = {'prompt': q, 'non_preferred_response': non_preferred_r, 'preferred_response': phil_r, 'style_id': 2}

    if i + 1 <= 95:
        train_dpo.append(tmp1)
        train_dpo.append(tmp2)
        train_dpo.append(tmp3)
    else:
        test_dpo.append(tmp1)
        test_dpo.append(tmp2)
        test_dpo.append(tmp3)


random.shuffle(train_dpo)
random.shuffle(test_dpo)

print(f'train: {len(train_dpo)}, test: {len(test_dpo)}')


with open('data/dpo_data/train.pkl', 'wb') as f:
    pickle.dump(train_dpo, f)

with open('data/dpo_data/validation.pkl', 'wb') as f:
    pickle.dump(test_dpo, f)