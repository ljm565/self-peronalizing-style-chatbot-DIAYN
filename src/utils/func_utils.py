import random
from utils import LOGGER, colorstr



def print_dpo_samples(prompt, target, prediction, style):
    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr('Style     : ') + str(style))
    LOGGER.info(colorstr('Prompt    : ') + prompt)
    LOGGER.info(colorstr('GT        : ') + target)
    LOGGER.info(colorstr('Prediction: ') + prediction)
    LOGGER.info('-'*100 + '\n')


def print_sft_samples(result_dict, n=10):
    keys = random.sample(list(result_dict.keys()), min(n, len(result_dict)))
    
    for key in keys:
        value  = result_dict[key]
        LOGGER.info('\n' + '-'*100)
        LOGGER.info(colorstr('Prompt: ') + key)
        
        for style, v in value.items():
            LOGGER.info(colorstr('Style ') + f'{style}: {v}')
        
        LOGGER.info('-'*100 + '\n')