# Self-Personalizing Chatbot Tailored to User’s Style

## Purposes
Developing a chatbot model that can understand and adapt to a user’s conversational style involves leveraging reinforcement learning (RL) to improve the model's performance dynamically.
By incorporating RL algorithms, the chatbot can learn from user interactions to tailor its responses more effectively, enhancing the user experience.
This approach allows for an adaptable system capable of recognizing individual conversational nuances and refining its behavior over time.
This project was initiated to demonstrate the potential of a chatbot to evolve in real-time and adapt to individual user preferences.
Additionally, a demo page is provided to showcase how the chatbot adjusts to real user styles.

* Developing a chatbot model that can understand a person’s conversational style and adapt its responses accordingly.
* Improving the model’s performance by incorporating RL algorithms.
* Comparing the effects when applying RL to chatbot and creating a demo page tailord to real-world user scenarios.
<br><br><br>


## Methods
### Pre-training a Language Model
For this project, I selected GPT-2 as the model and restricted the training data to everyday conversations.
To train the model with knowledge of daily conversations, I first performed pre-training using the [DailyDialog](http://yanran.li/dailydialog) dataset, which consists of multi-turn conversations.
The data statistics are shown in the table, and all data except for 2,000 validation and test samples were used for training. 
The model checkpoints were saved based on BLEU and NIST scores, and I evaluated using the validation set.
The training hyperparameters are detailed in the below.
* Model: Pretrained GPT-2
* Data: [DailyDialog](http://yanran.li/dailydialog) which is consists of everyday conversations and is structured as multi-turn exchanges.
    | Train | Validation | Test |
    |:-----:|:----------:|:----:|
    |11,118 | 1,000      | 1,000|
* Statistics
    | Total Dialogues | Avg. Turns per Dialogue | Avg. Tokens per Dialogue | Avg. Tokens per Utterance |
    |:---------------:|:-----------------------:|:------------------------:|:-------------------------:|
    | 13,118          | 7.9                     | 114,7                    | 14.6                      |

* Evaluation Metrics: BLEU-2, BLEU-4, NIST-2, NIST-4 
<br>I selected the model that achieved the best scores on the above metrics in the validation set.

* Hyperparameters
    | Steps | Batch size | LR0 | Warm up | User-turn Masking |
    |:-------:|:-----------------------:|:------------------------:|:---------:|:---------:|
    | 30,000          | 60                     | 1e-4       | 200      | After 15,000 steps|

<br><br><br>

