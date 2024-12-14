## Execution Method
### 1. Server Execution
#### 1.1 Arguments
There are several arguments for running `demo/server.py`:
* [`-d`, `--device`]: (default: `0`) GPU number to be executed.
* [`-c`, `--config`]: Path of `config.yaml` in the `config` folder.
* [`-m`, `--resume_path`]: Path of pre-trained GPT-2 model.

#### 1.2 Command
```bash
# Execute GPT-2 based model server
python3 demo/server.py --device 0 0 0 --config config/demo_config.yaml --resume_dir outputs/gpt2/diayn_dpo_01 outputs/gpt2/vanilla_dpo_01/ outputs/gpt2/style_sft_01/ --model_keys Diayn_DPO Vanilla_DPO SFT
```
<br><br>

### 2. Prompting Demo Page Execution
```bash
streamlit run demo/front.py
```
<br>
