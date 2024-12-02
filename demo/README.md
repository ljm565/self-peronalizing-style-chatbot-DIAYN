## Execution Method
### 1. Server Execution
#### 1.1 Arguments
There are several arguments for running `demo/server.py`:
* [`-d`, `--device`]: (default: `0`) GPU number to be executed.
* [`-c`, `--config`]: Path of `config.yaml` in the `config` folder.
* [`-m`, `--resume_path`]: Path of pre-trained GPT-2 model.

#### 1.2 Command
```bash
# Execute GPT-2 server
python3 demo/server.py -d cpu -c config/config.yaml -m checkpoint/model_epoch:83_step:15437_metric_best.pt
```
<br><br>

### 2. Prompting Demo Page Execution
```bash
streamlit run demo/front.py
```
<br>
