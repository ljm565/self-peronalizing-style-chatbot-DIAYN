import gc
import time
import random

import torch
import torch.optim as optim
from torch import distributed as dist

from tools.tokenizers import *
from tools import TrainingLogger, Evaluator
from trainer.loss import DPOLoss
from trainer.build import get_model, get_data_loader, get_tokenizers
from utils import RANK, LOGGER, colorstr, init_seeds
from utils.filesys_utils import *
from utils.training_utils import *
from utils.func_utils import print_samples




class DPOTrainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # Init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if self.is_ddp else 1
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # Path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path
        self.max_len = self.config.max_len
        self.train_metrics, self.val_metrics = self.config.train_metrics, self.config.val_metrics

        # Save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # Make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # Save run args

        # Init tokenizer, model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['train', 'validation', 'test']
        self.tokenizer = get_tokenizers(self.config)
        self.dataloaders = get_data_loader(self.config, self.tokenizer, self.modes, self.is_ddp)
        self.model, self.ref_model = self._init_model(self.config, self.tokenizer, self.mode)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)
        self.evaluator = Evaluator(self.tokenizer)

        # Init criterion, optimizer, etc.
        self.epochs = self.config.epochs
        self.criterion = DPOLoss(self.config.beta)
        if self.is_training_mode:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr0)


    def _init_model(self, config, tokenizer, mode):
        def _resume_model(model, resume_path, device, is_rank_zero):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoints['model'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return model

        # init models
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        model = get_model(config, tokenizer, self.device)
        ref_model = get_model(config, tokenizer, self.device)

        # resume model
        if do_resume:
            model = _resume_model(model, self.resume_path, self.device, config.is_rank_zero)
            ref_model = _resume_model(ref_model, self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
            ref_model = torch.nn.parallel.DistributedDataParallel(ref_model, device_ids=[self.device])
        
        return model, ref_model


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
            
            # Clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.model.train()
        self.ref_model.eval()
        
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # Init progress bar
        if RANK in (-1, 0):
            logging_header = ['DPO Loss', 'Preferred log prob.', 'Non-preferred log prob.', 'Rewards', 'Reward margin']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, batch in pbar:
            self.train_cur_step += 1
            preferred_prompt, nonpreferred_prompt, style_id = batch['preferred_prompt'], batch['nonpreferred_prompt'], batch['style_id']
            batch_size = style_id.size(0)
            preferred_prompt, nonpreferred_prompt, style_id = preferred_prompt.to(self.device), nonpreferred_prompt.to(self.device), style_id.to(self.device)
            
            self.optimizer.zero_grad()
            model_preferred_logits, model_nonpreferred_logits = self.model(preferred_prompt), self.model(nonpreferred_prompt)
            ref_preferred_logits, ref_nonpreferred_logits = self.ref_model(preferred_prompt), self.ref_model(nonpreferred_prompt)
            
            # DPO alignment training
            loss, preferred_log_prob, nonpreferred_log_prob, reward_acc, reward_margins = self.criterion(
                preferred_token=preferred_prompt,
                nonpreferred_token=nonpreferred_prompt,
                model_preferred_logits=model_preferred_logits,
                model_nonpreferred_logits=model_nonpreferred_logits,
                ref_preferred_logits=ref_preferred_logits,
                ref_nonpreferred_logits=ref_nonpreferred_logits,
            )
            
            loss.backward()
            self.optimizer.step()

            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'train_loss': loss.item(),
                       'tr_preferred_log_prob': preferred_log_prob.item(),
                       'tr_non_preferred_log_prob': nonpreferred_log_prob.item(),
                       'tr_reward': reward_acc.item(),
                       'tr_reward_margin': reward_margins.item()},
                )
                loss_log = [loss.item(), preferred_log_prob.item(), nonpreferred_log_prob.item(), reward_acc.item(), reward_margins.item()]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%25s' * 1 + '%25.4g' * len(loss_log)) % msg)
            
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        self.model.eval()
        self.ref_model.eval()

        with torch.no_grad():
            if self.is_rank_zero:
                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                logging_header = ['DPO Loss'] + self.val_metrics
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)

            
                for i, batch in pbar:
                    preferred_prompt, nonpreferred_prompt, style_id = batch['preferred_prompt'], batch['nonpreferred_prompt'], batch['style_id']
                    batch_size = style_id.size(0)
                    preferred_prompt, nonpreferred_prompt, style_id = preferred_prompt.to(self.device), nonpreferred_prompt.to(self.device), style_id.to(self.device)


                    model_preferred_logits, model_nonpreferred_logits = self.model(preferred_prompt), self.model(nonpreferred_prompt)
                    ref_preferred_logits, ref_nonpreferred_logits = self.ref_model(preferred_prompt), self.ref_model(nonpreferred_prompt)
                    
                    # DPO inference
                    loss, preferred_log_prob, nonpreferred_log_prob, reward_acc, reward_margins = self.criterion(
                        preferred_token=preferred_prompt,
                        nonpreferred_token=nonpreferred_prompt,
                        model_preferred_logits=model_preferred_logits,
                        model_nonpreferred_logits=model_nonpreferred_logits,
                        ref_preferred_logits=ref_preferred_logits,
                        ref_nonpreferred_logits=ref_nonpreferred_logits,
                    )

                    # Inference and calculate metrics
                    model = self.model.module if self.is_ddp else self.model
                    response_preds = [model.inference(prompt, self.tokenizer, self.max_len, self.device) for prompt in batch['prompt']]
                    response_gts = [response for response in batch['preferred_response']]
                    metric_results = self.metric_evaluation(response_preds, response_gts)
                    
                    # Logging
                    self.training_logger.update(
                        phase, 
                        epoch + 1,
                        self.train_cur_step,
                        batch_size, 
                        **{'validation_loss': loss.item(),
                           'vl_preferred_log_prob': preferred_log_prob.item(),
                           'vl_non_preferred_log_prob': nonpreferred_log_prob.item(),
                           'vl_reward': reward_acc.item(),
                           'vl_reward_margin': reward_margins.item()},
                        **metric_results
                    )

                    loss_log = [loss.item(), preferred_log_prob.item(), nonpreferred_log_prob.item(), reward_acc.item(), reward_margins.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}'] + loss_log + [metric_results[k] for k in self.val_metrics if k in metric_results])
                    pbar.set_description(('%25s' + '%25.4g' * (len(loss_log) + len(self.train_metrics))) % msg)

                    for gt, pred in zip(response_gts, response_preds):
                        print_samples(gt, pred)


                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(self.wdir, model)
                    self.training_logger.save_logs(self.save_dir)

    
    def metric_evaluation(self, response_pred, response_gt):
        metric_results = {k: 0 for k in self.val_metrics}
        for m in self.val_metrics:
            if m == 'bleu2':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=2)
            elif m == 'bleu4':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=4)
            elif m == 'nist2':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=2)
            elif m == 'nist4':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=4)

        metric_results = {key: value for key, value in metric_results.items() if value != 0}
        return metric_results
    
    
    def chatting(self, query: str, is_first_query=False):
        def _preprocess(query, is_first_query, query_cache=None):
            if is_first_query:
                query = [self.tokenizer.cls_token_id] + self.tokenizer.encode(query) + [self.tokenizer.sep_token_id]
                query_cache = torch.tensor(query, dtype=torch.long).unsqueeze(0).to(self.device)
            else:
                query = self.tokenizer.encode(query) + [self.tokenizer.sep_token_id]
                query_cache = torch.cat([query_cache, torch.tensor(query, dtype=torch.long).unsqueeze(0).to(self.device)], dim=1)
            return query_cache
            
        self.query_cache = None if is_first_query else self.query_cache
        self.query_cache = _preprocess(query, is_first_query, self.query_cache)
        query_done = False
        is_first_query = False

        answer = []
        while 1:
            output = self.model(self.query_cache)
            pred_token = torch.argmax(output[:, -1], dim=-1)
            answer.append(pred_token.item())
            self.query_cache = torch.cat((self.query_cache, pred_token.unsqueeze(1)), dim=1)

            if pred_token == self.tokenizer.sep_token_id:
                answer.pop()
                break
            elif pred_token == self.tokenizer.eos_token_id:
                answer.pop()
                query_done = True
                break
            
            if self.query_cache.size(1) >= self.max_len:
                query_done = True
                break
            
            if query_done:
                self.query_cache = None
                is_first_query = True
        
        answer = self.tokenizer.decode(answer)
        return self.query_cache, answer, query_done, is_first_query
