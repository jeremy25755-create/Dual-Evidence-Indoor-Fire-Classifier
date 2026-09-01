import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.dtw_metric import accelerated_dtw
from utils.metrics import metric
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    dict_eq,
    get_all_json_paths,
    save_args_to_json,
    visual,
)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.done = self._done()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _done(self):
        result_path = Path(self.args.results)
        if result_path.exists():
            result_jsons = get_all_json_paths(self.args.results, recursive=True)
            args_dict = vars(self.args)
            for result_json in result_jsons:
                result_json_path = Path(result_json)
                if result_json_path.name != 'args.json':
                    continue
                try:
                    with result_json_path.open('r', encoding='utf-8') as f:
                        result_dict = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(result_dict, dict):
                    continue
                if not set(args_dict).issubset(result_dict):
                    continue
                if dict_eq(result_dict, args_dict):
                    done_time = result_dict.get('done_time', 'unknown')
                    print(f"Already done: {result_json} {done_time}")
                    return True
        return False

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                if getattr(self.args, 'noise', False):
                    # 计算沿维度 t 的均值
                    std_x = torch.std(batch_x, dim=1, keepdim=True)  # 形状变为 [b, 1, c]

                    # 生成噪声
                    noise = std_x * torch.randn_like(batch_x)  # 根据均值生成噪声

                    # 将噪声加到 batch_x 上
                    batch_x = batch_x + noise

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if not self.done:
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

            path = Path(self.args.checkpoints) / setting
            path.mkdir(parents=True, exist_ok=True)
            path = str(path)

            time_now = time.time()

            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    if hasattr(self.args, 'noise'):
                        if self.args.noise:
                            # 计算沿维度 t 的均值
                            std_x = torch.std(batch_x, dim=1, keepdim=True)  # 形状变为 [b, 1, c]

                            # 生成噪声
                            noise = std_x * torch.randn_like(batch_x)  # 根据均值生成噪声

                            # 将噪声加到 batch_x 上
                            batch_x = batch_x + noise
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )

            return self.model

    def test(self, setting, test=0):
        if not self.done:
            test_data, test_loader = self._get_data(flag='test')
            if test:
                checkpoint = getattr(self.args, 'checkpoint', None)
                checkpoint_path = (
                    Path(checkpoint)
                    if checkpoint
                    else Path(self.args.checkpoints) / setting / 'checkpoint.pth'
                )
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint not found: {checkpoint_path}. "
                        "Pass --checkpoint when running config-based inference."
                    )
                print(f'loading model: {checkpoint_path}')
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                )

            preds = []
            trues = []
            folder_path = Path(self.args.results) / 'test_results' / setting
            folder_path.mkdir(parents=True, exist_ok=True)

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                    if self.args.noise:
                        # 计算沿维度 t 的均值
                        std_x = torch.std(batch_x, dim=1, keepdim=True)  # 形状变为 [b, 1, c]

                        # 生成噪声
                        noise = std_x * torch.randn_like(batch_x)  # 根据均值生成噪声

                        # 将噪声加到 batch_x 上
                        batch_x = batch_x + noise
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, str(folder_path / f'{i}.pdf'))

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = Path(self.args.results) / setting
            folder_path.mkdir(parents=True, exist_ok=True)

            # dtw calculation
            if self.args.use_dtw:
                dtw_list = []

                def manhattan_distance(x, y):
                    return np.abs(x - y)

                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1, 1)
                    y = trues[i].reshape(-1, 1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = -999

            mae, mse, rmse, mape, mspe = metric(preds, trues)

            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            summary_path = Path(self.args.results) / 'result_long_term_forecast.txt'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open('a', encoding='utf-8') as summary:
                summary.write(setting + "  \n")
                summary.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
                summary.write('\n\n')

            np.save(folder_path / 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path / 'pred.npy', preds)
            np.save(folder_path / 'true.npy', trues)

            self.args.mse = float(mse)
            self.args.mae = float(mae)
            self.args.done_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_args_to_json(self.args, folder_path / 'args.json')

            return
