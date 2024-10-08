import json
import os
import numpy as np
import itertools
import imageio
import tempfile
import pathlib
import torch
from contextlib import contextmanager, nullcontext
from tests.eval import fm
from utils.iwe import warp_events, compute_pol_iwe
from utils.reconstruction import ImageReconstructor
import cv2


class Logger:
    def __init__(self, config, test_name):
        self.config = config
        self.test_name = test_name
        self.metrics = dict()
        self.parse_log_fields()

    def parse_log_fields(self):
        def parse_index(index):
            for i, idx in enumerate(index):
                if isinstance(idx, str):
                    assert '-' in idx
                    a, b = idx.split('-')
                    index[i] = list(range(int(a), int(b) + 1))
            parsed_list = []
            for i in index:
                # this is because i can be type of omegaconf.listconfig
                if hasattr(i, '__len__'):
                    parsed_list.extend(i)
                else:
                    parsed_list.append(i)
            return parsed_list

        if not hasattr(self.config, "saved_tensors"):
            self.config.saved_tensors = None
        if self.config.saved_tensors is None:
            return
        for field, seqs in self.config.saved_tensors.items():
            if seqs is None:
                continue
            else:
                for seq, idx in seqs.items():
                    if idx is None:
                        continue
                    else:
                        self.config.saved_tensors[field][seq] = parse_index(
                            idx)

    class SeqLogger:
        def __init__(self, config, log_path, seq_name):
            self.config = config
            self.path = log_path
            self.seq_name = seq_name

        def log_tensors(self, batch, out, idx, save_all=False):
            if isinstance(batch, list):
                batch = batch[-1]
            assert isinstance(out, dict)
            # if 'save_submission' in batch:
            #     if isinstance(batch['save_submission'], torch.Tensor):
            #         batch['save_submission'] = batch['save_submission'].cpu().item()
            #     if batch['save_submission'] or save_all:
            #         self.save_submission(out['final_prediction'],
            #                              batch['file_index'].cpu().item())

            # self.save_iwe(out['final_prediction'], batch['events_old'], batch['file_index'].cpu().item())
            self.save_gray(out['final_prediction'], batch['events_old'], batch['file_index'].cpu().item())

            if getattr(self.config, "saved_tensors", None) is None:
                return
            for key, value in itertools.chain(batch.items(), out.items()):
                if key in self.config.saved_tensors:
                    if self.tblogged(self.config.saved_tensors[key], idx):
                        self.save_tensor(key, value, idx)

        def tblogged(self, logging_index, idx):
            if logging_index is None:
                return True
            elif self.seq_name not in logging_index:
                return False
            elif logging_index[self.seq_name] is None:
                return True
            else:
                return idx in logging_index[self.seq_name]

        def save_submission(self, flow, file_idx):
            os.makedirs(os.path.join(self.path, "submission"), exist_ok=True)
            if isinstance(flow, torch.Tensor):
                flow = flow.cpu().numpy()
            assert flow.shape == (1, 2, 480, 640)
            flow = flow.squeeze()
            _, h, w = flow.shape
            scaled_flow = np.rint(
                flow * 128 + 2 ** 15).astype(np.uint16).transpose(1, 2, 0)
            flow_image = np.concatenate((scaled_flow, np.zeros((h, w, 1),
                                                               dtype=np.uint16)), axis=-1)
            imageio.imwrite(os.path.join(self.path, "submission", f"{file_idx:06d}.png"),
                            flow_image, format='PNG-FI')

        def save_tensor(self, key, value, idx):
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
                np.save(os.path.join(self.path, f"{key}_{idx:05d}.npy"), value)
                return
            if isinstance(value, np.ndarray):
                np.save(os.path.join(self.path, f"{key}_{idx:05d}.npy"), value)
                return

        def save_iwe(self, flow, events, file_idx):
            iwe = warp_events(flow, events, (480, 640), flow_scaling=1, round_idx=False)
            # iwe = np.clip(iwe, 0, 255)
            iwe = (iwe - np.min(iwe)) / (np.max(iwe) - np.min(iwe))
            iwe = iwe * 255
            iwe = iwe.astype(np.uint8)

            # 直方图均衡化
            iwe = cv2.equalizeHist(iwe)
            # 中值滤波
            # iwe = cv2.medianBlur(iwe, 3)

            save_dir = os.path.join(self.path, "iwe")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imageio.imwrite(os.path.join(save_dir, f"{file_idx:06d}.jpg"),
                            iwe.astype(np.uint8))

        def save_gray(self, flow, events, file_idx):
            if not os.path.exists(os.path.join(self.path, "iwe")):
                os.makedirs(os.path.join(self.path, "iwe"), exist_ok=True)
                os.makedirs(os.path.join(self.path, "gray"), exist_ok=True)
            image_reconstructor = ImageReconstructor(flow)
            # [y,x,t,p]-->[t,y,x,p]
            events = events[:, :, [2, 0, 1, 3]]
            iwe, x_best = image_reconstructor.image_rec_from_events_l2(events, reg_weight=3e-1)
            x_best = x_best * 255
            x_best = x_best.astype(np.uint8)
            iwe = iwe * 255
            iwe = iwe.astype(np.uint8)
            try:
                imageio.imwrite(os.path.join(self.path, "iwe", f"{file_idx:06d}.jpg"), iwe)
            except Exception as e:
                print(e)
            try:
                imageio.imwrite(os.path.join(self.path, "gray", f"{file_idx:06d}.jpg"), x_best)
            except Exception as e:
                print(e)

        def log_metrics(self, results):
            self.results = results

        def compute_statistics(self):
            self.metric_stats = dict()
            for quantity, metrics in self.results.items():
                self.metric_stats[quantity] = dict()
                for metric, list_fm in metrics.items():
                    assert isinstance(list_fm[0], fm)
                    self.metric_stats[quantity][metric] = dict()
                    metric_list = list(map(lambda x: x.value, list_fm))
                    self.metric_stats[quantity][metric]['avg'] = sum(
                        metric_list) / len(metric_list)

    @contextmanager
    def record_sequence(self, seq_name, log_path):
        try:
            os.makedirs(os.path.join(log_path, seq_name), exist_ok=True)
            self.seq_logger = self.SeqLogger(
                self.config, os.path.join(log_path, seq_name), seq_name)
            yield self.seq_logger
        finally:
            self.seq_logger.compute_statistics()
            self.copy_metrics(self.seq_logger.results, self.seq_logger.metric_stats,
                              seq_name)

    @contextmanager
    def log_test(self, model):
        log_dir = self.config.get('save_dir', None)
        with tempfile.TemporaryDirectory(prefix='id2log', dir='/tmp') \
                if log_dir is None else nullcontext(pathlib.Path(log_dir)) as wd:
            try:
                yield wd
            finally:
                self.dump(model, wd)

    def copy_metrics(self, results, stats, seq_name):
        self.metrics[seq_name] = dict()
        self.metrics[seq_name]['results'] = results
        self.metrics[seq_name]['stats'] = stats

    def dump(self, model, wd):

        dict_to_dump = {"meta": {
            "test_name": self.test_name,
        }, **self.metrics}
        json.dump(dict_to_dump, open(os.path.join(
            wd, 'metrics.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(wd, 'model.pt'))

    def summary(self):
        summary_flat = dict()
        summary = dict()
        for seq_name, results in self.metrics.items():
            summary[seq_name] = dict()
            for quantity, metrics in results["stats"].items():
                summary[seq_name][quantity] = metrics
                for metric, stats in metrics.items():
                    for stat, value in stats.items():
                        assert isinstance(value, (int, float))
                        summary_flat['-'.join([seq_name,
                                               quantity, metric, stat])] = value
        return summary
