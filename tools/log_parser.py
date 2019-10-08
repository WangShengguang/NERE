import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)


class LogParser(object):
    def __init__(self):
        pass

    def joint_parse(self):
        """
        2019-10-08 12:49:27,230 - INFO joint_trainer.py run 140 - joint_0.1BERTCRF_0.9BERTMultitask_0.001TransE joint train NER global_step:7166 loss: 0.0011, acc: 0.7189, precision: 0.0980, recall: 0.1887, f1: 0.1290
        2019-10-08 12:49:27,238 - INFO joint_trainer.py run 145 - joint_0.1BERTCRF_0.9BERTMultitask_0.001TransE joint train RE global_step:7166 loss: 0.0000, acc: 1.0000, precision: 0.5000, recall: 0.5000, f1: 1.0000
        2019-10-08 12:49:27,238 - INFO joint_trainer.py run 148 - * joint global_step:7166, ner_loss: 0.0011, re_loss: 0.0000, transe_loss: 0.0000，joint_loss: 0.0001
        """
        ner_patten = re.compile("run 140 - (.*) joint train NER global_step:(\d+) loss: (\d+\.\d+), "
                                "acc: (\d+\.\d+), precision: (\d+\.\d+), recall: (\d+\.\d+), f1: (\d+\.\d+)")

        re_patten = re.compile("run 145 - (.*) joint train RE global_step:(\d+) loss: (\d+\.\d+), "
                               "acc: (\d+\.\d+), precision: (\d+\.\d+), recall: (\d+\.\d+), f1: (\d+\.\d+)")

        joint_patten = re.compile("run 148 - \* joint global_step:(\d+), "
                                  "ner_loss: (\d+\.\d+), re_loss: (\d+\.\d+), transe_loss: (\d+\.\d+)，joint_loss: (\d+\.\d+)")

        ner_logs = defaultdict(lambda: {"global_step": [], "ner_loss": [],
                                        "acc": [], "precission": [], "recall": [], "f1": []})
        re_logs = defaultdict(lambda: {"global_step": [], "re_loss": [],
                                       "acc": [], "precission": [], "recall": [], "f1": []})
        joint_logs = defaultdict(lambda: {"global_step": [],
                                          "ner_loss": [], "re_loss": [], "transe_loss": [], "joint_loss": []})
        for log_name in ["joint_train.log.1", "joint_train.log"]:
            log_file = os.path.join(root_dir, log_name)
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    ner_result = ner_patten.findall(line)
                    if ner_result:
                        model_name, global_step, ner_loss, acc, precission, recall, f1 = ner_result[0]
                        ner_logs[model_name]["global_step"].append(int(global_step))
                        ner_logs[model_name]["ner_loss"].append(float(ner_loss))
                        ner_logs[model_name]["acc"].append(float(acc))
                        ner_logs[model_name]["precission"].append(float(precission))
                        ner_logs[model_name]["recall"].append(float(recall))
                        ner_logs[model_name]["f1"].append(float(f1))
                    re_result = re_patten.findall(line)
                    if re_result:
                        model_name, global_step, re_loss, acc, precission, recall, f1 = re_result[0]
                        re_logs[model_name]["global_step"].append(int(global_step))
                        re_logs[model_name]["re_loss"].append(float(re_loss))
                        re_logs[model_name]["acc"].append(float(acc))
                        re_logs[model_name]["precission"].append(float(precission))
                        re_logs[model_name]["recall"].append(float(recall))
                        re_logs[model_name]["f1"].append(float(f1))
                    joint_result = joint_patten.findall(line)
                    if joint_result:
                        global_step, ner_loss, re_loss, transe_loss, joint_loss = joint_result[0]
                        joint_logs[model_name]["global_step"].append(int(global_step))
                        joint_logs[model_name]["ner_loss"].append(float(ner_loss))
                        joint_logs[model_name]["re_loss"].append(float(re_loss))
                        joint_logs[model_name]["transe_loss"].append(float(transe_loss))
                        joint_logs[model_name]["joint_loss"].append(float(joint_loss))
                    # if "train NER" in line or "train RE" in line or "run 148 - " in line:
                    #     print(line)
                    #     import ipdb
                    #     ipdb.set_trace()
        # for logs in
        #     for mode_name, metrics_dict in logs.items():

        return dict(ner_logs), dict(re_logs), dict(joint_logs)


def plot():
    import matplotlib.pyplot as plt
    import seaborn as sns
    ner_logs, re_logs, joint_logs = LogParser().joint_parse()

    # plt.figure(figsize=(10, 5))
    # plt.xlabel("step")
    # plt.ylabel("loss")
    # # plt.plot(steps, ner_loss,label="")
    # plt.plot_date(steps, ner_loss, '-', label="ner_loss")
    # plt.plot_date(steps, re_loss, '-', color='r', label="re_loss")
    # plt.plot_date(steps, transe_loss, '-', label="transe_loss")
    # plt.plot_date(steps, joint_loss, '-', label="joint_loss")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # print(loss_info)

    # poly = np.polyfit(list_x, list_y, 5)

    # sns.lineplot(x="step", y="value", hue="variable", data=pd.melt(df_data, ['step']))
    def draw(logs, max_step_num=None):
        for mode_name, metrics_dict in logs.items():
            global_steps = metrics_dict["global_step"]
            metrics_dict.pop("global_step")
            df_data = pd.DataFrame(metrics_dict)
            if isinstance(max_step_num, int):
                for i, step in enumerate(global_steps):
                    if step >= max_step_num:
                        df_data = df_data[:i]
                        print(df_data)
                        # import ipdb
                        # ipdb.set_trace()
                        break
            print(mode_name)
            # df_data.rolling(window=10000).mean()
            for i, key in enumerate(metrics_dict.keys()):
                # plt.subplot(i)
                plt.subplot(2, 4, i + 1)
                # sns.lineplot(x="global_step", y=key, data=df_data)
                # plt.plot(df_data["global_step"], df_data[key], 'bo-', markevery=100)

                # poly = np.polyfit(df_data["global_step"], df_data[key], 15)
                # poly_y = np.poly1d(poly)(df_data["global_step"])

                avg_interval = 500
                total_len = len(global_steps)
                # total_len = 10000
                x = np.arange(0, total_len, avg_interval)
                y = []
                for start in range(0, total_len, avg_interval):
                    end = start + avg_interval if start + avg_interval < total_len else total_len
                    y.append(np.mean(df_data[key][start:end]))
                plt.xlabel("step")
                plt.ylabel(key)
                plt.plot(x, y)

                # plt.plot(df_data["global_step"], poly_y)
                # df_data[key] = poly_y
                # sns.lineplot(x="global_step", y=key, data=df_data)

                # sns.lmplot('global_step', key, data=df_data, hue='global_step', ci=None, order=2, truncate=True)
            plt.show()

    # joint_y_keys = ["ner_loss", "re_loss", "transe_loss", "joint_loss"]
    # ner_y_keys = ["global_step", "ner_loss", "acc", "precission", "recall", "f1"]
    # re_y_keys = ["global_step", "re_loss", "acc", "precission", "recall", "f1"]

    max_step_num = None
    draw(joint_logs, max_step_num=max_step_num)


if __name__ == "__main__":
    plot()
