import os
import re
from collections import defaultdict
import pandas as pd
from matplotlib.pyplot import subplots_adjust, gca, margins
from matplotlib.ticker import NullLocator

cur_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(cur_dir, "logs")


class KELogParser(object):
    def __init__(self):
        pass

    def parse_all(self):
        """
        2019-10-08 13:45:39,248 - INFO trainer.py run 103 - traffic_500 Analogy train epoch_num: 3, global_step: 100, loss: 0.694, mr: 40.625, mrr: 0.125, Hit@10: 0.188, Hit@3: 0.156, Hit@1: 0.062
        2019-10-08 13:45:41,947 - INFO trainer.py run 113 - traffic_500 Analogy valid epoch_num: 3, global_step: 111, loss: 0.692, mr: 55.785, mrr: 0.036, hit_10: 0.090, hit_3: 0.005, hit_1: 0.000
        """
        log_file = os.path.join(log_dir, "all_all_train.log")
        train_patten = re.compile(
            "trainer.py run 103 - (.*?) (.*?) train epoch_num: (\d+), global_step: (\d+), loss: (\d+\.\d+), "
            "mr: (\d+\.\d+), mrr: (\d+\.\d+), Hit@10: (\d+\.\d+), Hit@3: (\d+\.\d+), Hit@1: (\d+\.\d+)")
        valid_patten = re.compile(
            "trainer.py run 113 - (.*?) (.*?) valid epoch_num: (\d+), global_step: (\d+), loss: (\d+\.\d+), "
            "mr: (\d+\.\d+), mrr: (\d+\.\d+), hit_10: (\d+\.\d+), hit_3: (\d+\.\d+), hit_1: (\d+\.\d+)")
        keys = ["epoch_num", "global_step", "loss", "mr", "mrr", "hit_10", "hit_3", "hit_1"]
        # "dataset", "model_name",
        metrics = {"train": defaultdict(lambda: defaultdict(lambda: {k: [] for k in keys})),
                   "valid": defaultdict(lambda: defaultdict(lambda: {k: [] for k in keys}))}
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                train_result = train_patten.findall(line)
                if train_result:
                    dataset, model_name, epoch_num, global_step, loss, mr, mrr, hit_10, hit_3, hit_1 = train_result[0]
                    # metrics["train"][dataset]["model_name"].append(model_name)
                    metrics["train"][dataset][model_name]["epoch_num"].append(int(epoch_num))
                    metrics["train"][dataset][model_name]["global_step"].append(int(global_step))
                    metrics["train"][dataset][model_name]["loss"].append(float(loss))
                    metrics["train"][dataset][model_name]["mr"].append(float(mr))
                    metrics["train"][dataset][model_name]["mrr"].append(float(mrr))
                    metrics["train"][dataset][model_name]["hit_10"].append(float(hit_10))
                    metrics["train"][dataset][model_name]["hit_3"].append(float(hit_3))
                    metrics["train"][dataset][model_name]["hit_1"].append(float(hit_1))

                valid_result = valid_patten.findall(line)
                # if "trainer.py run 113 -" in line:
                #     import ipdb
                #     ipdb.set_trace()
                if valid_result:
                    dataset, model_name, epoch_num, global_step, loss, mr, mrr, hit_10, hit_3, hit_1 = valid_result[0]
                    # metrics["valid"][dataset]["model_name"].append(model_name)
                    metrics["valid"][dataset][model_name]["epoch_num"].append(int(epoch_num))
                    metrics["valid"][dataset][model_name]["global_step"].append(int(global_step))
                    metrics["valid"][dataset][model_name]["loss"].append(float(loss))
                    metrics["valid"][dataset][model_name]["mr"].append(float(mr))
                    metrics["valid"][dataset][model_name]["mrr"].append(float(mrr))
                    metrics["valid"][dataset][model_name]["hit_10"].append(float(hit_10))
                    metrics["valid"][dataset][model_name]["hit_3"].append(float(hit_3))
                    metrics["valid"][dataset][model_name]["hit_1"].append(float(hit_1))
                    # import ipdb
                    # ipdb.set_trace()
        # import ipdb
        # ipdb.set_trace()
        return dict(metrics)


def plot():
    import matplotlib.pyplot as plt
    metrics = KELogParser().parse_all()

    # import ipdb
    # ipdb.set_trace()

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
    def draw(metrics, mode="valid", dataset="", max_step_num=None):
        for mode, dataset_metrics_dict in metrics.items():
            print(mode)
            if mode != "valid":
                continue
            y_keys = ["mr", "mrr", "hit_10"]  # , "hit_3", "hit_1"]
            x_key = "epoch_num"
            lines = []
            lines_labels = []
            fig, axs = plt.subplots(2, 3)
            title = ["dataset", "model_name"] + y_keys
            data_dict = {_key: [] for _key in title + [x_key]}
            for dataset, model_name_metrics_dict in dataset_metrics_dict.items():
                # i = {"lawdata": 0, "traffic_500": 1}[dataset]
                markers = ["1", "2", "3", "4", "s", "p", "h", "H", "8", "x", "*", "o", "d"]
                for k, (model_name, metrics_dict) in enumerate(model_name_metrics_dict.items()):
                    x = metrics_dict[x_key]
                    x = x[:300][::20]
                    data_dict[x_key].extend(x)
                    data_dict["dataset"].extend([dataset] * len(x))
                    data_dict["model_name"].extend([model_name] * len(x))
                    for j, y_key in enumerate(y_keys):  # 每个key对应一个指标的图
                        print("{}-{}-{}-{}".format(dataset, mode, model_name, y_key))
                        y = metrics_dict[y_key]
                        y = y[:300][::20]
                        data_dict[y_key].extend(y)
                        #
                        i = {"lawdata": 0, "traffic_500": 1}[dataset]
                        ax = axs[i][j]
                        line = ax.plot(x, y, label={"TransformerKB": "TransKB"}.get(model_name, model_name),
                                       marker=markers[k])[0]
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel({"hit_10": "Hit@10", "mr": "MR", "mrr": "MRR"}.get(y_key, y_key))
                        ax.grid(linestyle='-.')
                        if model_name not in lines_labels:
                            lines.append(line)
                            lines_labels.append({"TransformerKB": "TransKB"}.get(model_name, model_name))
            for k, v in data_dict.items():
                print(k, len(v))
            extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            gca().set_axis_off()
            # subplots_adjust(top=1, bottom=0, right=1, left=0,
            #                 hspace=0, wspace=0)
            # subplots_adjust(hspace=0, wspace=0)
            # margins(0, 0)
            # gca().xaxis.set_major_locator(NullLocator())
            # gca().yaxis.set_major_locator(NullLocator())

            fig.savefig("./kge.pdf", format='pdf',
                        # transparent=True, dpi=300,
                        # pad_inches=0,
                        # bbox_inches=extent
                        )
            fig.legend(lines[:11], lines_labels[:11], loc='upper center', ncol=11)

            pd.DataFrame(data_dict).to_csv(f"./kge_data.csv", index=False, sep=",")
            # plt.tight_layout()

            plt.show()

    # joint_y_keys = ["ner_loss", "re_loss", "transe_loss", "joint_loss"]
    # ner_y_keys = ["global_step", "ner_loss", "acc", "precission", "recall", "f1"]
    # re_y_keys = ["global_step", "re_loss", "acc", "precission", "recall", "f1"]

    max_step_num = None
    draw(metrics, max_step_num=max_step_num)


if __name__ == "__main__":
    plot()
