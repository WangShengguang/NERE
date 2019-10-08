import os
import re
import pandas as pd

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)


def get_loss_history(log_name):
    """
    :return:
    """
    log_file = os.path.join(root_dir, log_name)
    loss_patten = re.compile(".*global_step:(?P<global_step>\d+), "
                             "ner_loss: (?P<ner_loss>\d\.\d{4}), re_loss: (?P<re_loss>\d\.\d{4}), "
                             "transe_loss: (?P<transe_loss>\d\.\d{4})，"
                             "joint_loss: (?P<joint_loss>\d\.\d{4})")
    loss_info = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "global_step" not in line:
                continue
            match_group = loss_patten.search(line)
            # _loss_info = loss_patten.findall(line)
            if match_group:
                global_step = match_group.group("global_step")
                ner_loss = match_group.group("ner_loss")
                re_loss = match_group.group("re_loss")
                transe_loss = match_group.group("transe_loss")
                joint_loss = match_group.group("joint_loss")
                loss_info.append([int(global_step),
                                  float(ner_loss), float(re_loss), float(transe_loss),
                                  float(joint_loss)])
    return loss_info


def plot():
    import matplotlib.pyplot as plt
    log_name = "joint_train.log"
    loss_info = get_loss_history(log_name)
    ner_loss, re_loss, transe_loss, joint_loss = [], [], [], []
    steps = []
    for global_step, _ner_loss, _re_loss, _transe_loss, _joint_loss in loss_info:
        steps.append(global_step)
        ner_loss.append(_ner_loss)
        re_loss.append(_re_loss)
        transe_loss.append(_transe_loss)
        joint_loss.append(_joint_loss)
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

    import seaborn as sns
    df_data = pd.DataFrame({"step": steps,
                            "ner_loss": ner_loss, "re_loss": re_loss, "transe_loss": transe_loss,
                            "joint_loss": joint_loss})
    # sns.lineplot(x="step", y="value", hue="variable", data=pd.melt(df_data, ['step']))
    plt.figure()
    sns.lineplot(x="step", y="ner_loss", data=df_data)
    plt.figure()
    sns.lineplot(x="step", y="re_loss", data=df_data)
    plt.figure()
    sns.lineplot(x="step", y="transe_loss", data=df_data)
    plt.figure()
    sns.lineplot(x="step", y="joint_loss", data=df_data)

    plt.show()


if __name__ == "__main__":
    plot()
