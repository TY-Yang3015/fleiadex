import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LogReader:
    """
    the reader for the custom log, you are advised to use this in a
    jupyter notebook. see the example for usage.

    >>> reader = LogReader()
    >>> reader.read_log(log_dir).make_plot(save_dir).save_data(save_dir)

    """

    def __init__(self):
        self.step_list = []
        self.loss_list = []
        self.mse_list = []
        self.kld_list = []
        self.aux_list = []
        self.__has_log__ = False

        self.set_style()

    @classmethod
    def set_style(cls):
        sns.set_theme(context="paper", style="darkgrid", palette="husl")

    def read_log(self, path: str):
        """
        the function to read the log file.

        :param path: the path of the log file.

        :return: the LogReader instance.
        """
        f = open(path)
        for line in f:
            line = line.split("-")[-1]
            if line.replace(" ", "").startswith("step"):
                step, loss, mse, kld = line.split(",")
                self.step_list.append(int(step.split(":")[1]))
                self.loss_list.append(float(loss.split(":")[1]))
                self.mse_list.append(float(mse.split(":")[1]))
                self.kld_list.append(float(kld.split(":")[1]))
            elif line.replace(" ", "").startswith("auxiliary"):
                aux = float(line.split(":")[1])
                self.aux_list.append(aux)

        self.step_list = np.array(self.step_list)
        self.loss_list = np.array(self.loss_list)
        self.mse_list = np.array(self.mse_list)
        self.kld_list = np.array(self.kld_list)
        self.aux_list = np.array(self.aux_list)

        if len(self.aux_list) > 0:
            print("auxiliary SSIM data received.")

        self.__has_log__ = True
        print("read success.")

        return self

    def make_plot(self, save_dir, dpi=500):
        """
        the function to make the plots of the recorded data and save.

        :param save_dir: the path of the save directory.
        :param dpi: the dpi (proportional to resolution) of the plot.

        :return: the LogReader instance.
        """
        if self.__has_log__ is False:
            raise ValueError("please run the read_log method first.")

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.step_list, self.loss_list, label="loss")
        ax.plot(self.step_list, self.mse_list, label="mse")
        ax.plot(self.step_list, self.kld_list, label="kld")

        if len(self.aux_list) != 0:
            try:
                ax1 = ax.twinx()
                ax1.plot(self.step_list[1:], self.aux_list, label="ssim")
                ax1.grid(False)
                ax1.legend(loc="upper right", fontsize=12)
                ax1.set_ylim(-0.1, 1)
            except Exception as e:
                print("warning:", e)

        ax.legend(loc="upper left", fontsize=12)
        ax.set_yscale("log")

        fig.savefig(save_dir + "/log_plot.png", dpi=dpi)

        return self

    def save_data(self, save_dir):
        """
        the function to save the recorded data in .npy format.

        :param save_dir: the path of the save directory.

        :return: the LogReader instance.
        """
        if self.__has_log__ is False:
            raise ValueError("please run the read_log method first.")

        np.save(save_dir + "/step.npy", self.step_list)
        np.save(save_dir + "/loss.npy", self.loss_list)
        np.save(save_dir + "/mse.npy", self.mse_list)
        np.save(save_dir + "/kld.npy", self.kld_list)
        if len(self.aux_list) > 0:
            np.save(save_dir + "/ssim.npy", self.aux_list)

        print("data saved.")

        return self
