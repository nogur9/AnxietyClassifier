def plot_timeline_within_trial(self):
    subjects = list(sorted(set(self.fixation_dataset.Subject)))
    trials = [set(self.fixation_dataset.Trial[self.fixation_dataset.Subject == i]) for i in subjects]

    ds = self.fixation_dataset[(self.fixation_dataset.AOI_Group == "D")]
    ns = self.fixation_dataset[(self.fixation_dataset.AOI_Group == "N")]
    ws = self.fixation_dataset[(self.fixation_dataset.AOI_Group == "White Space")]

    plt.scatter(ds.Fixation_Start, ds.Fixation_Duration, c='b')
    plt.scatter(ns.Fixation_Start, ns.Fixation_Duration, c='r')
    plt.scatter(ws.Fixation_Start, ws.Fixation_Duration, c='g')
    plt.show()
    plt.close()
