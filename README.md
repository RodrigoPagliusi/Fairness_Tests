# Fairness_Tests

This code is based on the article: https://dl.acm.org/doi/10.1145/3636555.3636868

unfair_dataset_generation-main.zip is the original code.

adult_dataset.py configures the adult dataset for use, was totally made by me.

eval.py measures the performance and fairness metrics of the datasets, was totally made by me.

genetic_algorithm.py and genetic_algorithm_base.py are unmodified.

unfairness_metrics.py has little modifications, I simply added the equalized oods metrics and the "label_flip" option in the UNFAIRNESS METRICS list.

run.ps1, i did to run in the PowerShell using VSCode.

In main_paralelo.py, I did little modifications in the class calculation() (just to be able to run it in parallel) and little to no modification in the def run()
I added the def flip_labels and made modifications after the if __name__ == '__main__':, for it to be able to generate many datasets at once.
