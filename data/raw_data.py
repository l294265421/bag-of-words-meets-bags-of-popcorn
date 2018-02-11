import pandas as pd

base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Bag of Words Meets Bags of Popcorn\\'

train_file_name = 'labeledTrainData.tsv'
test_file_name = 'testData.tsv'

train_df = pd.read_table(base_dir + train_file_name)
test_df = pd.read_table(base_dir + test_file_name)