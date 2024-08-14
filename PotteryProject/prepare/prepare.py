import os
import datetime
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def log(fd,  message, time=True):
    if time:
        message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)

def prepare_logger(params):
    # prepare directory
    make_dir(params.log_dir)
    cur_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    exp_filename = params.exp_name + "_" + params.scheme + "_" + cur_time
    make_dir(os.path.join(params.log_dir, exp_filename))

    logger_path = os.path.join(params.log_dir, exp_filename)
    dataset_dir = os.path.join(params.log_dir, exp_filename, 'dataset')
    ckpt_dir = os.path.join(params.log_dir, exp_filename, 'checkpoint')
    val_dir = os.path.join(params.log_dir, exp_filename, 'val_result')
    test_dir = os.path.join(params.log_dir, exp_filename, 'test_result')

    make_dir(logger_path)
    make_dir(dataset_dir)
    make_dir(ckpt_dir)
    make_dir(val_dir)
    make_dir(test_dir)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train_tb'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val_tb'))

    logger_file = os.path.join(params.log_dir, exp_filename, 'logger.log')

    log_fd = open(logger_file, 'a')
    # log(log_fd, "Experiment: {}".format(exp_filename), False)
    # log(log_fd, "Logger directory: {}".format(logger_path), False)
    # log(log_fd, str(params), False)

    return dataset_dir, val_dir, test_dir, ckpt_dir, log_fd, train_writer, val_writer

def dataset_divider(all_labels_file, train_ratio = 0.8, save_dir = ''):
    all_labels_df = pd.read_csv(all_labels_file, header=None, names=['file_name', 'label'])

    train_df = pd.DataFrame(columns=all_labels_df.columns)
    val_df = pd.DataFrame(columns=all_labels_df.columns)
    test_df = pd.DataFrame(columns=all_labels_df.columns)

    for label in all_labels_df['label'].unique():
        label_df = all_labels_df[all_labels_df['label'] == label]
        train, temp = train_test_split(label_df, test_size=1-train_ratio, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_df = pd.concat([train_df, train], axis=0)
        val_df = pd.concat([val_df, val], axis=0)
        test_df = pd.concat([test_df, test], axis=0)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_file = os.path.join(save_dir, 'train_labels.csv')
    val_file = os.path.join(save_dir, 'val_labels.csv') 
    test_file = os.path.join(save_dir, 'test_labels.csv')

    train_df.to_csv(train_file, index=False, header=False)
    val_df.to_csv(val_file, index=False, header=False)
    test_df.to_csv(test_file, index=False, header=False)

    return train_file, val_file, test_file
