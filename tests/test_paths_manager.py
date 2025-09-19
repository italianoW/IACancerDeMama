import paths_mngr
import os

def test_paths_contain_data():

  train_dir_exists = os.path.exists(paths_mngr.get_train_path())
  test_dir_exists = os.path.exists(paths_mngr.get_test_path())
  history_exists = os.path.exists(paths_mngr.get_history_path())
  model_exists = os.path.exists(paths_mngr.get_model_path())

  assert train_dir_exists, "'train' directory not found in the dataset path."
  assert test_dir_exists, "'test' directory not found in the dataset path."
  assert history_exists, "'history.pkl' not found in the current directory."
  assert model_exists, "'modelo.h5' not found in the current directory."
