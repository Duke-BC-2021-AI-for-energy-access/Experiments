from dataset import Dataset
import os
import subprocess
import itertools
import argparse
import pwd

repo_path = os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/") + 'repro_bass/'

parser = argparse.ArgumentParser()

#os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/")+'MW_batch_size_8/'
#out_path- /scratch/public/results/25background_experiment/
parser.add_argument('--out_path', default='/scratch/public/jitter/wt/experiment_results/')
#train_path- /scratch/public/txt_files/25background_experiment/
parser.add_argument('--train_path', default='/scratch/public/jitter/wt/experiments/')
parser.add_argument('--experiment')
parser.add_argument('--experiment_name')
parser.add_argument('--val_path', default='/scratch/public/jitter/wt/experiments/Test/')
parser.add_argument('--epochs', default='300')
parser.add_argument('--device')
parser.add_argument('--supplemental_batch_size', default='0')

args = parser.parse_args()

out_path = args.out_path
train_path = args.train_path
val_path = args.val_path
experiment = args.experiment
experiment_name = args.experiment_name
supplemental_batch_size =  args.supplemental_batch_size

domains = ["EM", "SW"]

combinations = list(itertools.product(domains, repeat=2))

#Adds filter as necessary

#def containsDuplicate(element):
#  return element[0] != element[1]

#reg_combos = list(filter(containsDuplicate, combinations))

def optimalRatioFilter(element):
  return element[0] != element[1]

optimal_ratio_combos = list(filter(optimalRatioFilter, combinations))

experiment_path = os.path.join(train_path, experiment + "/")
experiment_out_path = os.path.join(out_path, experiment_name + "/")

datasets = []
for combo in optimal_ratio_combos:
  print(combo)
  for i in range(0,4):
    if not experiment == "Baseline":
      dataset_string = """Dataset(img_txt=experiment_path+'Train_{src}_Test_{dst}_Images.txt',
                        lbl_txt=experiment_path+'Train_{src}_Test_{dst}_Labels.txt',
                        out_dir=experiment_out_path+'t_{src}_v_{dst}_{i}/',
                        img_txt_val=val_path+'{dst}_Images.txt',
                        lbl_txt_val=val_path+'{dst}_Labels.txt',
                        img_txt_supplement=experiment_path+'Train_{src}_Test_{dst}_Supplement_Images.txt',
                        lbl_txt_supplement=experiment_path+'Train_{src}_Test_{dst}_Supplement_Labels.txt')""".format(src=combo[0],dst=combo[1],i=i)
    else:
      dataset_string = """Dataset(img_txt=experiment_path+'Train_{src}_Test_{dst}_Images.txt',
                  lbl_txt=experiment_path+'Train_{src}_Test_{dst}_Labels.txt',
                  out_dir=experiment_out_path+'t_{src}_v_{dst}_{i}/',
                  img_txt_val=val_path+'{dst}_Images.txt',
                  lbl_txt_val=val_path+'{dst}_Labels.txt')""".format(src=combo[0],dst=combo[1],i=i)
    datasets.append(eval(dataset_string))

#Could create some variable that does not use every trial

for trial in datasets:
  subprocess.run(['python', 'run_save_train_test.py',
                    '--img_list', trial.get_img_txt(), 
                    '--lbl_list', trial.get_lbl_txt(),
                    '--epochs', args.epochs,
                    '--out_dir', trial.get_out_dir(),
                    '--img_list_val', trial.get_img_txt_val(),
                    '--lbl_list_val', trial.get_lbl_txt_val(),
                    '--version', 'v2',
                    '--device', args.device,
                    '--experiment', experiment,
                    '--supplement_batch_size', supplemental_batch_size,
                    '--img_list_supplement', trial.get_img_txt_supplement(),
                    '--lbl_list_supplement', trial.get_lbl_txt_supplement()])
