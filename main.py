import click
import torch.backends.cudnn as cudnn
import urllib.request
import zipfile
import os

from conf import Conf
from trainer import Trainer
from path import Path

def dataset_download():
    url = 'https://drive.google.com/file/d/1dVKbd_G039Ai6VG7HHc640IRDKTKWouT/view?usp=sharing'
    #dowload from google drive a zip file from the url
    urllib.request.urlretrieve(url, 'sample.zip')
    #unzip the file to the folder ./dataset/samples
    with zipfile.ZipFile('sample.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset/samples')
    #remove the zip file
    os.remove('sample.zip')

# --- enable cuDNN benchmark:
# cuDNN benchmarks multiple convolution algorithms and select the fastest.
# This mode is good whenever the input sizes for your network do not vary.
# In case of changing input size, cuDNN will benchmark every time a new
# input size appears, which will probably lead to worse performance. If you
# want perfect reproducibility, you should set this to `False`
cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, conf_file_path, seed):
    # type: (str, str, int) -> None

    #create folder ./dataset/samples if it does not exist
    if not Path('dataset/samples').exists():
        print('dataset not found, downloading...')
        Path('dataset/samples').makedirs()
        dataset_download()
        print('dataset downloaded successfully! in repository ./dataset/samples')

    # if `exp_name` is None,
    # ask the user to enter it
    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    # if `exp_name` contains '!',
    # `log_each_step` becomes `False`
    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(
        cnf_path=conf_file_path, seed=seed,
        exp_name=exp_name, log=log_each_step
    )
    print(f'\n{cnf}')

    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    trainer = Trainer(cnf=cnf)
    trainer.run()


if __name__ == '__main__':
    main()
