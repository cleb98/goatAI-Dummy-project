import click
import torch.backends.cudnn as cudnn


from conf import Conf
from trainer import Trainer




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
    # if `exp_name` is None,
    # ask the user to enter it
    if exp_name is None:
        # exp_name = click.prompt('▶ experiment name', default='default')
        exp_name = 'default'
        exp_name = exp_name

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

'''
srun --partition=all_usr_prod --account=tesi_cbellucci --time=00:15:00 --gres=gpu:1 --pty bash
'''
if __name__ == '__main__':
    # main()
    main(exp_name = '1class')

