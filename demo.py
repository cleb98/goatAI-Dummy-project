import cv2
from path import Path

from conf import Conf
# from models.model import DummyModel
from models.model import UNet
from post_processing import PostProcessor
from pre_processing import PreProcessor


def demo(img_path, exp_name):
    # type: (str, str) -> None
    """
    Quick demo of the complete pipeline on a val/test image.

    :param img_path: path of the image you want to test
    :param exp_name: name of the experiment you want to test
    """

    cnf = Conf(exp_name=exp_name)

    # init model and load weights of the best epoch
    model = UNet()
    model.eval()
    model.requires_grad(False)
    model = model.to(cnf.device)
    model.load_w(cnf.exp_log_path / 'best.pth')

    # init pre- and post-processors
    pre_proc = PreProcessor(unsqueeze=True, device=cnf.device)
    post_proc = PostProcessor(out_ch_order='RGB')

    # read image and apply pre-processing
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    y_path = Path(img_path.replace('_x.png', '_y.png'))
    x = pre_proc.apply(img)
    y = cv2.imread(y_path)
    # forward pass and post-processing
    y_pred = model.forward(x)
    img_pred = post_proc.apply(y_pred, save_path=cnf.exp_log_path)

    # # show input and output
    # cv2.imshow('input', img[..., ::-1])
    # cv2.imshow('output', img_pred[..., ::-1])
    # cv2.waitKey(0)

    # save input and output
    cv2.imwrite(str(cnf.exp_log_path / 'input.png'), img[..., ::-1])
    cv2.imwrite(str(cnf.exp_log_path / 'target.png'), y[..., ::-1])
    cv2.imwrite(str(cnf.exp_log_path / 'output.png'), img_pred[..., ::-1])
# to download from server use:
# go to you local folder where you want to download the files using the terminal and type:
# $ scp cbellucci@ailb-login-03.ing.unimore.it:/homes/cbellucci/segmentation/log/default/*.png "C:/Users/cribe/OneDrive/Desktop/GoatAI/log/default"



if __name__ == '__main__':
    __p = Path(__file__).parent / 'dataset' / 'kwasir-seg' / 'val' / '42_x.png'
    try:
        demo(img_path=__p, exp_name='default')
    except Exception as e:
        print(f'Error: {e}')
