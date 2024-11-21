import cv2
from path import Path

from conf import Conf
from models.model import DummyModel
from post_processing import PostProcessor
from pre_processing import PreProcessor


def demo(img_path, exp_name):
    # type: (str, str) -> None
    """
    Quick demo of the complete pipeline on a val/test image.

    :param img_path: path of the image you want to test
    :param exp_name: name of the experiment you want to test
    """
    try:
        cv2.imshow('input', cv2.imread(img_path))
        cv2.waitKey(0)
    except Exception as e:
        print(f'Error trying to show input image: {e}')

    cnf = Conf(exp_name=exp_name)

    # init model and load weights of the best epoch
    model = DummyModel()
    model.eval()
    model.requires_grad(False)
    model = model.to(cnf.device)
    model.load_w(cnf.exp_log_path / 'best.pth')

    # init pre- and post-processors
    pre_proc = PreProcessor(unsqueeze=True, device=cnf.device)
    post_proc = PostProcessor(out_ch_order='RGB')

    # read image and apply pre-processing
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    x = pre_proc.apply(img)

    # forward pass and post-processing
    y_pred = model.forward(x)
    img_pred = post_proc.apply(y_pred)

    # show input and output
    cv2.imshow('input', img[..., ::-1])
    cv2.imshow('output', img_pred[..., ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    __p = Path(__file__).parent / 'dataset' / 'samples' / 'val' / '20_x.png'
    try:
        demo(img_path=__p, exp_name='default')
    except Exception as e:
        print(f'Error: {e}')
