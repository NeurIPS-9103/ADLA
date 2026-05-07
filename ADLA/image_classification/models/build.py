
from .adla_deit import adla_deit_tiny, adla_deit_small, adla_deit_base


def build_model(config):
    model_type = config.MODEL.TYPE
                              
    if model_type in ['adla_deit_tiny', 'adla_deit_small', 'adla_deit_base']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE)')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
