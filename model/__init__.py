from config.structured import ProjectConfig
from .model import ConditionalPointCloudDiffusionModel
from utils.model_utils import set_requires_grad


def get_model(cfg: ProjectConfig):
    model = ConditionalPointCloudDiffusionModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model