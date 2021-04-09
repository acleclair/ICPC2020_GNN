from models.codegnngru import CodeGNNGRU
from models.codegnnbilstm import CodeGNNBiLSTM
from models.codegnndense import CodeGNNDense
from models.attendgru import AttentionGRUModel
from models.ast_attendgru import AstAttentionGRUModel


def create_model(modeltype, config):
    mdl = None

    if modeltype == 'codegnngru':
        mdl = CodeGNNGRU(config)
    elif modeltype == 'codegnnbilstm':
    	mdl = CodeGNNBiLSTM(config)
    elif modeltype == 'codegnndense':
    	mdl = CodeGNNDense(config)
    elif modeltype == 'attendgru':
        mdl = AttentionGRUModel(config)
    elif modeltype == 'ast-attendgru':
        mdl = AstAttentionGRUModel(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
