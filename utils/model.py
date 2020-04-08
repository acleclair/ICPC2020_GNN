from models.codegnngru import CodeGNNGRU
from models.codegnnbilstm import CodeGNNBiLSTM
from models.codegnndense import CodeGNNDense



def create_model(modeltype, config):
    mdl = None

    if modeltype == 'codegnngru':
        mdl = CodeGNNGRU(config)
    elif modeltype == 'codegnnbilstm':
    	mdl = CodeGNNBiLSTM(config)
    elif modeltype == 'codegnndense':
    	mdl = CodeGNNDense(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
