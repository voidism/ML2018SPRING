from keras.models import load_model
from keras import layers
from keras.models import Model
from keras.layers import Input

def ensemble_models(models=['my_model_67901.h5','my_model_68013.h5','my_model_67539.h5']):
    model_collection = []
    for i in models:
        model_collection.append(load_model(i))
    model_input = Input(shape=model_collection[0].input_shape[1:])
    pre_avg=[model(model_input) for model in model_collection] 
    Avg=layers.average(pre_avg)
    modelEns = Model(inputs=model_input, outputs=Avg, name='ensemble')
    modelEns.save('ensemble.h5')
    return modelEns

if __name__ == "__main__":
    ensemble_models()