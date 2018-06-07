from train import *
from keras.layers import average

def ensemble_models(model_collection):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    model_input = [user_input,movie_input]
    pre_avg=[model(model_input) for model in model_collection] 
    Avg=average(pre_avg)
    modelEns = Model(inputs=model_input, outputs=Avg, name='ensemble')
    modelEns.save('ensemble.h5')
    return modelEns


if __name__ == "__main__":
    _mean, _std = mean_std()
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    model_collection = []
    models=['model_84178.h5','model_84507.h5','model_84794.h5']
    for i in models:
        model_collection.append(load_model(i, custom_objects={'rmse':rmse}))
    ensemble_models(model_collection)
    test(out_name = "ens_ans.csv",model_name='ensemble.h5')
