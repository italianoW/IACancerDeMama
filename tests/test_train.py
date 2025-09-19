import train
import model


def test_treino():
    
    local_model = model.model()
    
    train.train_model(local_model)
     
    assert hasattr(local_model, "_is_fitted") and local_model._is_fitted, "O modelo não foi treinado"
