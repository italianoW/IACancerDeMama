import train
import model


def test_treino():
    
    local_model = model.model()
    
    trained_model = train.train_model(local_model)
     
    assert hasattr(trained_model, "_is_fitted") and trained_model._is_fitted, "O modelo não foi treinado"
