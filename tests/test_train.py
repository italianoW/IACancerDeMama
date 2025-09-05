import train
import model


def test_treino():
    
    local_model = model.provide_model()
    
    train.train_model(local_model)
    
    # Criem um atributo _is_fitted no modelo quando treinarem:
    # model._is_fitted = True 
    #Verifica se o modelo foi treinado 
    assert hasattr(local_model, "_is_fitted") and local_model._is_fitted, "O modelo não foi treinado"
