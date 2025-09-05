import model
import tensorflow as tf
from tensorflow import keras

def test_model_structure():
    
    local_model = model.provide_model()
  
    #Verifica se o retorno da função obter_modelo retorna mesmo um modelo Keras.   
    assert (isinstance(local_model,keras.Model)), "provide_model não retorna um modelo Keras."
    
    #Verifica se a última camada é Dense.
    assert isinstance(local_model.layers[-1], keras.layers.Dense), "Última camada não é Dense"
    
    #Verifica se a última camada do modelo retorna um one-hot de tamanho 2.
    assert (local_model.layers[-1].units == 2), "A última camada do modelo não retorna um one-hot de 2 valores."
    
    #Verifica se a última camada do modelo aplica Softmax para categorização correta.
    assert (local_model.layers[-1].activation.__name__ == "softmax"), "A última camada não usa softmax"

def test_save_model():
    
    #TODO
    assert 1
