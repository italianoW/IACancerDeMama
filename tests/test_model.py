import model

def test_model():

  local_model_structure = model.model()

  assert local_model is not None, "O modelo não foi carregado corretamente"
  assert (isinstance(local_model,tf.keras.Model)), "provide_model não retorna um modelo Keras."
  assert isinstance(local_model.layers[-1], tf.keras.layers.Dense), "Última camada não é Dense"
  assert (local_model.layers[-1].units == 2), "A última camada do modelo não retorna um one-hot de 2 valores."
  assert (local_model.layers[-1].activation.__name__ == "softmax"), "A última camada não usa softmax"
