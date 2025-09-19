import util
import tensorflow as tf

def test_util():

  model = util.load_trained_model()

  assert model is not None, "O modelo não foi carregado corretamente"
  assert (isinstance(model,tf.keras.Model)), "provide_model não retorna um modelo Keras."
  assert isinstance(model.layers[-1], tf.keras.layers.Dense), "Última camada não é Dense"
  assert (model.layers[-1].units == 2), "A última camada do modelo não retorna um one-hot de 2 valores."
  assert (model.layers[-1].activation.__name__ == "softmax"), "A última camada não usa softmax"
  assert hasattr(model, "_is_fitted") and model._is_fitted, "O modelo não foi treinado"


def test_image_generators():

  train_gen = util.get_train_image_generator()
  test_gen = util.get_test_image_generator()

  assert isinstance(train_gen, tf.keras.preprocessing.image.ImageDataGenerator), "get_train_image_generator não retorna um gerador de imagem."
  assert isinstance(test_gen, tf.keras.preprocessing.image.ImageDataGenerator), "get_test_image_generator não retorna um gerador de imagem."

  imgs_train, labels_train = next(train_gen)
  imgs_test, labels_test = next(test_gen)

  assert isinstance(imgs_train, tf.Tensor), "imgs_train não é um tensor"
  assert isinstance(labels_train, tf.Tensor), "labels_train não é um tensor"
