import logging
import io, requests
import numpy as np
import tensorflow as tf
from PIL import Image
from django.http import JsonResponse
from django.http import HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

# Create a logger object
logger = logging.getLogger(__name__)

ruta_local_modelo = "model/model_inception.h5"
modelo_nube_url = 'https://github.com/SilviaDS00/RecoFood/raw/main/Modelo_Entrenado/model_inception.h5'

def descargar_modelo():
    try:
        response = requests.get(modelo_nube_url)
        with open(ruta_local_modelo, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        logger.error(f"Error al descargar el modelo: {str(e)}")


@csrf_exempt
def prediction(request):
    if "model" not in globals():
        try:
            descargar_modelo()
            model = tf.keras.models.load_model(ruta_local_modelo)
        except Exception as e:
            return JsonResponse({"error": f"Error al cargar el modelo: {str(e)}"}, status=500)
    if request.method == "POST":
        try:
            # Obtén la imagen del cuerpo de la solicitud POST
            image_data = request.FILES['imagen'].read()
            image = Image.open(io.BytesIO(image_data))

            # Preprocesa la imagen para que coincida con el formato esperado por el modelo
            image = image.resize((200, 200)) 
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # Realiza la predicción con el modelo cargado
            prediction = model.predict(image)
            # En este ejemplo, simplemente se obtiene la clase con la mayor probabilidad
            top5_classes = np.argsort(-prediction[0])[:6]
            print("Top 5 clases: ",top5_classes)
# En este ejemplo, simplemente se obtiene la clase con la mayor probabilidad
            predicted_class = top5_classes[0]

            # Devuelve la respuesta en formato JSON
            response_data = {
                "message": "Predicción exitosa",
                "predicted_class": int(predicted_class),
                "confidence": float(prediction[0][predicted_class]),
                "top5_classes": top5_classes.astype(int).tolist() if top5_classes is not None else None
            }
            return JsonResponse(response_data)
        except Exception as e:
                return JsonResponse({"error": str(e)}, status=500)

    elif request.method == "GET":
        response_data = {
            "message": "Hola. Has realizado una solicitud GET a la página de inicio."
        }
        return JsonResponse(response_data)

    else:
        return JsonResponse({"message": "Método no permitido"}, status=405)
