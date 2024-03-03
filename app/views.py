import logging, io
import numpy as np
import tensorflow as tf
from PIL import Image
from django.http import JsonResponse
from django.http import HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

# Create a logger object
logger = logging.getLogger(__name__)

model = tf.keras.models.load_model('model/model_inception.h5')

@csrf_exempt
def prediction(request):
    # print(request)
    if request.method == 'POST':
        try:
            image_data = request.FILES['imagen'].read()
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((200, 200)) 
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            top5_classes = np.argsort(-prediction[0])[:6]
            print("Top 5 clases: ",top5_classes)
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
        logger.warning("Método no soportado %s", request.method)
        return HttpResponseBadRequest('Método no soportado')
