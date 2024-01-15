import replicate
from replicate.exceptions import ReplicateException

training_model_id = "nateraw/axolotl-trainer:2805a0dd68b361a3a77fcbbcca4590ef5165b4c668ce790571c3ff820e360fc4"
dest_model_id = "nateraw/test-tinyllama-english-to-hinglish"
visibility = "private"

try:
    replicate.models.create(
        owner=dest_model_id.split("/")[0],
        name=dest_model_id.split("/")[1],
        visibility=visibility,
        hardware="gpu-a40-large",  # This is the hardware used for inference
    )
except ReplicateException as e:
    print("Model already exists")

training = replicate.trainings.create(
    version=training_model_id,
    input={"config": open("config/tinyllama_debug.yaml", "rb")},
    destination=dest_model_id,
)
print(training)
