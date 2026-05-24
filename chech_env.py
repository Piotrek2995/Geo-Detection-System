import torch
from ultralytics import YOLO

print("torch:", torch.__version__, "| CUDA dostępne:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

m = YOLO("LudzieiNamioty.pt")
print("Klasy modelu:", m.names)  # sprawdź ID klas namiot/człowiek!