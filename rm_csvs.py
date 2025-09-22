from pathlib import Path


pasta = Path("/home/danieldcs/Save/output_forecast/predictions_tmp")

for arquivo in pasta.glob("*.csv"):
    arquivo.unlink()