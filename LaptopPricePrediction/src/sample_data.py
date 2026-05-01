from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def create_sample_dataset(path: Path, rows: int = 250) -> None:
    companies = ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"]
    type_names = ["Ultrabook", "Gaming", "Notebook", "2 in 1", "Workstation"]
    resolutions = ["1920x1080", "2560x1440", "3840x2160", "1366x768"]
    cpus = ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Intel Core i9", "AMD Ryzen 5", "AMD Ryzen 7"]
    rams = [4, 8, 16, 32]
    memories = ["256GB SSD", "512GB SSD", "1TB SSD", "256GB SSD + 1TB HDD", "512GB SSD + 1TB HDD"]
    gpus = ["Intel", "Nvidia GTX 1650", "Nvidia RTX 3050", "Nvidia RTX 3060", "Nvidia RTX 4070"]
    oses = ["Windows", "macOS", "Linux", "Chrome OS"]
    weights = [1.1, 1.3, 1.6, 2.0, 2.5]
    inches = [13.3, 14.0, 15.6, 16.0, 17.3]

    rows_data = []
    for _ in range(rows):
        company = random.choice(companies)
        typ = random.choice(type_names)
        resolution = random.choice(resolutions)
        cpu = random.choice(cpus)
        ram = random.choice(rams)
        memory = random.choice(memories)
        gpu = random.choice(gpus)
        op_sys = random.choice(oses)
        weight = round(random.choice(weights), 2)
        inch = random.choice(inches)

        price = 250
        price += 120 * (ram / 8)
        price += 45 * (inch - 13.3)
        price += 80 * (weights.index(weight) if weight in weights else 0)
        price += 100 * (type_names.index(typ) if typ in type_names else 0)
        price += 150 * (cpus.index(cpu) if cpu in cpus else 0)
        price += 100 * (gpus.index(gpu) if gpu in gpus else 0)
        price += 80 * (resolutions.index(resolution) if resolution in resolutions else 0)
        price += 50 * (memory.count("512GB") + memory.count("1TB"))
        if op_sys == "macOS":
            price += 200
        price *= random.uniform(0.9, 1.15)
        price = max(250, round(price, 2))

        rows_data.append(
            {
                "Company": company,
                "TypeName": typ,
                "Inches": inch,
                "Resolution": resolution,
                "Cpu": cpu,
                "Ram": ram,
                "Memory": memory,
                "Gpu": gpu,
                "OpSys": op_sys,
                "Weight": weight,
                "Price_euros": price,
            }
        )

    df = pd.DataFrame(rows_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    create_sample_dataset(Path("data/laptop_price.csv"))
