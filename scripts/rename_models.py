import os

results_folder = "results"

for filename in os.listdir(results_folder):
    if filename.startswith("dqn_model_part.") and filename.endswith(".gml.pth"):
        old_path = os.path.join(results_folder, filename)
        new_filename = filename.replace(".gml.pth", ".pth")
        new_path = os.path.join(results_folder, new_filename)
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {filename} -> {new_filename}")

print("🎉 All filenames updated.")
