import os
import zipfile

project_dir = "/home/karthik/code/frouros"          # Current project
output_zip = "/home/karthik/code/project.zip" # Output zip file

exclude_dirs = {
    "__pycache__",
    "data",
    "model",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    "faces"
}

exclude_extensions = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".trt",
    ".ckpt",
    ".h5",
    ".pb",
    ".tflite",
    ".weights",
    ".zip"
}

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file == output_zip:
                continue

            if any(file.endswith(ext) for ext in exclude_extensions):
                continue

            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, project_dir)
            zipf.write(file_path, arcname)

print(f"✅ Project zipped successfully as '{output_zip}'")