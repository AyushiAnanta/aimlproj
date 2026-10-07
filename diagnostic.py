# diagnostic.py — run this in the same environment you run Streamlit (or in a notebook)
import glob, os, joblib, sys, traceback

model_names = ["pipe_lr.pkl", "pipe_svm.pkl", "pipe_sgd.pkl"]
search_bases = [
    ".",                   # current dir
    os.path.expanduser("~"),
    r"C:\\", r"D:\\",      # common Windows roots
    "./models"             # local models directory
]

found = {}
print("Python executable:", sys.executable)
print("Current working dir:", os.getcwd())
print("Searching for model files...")

for base in search_bases:
    for name in model_names:
        pattern = os.path.join(base, "**", name)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            for m in matches:
                folder = os.path.dirname(m)
                found.setdefault(folder, []).append(os.path.basename(m))

if not found:
    print("No model files found in the searched places. Try running this in your notebook where you saved the models.")
else:
    print("Found model files in the following folders:")
    for folder, files in found.items():
        print(" -", folder, "->", files)

    # take the first folder and try to load each file
    first_folder = next(iter(found.keys()))
    print("\nAttempting to joblib.load() each file from:", first_folder)
    for fname in model_names:
        path = os.path.join(first_folder, fname)
        if os.path.exists(path):
            print(" Loading", path)
            try:
                obj = joblib.load(path)
                print("  OK — loaded type:", type(obj))
            except Exception as e:
                print("  FAILED to load:", e)
                traceback.print_exc()
        else:
            print("  Missing:", path)

# Final helpful check: print contents of the models dir(s)
print("\nDirectory listing of found folders:")
for folder in found.keys():
    try:
        print(folder, "->", os.listdir(folder))
    except Exception as e:
        print("  cannot list", folder, "-", e)
