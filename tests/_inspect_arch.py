import json
import zipfile
import io
import pickle


class FakeTensor:
    def __init__(self, storage, shape, stride):
        self.shape = tuple(shape)
        self.storage = storage


def rebuild(storage, offset, size, stride, *a):
    return FakeTensor(storage, size, stride)


class U(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict

            return OrderedDict
        if name == "_rebuild_tensor_v2":
            return rebuild
        return lambda *a, **k: None

    def persistent_load(self, pid):
        return pid[2]


path = r"C:\Users\Shaik Asif\OneDrive\Desktop\Plant Pathology\models\plant_model.pth"
z = zipfile.ZipFile(path)
sd = U(io.BytesIO(z.read("plant_model/data.pkl"))).load()

# Group keys per top-level block
for block in ["features.0", "features.1", "features.2", "features.17", "features.18", "classifier"]:
    print("=== " + block + " ===")
    for k in sd:
        if k == block or k.startswith(block + "."):
            print("   ", k, tuple(sd[k].shape))