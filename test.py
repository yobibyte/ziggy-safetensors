from safetensors import safe_open
file_path = "/home/yobibyte/Downloads/model.safetensors"

data = {}
with safe_open(file_path, framework="pt") as f:
    for k in f.keys():
        data[k] = f.get_tensor(k)

print(data["model.layers.9.self_attn.v_proj.weight"])
