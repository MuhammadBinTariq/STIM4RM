# Save this as: src/apply_patches.py
import os
import sys
import site

def patch_olmo_library():
    print(">>> 🛡️  APPLYING ENVIRONMENT PATCHES...")
    
    # 1. FIND THE LIBRARY PATH
    try:
        site_packages = site.getsitepackages()[0]
        base_path = os.path.join(site_packages, "olmo")
    except:
        # Fallback for Colab/Debian specific paths
        base_path = "/usr/local/lib/python3.9/dist-packages/olmo"

    model_path = os.path.join(base_path, "model.py")
    util_path = os.path.join(base_path, "util.py")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Could not find OLMo library at {base_path}")
        return

    # 2. PATCH model.py (The 'NoneType' Crash Fix)
    with open(model_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    model_patched = False
    for line in lines:
        if "past_length =" in line and "size(-2)" in line and "try:" not in line:
            indent = line[:line.find("past_length")]
            safe_block = (
                f"{indent}try:\n"
                f"{indent}    past_length = past_key_values[0][0].size(-2)\n"
                f"{indent}except (AttributeError, TypeError, IndexError):\n"
                f"{indent}    past_length = 0\n"
            )
            new_lines.append(safe_block)
            model_patched = True
        else:
            new_lines.append(line)
    
    if model_patched:
        with open(model_path, "w") as f:
            f.writelines(new_lines)
        print("✅ Patched olmo/model.py (KV Cache Crash Fix)")

    # 3. PATCH util.py (The Python 3.9 Type Hint Fix)
    if os.path.exists(util_path):
        with open(util_path, "r") as f:
            content = f.read()
        
        if "datasets.DatasetDict | datasets.Dataset" in content:
            new_content = content.replace("datasets.DatasetDict | datasets.Dataset", "object")
            with open(util_path, "w") as f:
                f.write(new_content)
            print("✅ Patched olmo/util.py (Python 3.9 Fix)")

if __name__ == "__main__":
    patch_olmo_library()