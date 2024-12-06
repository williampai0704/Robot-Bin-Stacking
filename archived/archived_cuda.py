import sys
import torch

def check_gpu_setup():
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    
    # Basic CUDA Check
    print("\n--- CUDA Availability ---")
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    
    # Detailed GPU Information
    try:
        if torch.cuda.is_available():
            print("\n--- GPU Details ---")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Current Device Info
            print("\n--- Current Device ---")
            print("Current Device Index:", torch.cuda.current_device())
            
            # Memory Information
            print("\n--- GPU Memory ---")
            print("Total Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
            print("Allocated Memory:", torch.cuda.memory_allocated() / 1e9, "GB")
            print("Cached Memory:", torch.cuda.memory_reserved() / 1e9, "GB")
        else:
            print("No CUDA-capable devices found.")
    
    except Exception as e:
        print(f"Error gathering GPU information: {e}")

    # Additional Diagnostics
    print("\n--- Additional Checks ---")
    print("cuDNN Available:", torch.backends.cudnn.is_available())
    print("cuDNN Version:", torch.backends.cudnn.version())

if __name__ == "__main__":
    check_gpu_setup()