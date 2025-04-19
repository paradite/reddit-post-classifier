import torch
import sys

def check_cuda():
    """
    Check if PyTorch is running on GPU (CUDA) and display detailed information
    about the PyTorch installation and GPU.
    """
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("Current CUDA Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(0))
        print("Device Count:", torch.cuda.device_count())
        print("Device Properties:", torch.cuda.get_device_properties(0))
        
        # Test CUDA with a simple tensor operation
        print("\nTesting CUDA with a simple tensor operation:")
        x = torch.rand(5, 3)
        print("CPU Tensor:", x)
        
        x = x.cuda()
        print("GPU Tensor:", x)
        
        # Check if tensor is on GPU
        print("Tensor is on GPU:", x.is_cuda)
        
        # Memory information
        print("\nGPU Memory Information:")
        print("Allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
        print("Cached:", torch.cuda.memory_reserved(0) / 1024**2, "MB")
    else:
        print("\nCUDA is not available. PyTorch will use CPU only.")
        print("To use GPU, make sure you have:")
        print("1. A CUDA-capable GPU")
        print("2. CUDA toolkit installed")
        print("3. PyTorch installed with CUDA support")
        
        # Check if CUDA is installed but not available
        if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
            print("\nCUDA toolkit is installed but not available to PyTorch.")
            print("This might be due to a mismatch between PyTorch and CUDA versions.")
            print("Try reinstalling PyTorch with the correct CUDA version.")

if __name__ == "__main__":
    check_cuda() 