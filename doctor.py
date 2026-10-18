"""Read-only compatibility check; does not load model weights or allocate test tensors."""
import argparse
import json
import subprocess
import sys
from rsi_tools.common import MODEL, REVISION, environment, verify_vendor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true', help='Query installedTorch/CUDA without model inference')
    a = p.parse_args()
    source = verify_vendor()
    result = {'environment': environment(), 'baseline_commit': source['commit'],
              'baseline_files_verified': len(source['files']), 'model': MODEL, 'revision': REVISION,
              'model_loaded': False, 'torch_cuda_runtime_tested': False}
    if a.gpu:
        try:
            import torch
            available = torch.cuda.is_available()
            result['cuda_available'] = available
            result['torch_cuda_build'] = torch.version.cuda
            result['compiled_architectures'] = torch.cuda.get_arch_list()
            if available:
                d = torch.cuda.get_device_properties(0)
                result['gpu'] = {'name': d.name, 'total_memory_bytes': d.total_memory,
                                 'compute_capability': [d.major, d.minor]}
                result['sm70_compiled'] = 'sm_70' in torch.cuda.get_arch_list()
            result['torch_cuda_runtime_tested'] = False
        except ImportError:
            result['cuda_available'] = None
            result['setup_error'] = 'torch not installed in this interpreter'
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
