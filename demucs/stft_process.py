import numpy as np
import torch

# To export your own STFT process ONNX model, set the following values.

DYNAMIC_AXES = True                                 # Default dynamic axes is input audio (signal) length.
NFFT = 1024                                         # Number of FFT components for the STFT process
HOP_LENGTH = 256                                    # Number of samples between successive frames in the STFT
INPUT_AUDIO_LENGTH = 16000                          # Set for static axes. Length of the audio input signal in samples.
MAX_SIGNAL_LENGTH = 2048                            # Maximum number of frames for the audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hann'                                # Type of window function used in the STFT
PAD_MODE = 'reflect'                                # Select reflect or constant
NORMALIZED = True                                   # Whether to normalize the window function
STFT_TYPE = "stft_B"                                # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_B"                              # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [magnitude, real_part, imag_part], The dtype of imag_part is float format.
export_path_stft = f"{STFT_TYPE}.onnx"              # The exported stft onnx model save path.
export_path_istft = f"{ISTFT_TYPE}.onnx"            # The exported istft onnx model save path.


# Precompute constants to avoid calculations at runtime
HALF_NFFT = NFFT // 2
STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1

# Sanity checks for parameters
NFFT = min(NFFT, INPUT_AUDIO_LENGTH)
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)

# Create window function lookup once
WINDOW_FUNCTIONS = {
    'bartlett': torch.bartlett_window,
    'blackman': torch.blackman_window,
    'hamming': torch.hamming_window,
    'hann': torch.hann_window,
    'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
}
# Define default window function
DEFAULT_WINDOW_FN = torch.hann_window
# Initialize window - only compute once
WINDOW = WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(NFFT).float()


class STFT_Process(torch.nn.Module):
    def __init__(self, model_type=STFT_TYPE, n_fft=NFFT, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, normalized=NORMALIZED):
        super(STFT_Process, self).__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.max_frames = max_frames
        self.window_type = window_type
        self.normalized = normalized
        self.half_n_fft = n_fft // 2  # Precompute once
        
        # Get window function and compute window once
        window = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(n_fft).float()
        
        # Register common buffers for all model types
        self.register_buffer('padding_zero', torch.zeros((1, 1, self.half_n_fft), dtype=torch.float32))
        
        # Pre-compute model-specific buffers
        if self.model_type in ['stft_A', 'stft_B']:
            # STFT forward pass preparation
            time_steps = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
            frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
            
            # Calculate omega matrix once
            omega = 2 * torch.pi * frequencies * time_steps / n_fft

            # Calculate window function
            cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
            sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

            # Normalize window if needed
            if normalized:
                cos_kernel = cos_kernel / torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))
                sin_kernel = sin_kernel / torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))

            # Register conv kernels as buffers
            self.register_buffer('cos_kernel', (cos_kernel))
            self.register_buffer('sin_kernel', (sin_kernel))

    def forward(self, *args):
        # Use direct method calls instead of if-else cascade for better ONNX export
        if self.model_type == 'stft_A':
            return self.stft_A_forward(*args)
        if self.model_type == 'stft_B':
            return self.stft_B_forward(*args)
        if self.model_type == 'istft_A':
            return self.istft_A_forward(*args)
        if self.model_type == 'istft_B':
            return self.istft_B_forward(*args)
        # In case none match, raise an error
        raise ValueError(f"Unknown model type: {self.model_type}")

    def stft_A_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x_padded = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        
        # Single conv operation
        return torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len, groups=2)

    def stft_B_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x_padded = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        
        # Perform convolutions
        real_part = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        image_part = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)

        # return real_part, image_part
        return torch.stack((real_part, image_part), dim=-1)
        # return torch.view_as_complex(torch.stack((real_part, image_part), dim=-1))


def test_stft_B(input_signal):
    torch_stft_output = torch.view_as_real(torch.stft(
        input_signal.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        win_length=NFFT,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode=PAD_MODE,
    ))
    pytorch_stft_real = torch_stft_output[..., 0].squeeze().numpy()
    pytorch_stft_imag = torch_stft_output[..., 1].squeeze().numpy()

    stft_model = STFT_Process(
        model_type=STFT_TYPE,
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        max_frames=MAX_SIGNAL_LENGTH,
        window_type=WINDOW_TYPE,
        normalized=True
    ).eval()

    with torch.no_grad():
        stft_output = stft_model(input_signal)
        # stft_output = torch.view_as_real(stft_output)
        custom_stft_real = stft_output[..., 0].squeeze().numpy()
        custom_stft_imag = stft_output[..., 1].squeeze().numpy()

    mean_diff_real = np.abs(pytorch_stft_real - custom_stft_real).mean()
    mean_diff_imag = np.abs(pytorch_stft_imag - custom_stft_imag).mean()
    mean_diff = (mean_diff_real + mean_diff_imag) * 0.5
    print("\nSTFT Result: Mean Difference =", mean_diff)

def main():
    with torch.inference_mode():

        # Create dummy input signal
        dummy_stft_input = torch.randn((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.float32) # Shape: (batch, channels, samples)

        print("\nTesting the Custom.STFT versus Pytorch.STFT ...")
        test_stft_B(dummy_stft_input)

if __name__ == "__main__":
    main()