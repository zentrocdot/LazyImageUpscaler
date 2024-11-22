#!usr/bin/bash
#
# Copy this file in the folder diffusion models amd then run it.
#
# Afterwords one needs to exchange the pointers by the real file.

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone the repository.
# git clone https://huggingface.co/stabilityai/sd-x2-latent-upscaler

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/sd-x2-latent-upscaler

exit 0
