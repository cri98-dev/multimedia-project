# Stego Tool

This is the project realized for the exam of the subject "Multimedia e Laboratorio" (University of Catania, Computer Science Department).

# How to start this project?

1. Install Python 3 (project developed using Python 3.8.10).
2. Run `pip install -r requirements.txt` in main project directory.
3. Run `python3 iface.py` in main project directory.
4. Enjoy!

# Core functionality working scheme

## Spatial Domain

### Embedding phase:

<p align='center'>
  <img src='Embedding.svg'>
</p>

### Retrieval phase:

<p align='center'>
  <img src='Retrieval.svg'>
</p>

## Frequency Domain

The following image is from the paper at https://www.researchgate.net/publication/269705199_Digital_Image_Steganography_An_FFT_Approach.

### Embedding phase:

<p align='center'>
  <img src='paper_embedding_scheme.png'>
</p>

# Benchmarking

### LSB-embeddings only

<p align='center'>
  <img src='benchmark_table.png'>
</p>

### All embeddings

<b>N.B.:</b> The suffix "_x" in the embedding algorithms names indicates that the data has been hidden in the chrominance-x channel spectral magnitude.

<p align='center'>
  <img src='benchmarking_fourier.png'>
</p>
