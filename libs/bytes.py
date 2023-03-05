from typing import List


word_len = 8 # bits

def convert_bytes_to_bits(file_bytes:bytes) -> str:
    bits = ''.join([format(b, f'0{word_len}b') for b in file_bytes])
    return bits


def convert_bits_to_bytes(bits:str) -> List[bytes]:
    file_body = b''.join([int(bits[i:i+word_len], 2).to_bytes(1, 'big') for i in range(0, len(bits), word_len)])
    return file_body


def save_bytes_as_file(dest_path:str, data:List[bytes]) -> bool:
    open(dest_path, 'wb').write(data)
    print(f'File saved at {dest_path}')
    return True