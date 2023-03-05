from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA512, SHA256
from Crypto.Cipher import AES
import os
from typing import Tuple

def derive_enc_key(secret:str, key_len:int=32) -> bytes:
    salt = b'0'*16
    key = PBKDF2(secret.encode(), salt, key_len, count=1000000, hmac_hash_module=SHA512)
    return key


def encrypt_file(file_path:str, key:bytes, terminating_string:str=None) -> bytes:
    iv = b'0'*AES.block_size
    cipher = AES.new(key, AES.MODE_OFB, iv=iv)
    enc_bytes = cipher.encrypt(open(file_path, 'rb').read())
    if terminating_string:
        enc_bytes += cipher.encrypt(terminating_string[::-1].encode())
        enc_bytes += cipher.encrypt(os.path.basename(file_path).encode())
        enc_bytes += cipher.encrypt(terminating_string.encode())
    return enc_bytes


def decrypt_bytes(file_bytes:bytes, key:bytes, terminating_string:str=None) -> Tuple[bytes, str]:
    iv = b'0'*AES.block_size
    cipher = AES.new(key, AES.MODE_OFB, iv=iv)
    dec_bytes = cipher.decrypt(file_bytes)
    filename = None
    if terminating_string and terminating_string.encode() in dec_bytes:
        dec_bytes, filename = dec_bytes[:dec_bytes.find(terminating_string.encode())].split(terminating_string[::-1].encode())
        filename = filename.decode()
    return dec_bytes, filename


def compare_files_hashes(file1_path:str, file2_path:str) -> bool:
    file1 = SHA256.new(data=open(file1_path, 'rb').read())
    file2 = SHA256.new(data=open(file2_path, 'rb').read())
    return file1.hexdigest() == file2.hexdigest()