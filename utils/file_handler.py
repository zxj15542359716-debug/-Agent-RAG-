#文件处理
import logging
import os,hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader

def get_file_md5_H(filepath:str):    #获取文件md5的十六进制字符串
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}not exist")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]文件{filepath}not file")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096    #分片避免文件过大内存爆炸
    try:
        with open(filepath, "rb") as f:  #必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)    #不断计算更新md5

            md5_H = md5_obj.hexdigest()
            return md5_H
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败,{str(e)}")
        return None

def listdir_with_allowed_type(path: str,allowed_types:tuple[str]):  #返回文件夹的文件列表
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))    #传回文件夹和文件名组合的名字
    return tuple(files)

def pdf_loader(filepath:str,password=None)->list[Document]:
    return PyPDFLoader(filepath,password).load()

def txt_loader(filepath:str):
    return TextLoader(filepath,encoding="utf-8").load()