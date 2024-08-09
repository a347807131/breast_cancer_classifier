import csv
import json
import logging
import os
import pickle
import subprocess
import tempfile
import uuid
from typing import Union, Optional, Any, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

PATCH_MODEL_PATH = 'models/sample_patch_model.p'
IMAGE_MODEL_PATH = 'models/ImageOnly__ModeImage_weights.p'
IMAGEHEATMAPS_MODEL_PATH = 'models/ImageHeatmaps__ModeImage_weights.p'

app = FastAPI()

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """处理所有异常"""
    return JSONResponse(
        status_code=500,
        content={"message": str(exc.args)}
    )


def run_bash_script(script_path, *args):
    command = ['bash', script_path] + list(args)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode(), error.decode()


class ResponseModel(BaseModel):
    message: Optional[str] = None
    data: Optional[Any] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/predict/breast_cancer")
def predict_breast_cancer(view: str = Form(...), image_file: UploadFile = File(...),
                          with_heatmap: bool = Form(default=False)):
    logger.info(f"Received image file to do breast cancer. : {image_file.filename}")
    # 写入文件
    temp_dir = tempfile.TemporaryDirectory()
    out_dir_path = temp_dir.name
    # 确保文件夹存在
    with open(f"{temp_dir.name}/{image_file.filename}", "wb") as local_file:
        local_file.write(image_file.file.read())

    bash_result, err_out = run_bash_script(
        "run_single.sh" if not with_heatmap else "run_single_with_heatmap.sh",
        local_file.name,
        view,
        out_dir_path
    )
    temp_dir.cleanup()
    logger.debug(f"filename:{image_file.filename} result: {bash_result}")
    if err_out != "":
        raise Exception(err_out)
    out_json_str = bash_result.split("\n")[-2]
    data = json.loads(out_json_str)
    return JSONResponse(content=ResponseModel(message="Success", data=data).dict())


@app.post("/api/predict/breast_cancer_4")
def predict_breast_cancer4(
    lcc: UploadFile = File(...),
    rcc: UploadFile = File(...),
    lmlo: UploadFile = File(...),
    rmlo: UploadFile = File(...),
    with_heatmap: bool = Form(default=False),
):
    logger.info("processing breast cancer 4")
    #写入文件
    temp_dir = tempfile.TemporaryDirectory()
    with open(f"{temp_dir.name}/0_L_CC.png", "wb") as local_file:
        local_file.write(lcc.file.read())
    with open(f"{temp_dir.name}/0_R_CC.png", "wb") as local_file:
        local_file.write(rcc.file.read())
    with open(f"{temp_dir.name}/0_L_MLO.png", "wb") as local_file:
        local_file.write(lmlo.file.read())
    with open(f"{temp_dir.name}/0_R_MLO.png", "wb") as local_file:
        local_file.write(rmlo.file.read())

    out_dir_path = temp_dir.name
    out_file_path = f"{out_dir_path}/out.csv"
    bash_result, err_out = run_bash_script(
        "run_without_heatmap.sh" if not with_heatmap else "run.sh",
        out_dir_path,  #images
        out_file_path
    )
    logger.info(bash_result)
    if(not os.path.exists(out_file_path)):
        logger.error(err_out)
        JSONResponse(content=ResponseModel(message=f"failed:{err_out}").dict())
    with open(out_file_path, encoding='utf-8') as csv_file:
        reader=csv.DictReader(csv_file)
        for row in reader:
            data=row
    temp_dir.cleanup()
    return JSONResponse(content=ResponseModel(message="Success", data=data).dict())


if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000, log_config="./log-config.json")
