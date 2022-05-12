import os, sys, json, uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from im_encode import encode
from style_gan_clip import main
import cv2
import numpy as np

app = FastAPI(
    title="Styler",
    previx="/api/v1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.neovotech.com",
        "http://localhost:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", tags=["Welcome"])
async def index():
    """
        Greetings for base link routing
        params: None
        return: str
    """
    return "Welcome to testing of Styler"

@app.post("/generate/", tags=['GENERATE CLOTHES'])
async def get_generated_clothes(request: dict):
    """"
        Post request for getting news with ISIN.
        params: dict type
        return: byte images
    """
    try:
        print(request)
        keys = [*request]

        concatenated_input = request[keys[0]] + ' ' + request[keys[3]] + ' ' + request[keys[2]] + ' ' + request[keys[1]] + ' ' + request[keys[4]]
        img_list = main(concatenated_input=concatenated_input, number_of_clothes=concatenated_input[5])

        response_img = np.empty((224, 448), dtype=np.int8)
        for i, img in enumerate(img_list):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if i == 0:
                response_img = img
            else:
                response_img = np.append(response_img, img, 0)
            # cv2.imwrite(f"./images/{'-'.join(concatenated_input.split())}-{i}.jpg", img)
        byte_img = encode(response_img)
        return Response(content=byte_img, media_type="image/jpg")
    except Exception:
        pass

if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8081)