"""
# ==============
# References
# ==============

[1] https://medium.com/@lankinen/fastai-model-to-production-this-is-how-you-make-web-app-that-use-your-model-57d8999450cf
"""

import os
from io import BytesIO
from pathlib import Path

import aiohttp
import uvicorn
from fastai.vision import open_image, load_learner
from starlette.applications import Starlette
from starlette.responses import JSONResponse

CSV_DATA_PATH = Path(os.path.join('data', 'casino'))
IMAGE_DATA_PATH = CSV_DATA_PATH / 'images'

app = Starlette(debug=True)
learner = load_learner(IMAGE_DATA_PATH, file='resnet34_prod.pkl')


@app.route('/')
async def homepage(request):
    return JSONResponse({'hello': 'world'})


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@app.route("/get_preds", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _, _, losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(zip(learner.data.classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    })


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
