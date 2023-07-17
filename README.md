# Imporoved RoutingConvNet
<hr>


## 01. Data preprocessing
<hr>

- feature : MFCC
  - `signal_len < 100,000` ... zero-padding
  - `else` ... cut


## 02. Model architecture
<hr>

- Model

<p align="center"><img src="https://private-user-images.githubusercontent.com/33558083/248092761-a7ea3729-b9ba-4866-9739-9409dec9a3af.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg4MTcwMTIyLCJuYmYiOjE2ODgxNjk4MjIsInBhdGgiOiIvMzM1NTgwODMvMjQ4MDkyNzYxLWE3ZWEzNzI5LWI5YmEtNDg2Ni05NzM5LTk0MDlkZWM5YTNhZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwNzAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDcwMVQwMDAzNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wMWE5MjY4NmUwZGJiNjBjZjVlMDg0MTM1NWJmMDFjY2YwYTU1NGM3N2E2ZjhkZjc1ZDkwMjZkYTJiYjJlYzAwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.J95Ol5tMkDqyL2T-WEeD1N4oPnxWNeDyytHGtWhkg6c" height="100px" width="500px"></p>

- legend

<p align="center"><img src="https://private-user-images.githubusercontent.com/33558083/248094293-cc422075-ab56-48cf-86a8-8c717dc1d250.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg4MTcwMTIyLCJuYmYiOjE2ODgxNjk4MjIsInBhdGgiOiIvMzM1NTgwODMvMjQ4MDk0MjkzLWNjNDIyMDc1LWFiNTYtNDhjZi04NmE4LThjNzE3ZGMxZDI1MC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwNzAxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDcwMVQwMDAzNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kZjEwYWM3YmRlODg0MmE4OWZlMDYyMDBmOTU2ODVjMzVlNjAyODkyZDNhZmYwOGIyOGVjM2Y1MDBhMjFjMTg5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.b-nvHz3C3LQvG17My_B8IVqLrBHGcKmFLfBTkmqN3nA" height="100px" width="200px"></p>


## 03. Experiments - Performance
<hr>

- meaning
  - proposed : proposed model
  - capsule : remove dynamic routing
  - SA : remove self-attention
  - reshape : remove reshape
  - CBAM : remove CBAM(Convolution Block Attention Module)
  - Spa : remove spatial-attention
  - CBAM, Spa : remove CBAM, Spa


- <span style="color:blue">**EMO-DB**</span>

|Name|#Params|max_WA(%)|min_WA(%)|avg_WA(%)|code|loss curve|Visualization|
|----|----------|---------|---------|---------|-------|------------------|-------------|
|proposed|93,872|96.05|83.09|89.09|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/training/proposed.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/7#issue-1804415490)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/visualization/proposed.ipynb)|
|capsule|69,296|96.87|78.72|83.83|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/training/remove%20capsule.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/7#issuecomment-1635467095)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/visualization/remove_dynamic-routing.ipynb)|
|SA|93,743|93.67|82.66|88.00|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/training/remove%20sa.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/7#issuecomment-1635468522)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/visualization/remove_sa.ipynb)|
|reshape|68,272|84.65|72.39|76.87|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/training/remove%20reshape.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/7#issuecomment-1635469062)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/visualization/remove_reshape.ipynb)|
|CBAM, Spa|89,476|96.05|79.26|87.54|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/training/remove%20CBAM.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/7#issuecomment-1635469763)|Link|
|Spa|93,770|96.05|82.66|88.18|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/EMO-DB/training/remove%20spatial.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/7#issuecomment-1635470553)|Link|

- <span style="color:blue">**RAVDESS**</span>

|Name|#Params|max_WA(%)|min_WA(%)|avg_WA(%)|code|loss curve|Visualization|
|----|----------|---------|---------|---------|-------|------------------|-------------|
|proposed|93,937|90.00|80.00|85.69|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/training/proposed.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/9#issue-1804540413)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/visualization/proposed.ipynb)|
|capsule|69,361|75.62|64.38|69.56|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/training/remove%20capsule.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/9#issuecomment-1635578450)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/visualization/remove_capsule.ipynb)|
|SA|93,808|87.50|79.38|83.81|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/training/remove%20sa.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/9#issuecomment-1635579801)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/visualization/remove_sa.ipynb)|
|reshape|68,337|78.75|65.62|72.19|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/training/remove%20reshape.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/9#issuecomment-1635581454)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/visualization/remove_reshape.ipynb)|
|CBAM, Spa|89,541|83.75|73.12|81.38|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/training/remove%20cbam%2C%20spatial-attention.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/9#issuecomment-1635582157)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/visualization/remove_CBAM.ipynb)|
|Spa|93,835|85.62|74.38|81.38|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/training/remove%20spatial.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/9#issuecomment-1635583236)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/RAVDESS/visualization/remove_spatial.ipynb)|

- <span style="color:blue">**IEMOCAP**</span>

|Name|#Params|max_WA(%)|min_WA(%)|avg_WA(%)|code|loss curve|Visualization|
|----|----------|---------|---------|---------|-------|------------------|-------------|
|proposed|93,677|70.56|63.59|66.22|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/IEMOCAP/training/proposed.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issue-1804422136)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issue-1804422136)|
|capsule|69,101|67.88|62.63|65.04|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/IEMOCAP/training/remove%20capsule.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issuecomment-1635472280)|Link|
|SA|93,548|69.09|62.90|65.95|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/IEMOCAP/training/remove%20sa.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issuecomment-1638289277)|Link|
|reshape|68,077|63.09|58.98|61.23|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/IEMOCAP/training/remove%20reshape.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issuecomment-1635472871)|Link|
|CBAM, Spa|89,281|69.17|62.92|66.46|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/IEMOCAP/training/remove%20CBAM.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issuecomment-1638290439)|Link|
|Spa|93,575|68.82|63.43|65.78|[Link](https://github.com/devLupin/Improved-RoutingConvNet/blob/main/IEMOCAP/training/remove%20spatial.ipynb)|[Link](https://github.com/devLupin/Improved-RoutingConvNet/issues/8#issuecomment-1638291450)|Link|


## 04. Experiments - Real-time
<hr>

- setting
  - `batch_size = 1`
  - `Eq = all test dataset inference time / number of test dataset`
    - i.e. Average

- `Inference time / wav` (sec)

|H/W|EMO-DB|RAVDESS|IEMOCAP|
|---|------|-------|-------|
|RTX 3080TI|0.04371|0.03033|0.03416|
|i7-12700K|0.05000|0.04545|0.04510|
|RTX 2080TI|0.07182|0.06225|0.04953|
|i7-8700|0.07622|0.07257|0.06538|
|Raspberry Pi|1.42443|1.35941|1.22835|


## 05. Experiments - Memory usage
<hr>

- GPU peak memory usage
  - Maximum usage of GPU memory at the moment
  - via `tf.config.experimental.get_memory_info(‘GPU:0’)`
- Model size
  - saved model weights size

|Model|Num.Params|Peak memory usage(GB)|Model size(Mb)|
|-----|----------|---------------------|--------------|
|Proposed|93K|0.000627|0.433616|
