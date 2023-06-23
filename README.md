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

<p align="center"><img src="https://private-user-images.githubusercontent.com/33558083/248092761-a7ea3729-b9ba-4866-9739-9409dec9a3af.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg3NDc3ODUyLCJuYmYiOjE2ODc0Nzc1NTIsInBhdGgiOiIvMzM1NTgwODMvMjQ4MDkyNzYxLWE3ZWEzNzI5LWI5YmEtNDg2Ni05NzM5LTk0MDlkZWM5YTNhZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwNjIyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDYyMlQyMzQ1NTJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hMTE4MTk5ODEyNDdhY2JjYjZjZmVmYzNhNzgzNDQ0ZDEyMzg5MDBhMzQ3MmE4MmM4OGRhM2FiZmVkZTVlNTRiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.i1KAJtP-Kea0cRUTNCEKStFXLe3dt6F8X4iNqeIgwmQ" height="100px" width="500px"></p>

- legend

<p align="center"><img src="https://private-user-images.githubusercontent.com/33558083/248094293-cc422075-ab56-48cf-86a8-8c717dc1d250.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXkiOiJrZXkxIiwiZXhwIjoxNjg3NDc3OTAzLCJuYmYiOjE2ODc0Nzc2MDMsInBhdGgiOiIvMzM1NTgwODMvMjQ4MDk0MjkzLWNjNDIyMDc1LWFiNTYtNDhjZi04NmE4LThjNzE3ZGMxZDI1MC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMwNjIyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMDYyMlQyMzQ2NDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04NWMyMjdmNmFiYTZhNTFjYjdlM2ZiM2I5MDIxODhiMjc5MTk4ZjA3MmMxNWUwZDljOWViNGJlODIwZmNhODM3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Lex6earCz5lGVKk-9iFUe89uYIMXMDcrD34SZupMHNM" height="100px" width="200px"></p>


## 03. Experiments - Performance
<hr>

- <span style="color:blue">**EMO-DB**</span>

|Name|Num.Params|max_WA(%)|min_WA(%)|avg_WA(%)|Jupyter|10-Fold loss curve|Visualization|
|----|----------|---------|---------|---------|-------|------------------|-------------|
|Proposed|93,872|96.05|83.09|89.09|-------|------------------|-------------|
|remove Capsule|69,296|96.87|78.72|83.83|-------|------------------|-------------|
|remove Self-attention|93,743|93.67|82.66|88.00|-------|------------------|-------------|
|remove Reshape|68,272|84.65|72.39|76.87|-------|------------------|-------------|
|remove CBAM, Spaital-attention|----------|------|------|------|-------|------------------|-------------|
|remove CBAM|89,476|96.05|79.26|87.54|-------|------------------|-------------|
|remove Spatial-attention|93,770|96.05|79.26|87.54|-------|------------------|-------------|

- <span style="color:blue">**RAVDESS**</span>

|Name|Num.Params|max_WA(%)|min_WA(%)|avg_WA(%)|Jupyter|10-Fold loss curve|Visualization|
|----|----------|---------|---------|---------|-------|------------------|-------------|
|Proposed|93,872|96.05|83.09|89.09|-------|------------------|-------------|
|remove Capsule|69,361|75.62|64.38|69.56|-------|------------------|-------------|
|remove Self-attention|93,808|87.50|79.38|83.81|-------|------------------|-------------|
|remove Reshape|68,337|78.75|65.62|72.19|-------|------------------|-------------|
|remove CBAM, Spaital-attention|89,541|83.75|73.12|81.38|-------|------------------|-------------|
|remove CBAM|----------|---------|---------|---------|-------|------------------|-------------|
|remove Spatial-attention|93,835|85.62|74.38|81.38|-------|------------------|-------------|

- <span style="color:blue">**IEMOCAP**</span>

|Name|Num.Params|max_WA(%)|min_WA(%)|avg_WA(%)|Jupyter|10-Fold loss curve|Visualization|
|----|----------|---------|---------|---------|-------|------------------|-------------|
|Proposed|93,677|70.56|63.59|66.22|-------|------------------|-------------|


## 04. Experiments - Real-time
<hr>

- setting
  - `batch_size = 1`
  - `Eq = all test dataset inference time / number of test dataset`
    - i.e. Average

- `Inference time / wav` (sec)

|H/W|EMO-DB|RAVDESS|IEMOCAP|
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