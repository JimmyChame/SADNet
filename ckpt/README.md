### Test
1. Download the trained models from [Google Drive](https://drive.google.com/file/d/10HdJeTwvcJ804lQOZPk4fMLJEQaJx8Yc/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1xBEFW4EGcpKF8eArxNMn0A)(code:l9qr) and place them in `/ckpt/`.
2. Place the testing dataset in `/dataset/test/` or set the testing path in `option.py` to your own path.
3. Set the parameters in `option.py` (eg. 'epoch_test', 'gray' and etc.)
3. test the trained models:
```
cd $SADNet_ROOT
python test.py
```
