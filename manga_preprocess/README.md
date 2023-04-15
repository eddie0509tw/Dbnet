# A simple preprocess script to convert Manga109 dataset into txt labels

## 0. Demo

- txt labels look like this:
```
304,99,565,99,565,282,304,282,###
268,446,376,446,376,566,268,566,###
1081,649,1183,649,1183,748,1081,748,###
1221,327,1302,327,1302,406,1221,406,###
1513,773,1623,773,1623,979,1513,979,###
1410,750,1483,750,1483,826,1410,826,###
1027,786,1223,786,1223,1041,1027,1041,###
231,539,349,539,349,641,231,641,###
166,123,186,123,186,236,166,236,ーーわかった
1453,56,1507,56,1507,150,1453,150,ごめんねーー･･･
1450,634,1469,634,1469,665,1450,665,え？
1261,628,1323,628,1323,733,1261,733,どうしたら森と付き合える？
915,794,998,794,998,1136,915,1136,マジックスターで･･･立派なアニマルマスターになれたら自分の気持ちに素直になれるんじゃないかなって･･･
1027,345,1044,345,1044,381,1027,381,森･･･
1319,764,1373,764,1373,920,1319,920,わ･･･？わかんない･･･でも私は
1529,1012,1567,1012,1567,1128,1529,1128,俺はどうしたらいい？
1588,623,1605,623,1605,747,1588,747,ーーそれじゃあ
```

## 1. Get started
1. Install requirements: see `requirements.txt`. Please note that this requirement file is generated on Ubuntu 22.04, different platforms may require different packages
2. `process_texts.py` will generate all txt label files. You may modify the path to store annotations in this file.
3. `process_images.py` will move all Manga109 images into one folder, properly named by {book name}_{page index}
4. `demo.py` is just a simple demo provided by `manga109api`, it generates an image with texts properly labelled.
5. `split_dataset.py` will split the whole dataset into train/val/test datasets according to a certain ratio, you can modify it as you wish.

## 2. Credit
Largely depend on `manga109api`, [GitHub repo here](https://github.com/shinya7y/manga109api).

