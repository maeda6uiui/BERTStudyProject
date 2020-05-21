# BERTStudyProject

BERTの勉強用プロジェクト

# Todo

[aio_test.py](./aio_test.py)を完成させる。

# 実行結果メモ

## question_answering_en.py

### テキスト

```
Saint Petersburg, formerly known as Petrograd (1914–1924), then Leningrad (1924–1991), is a city situated on the Neva River, at the head of the Gulf of Finland on the Baltic Sea. It is Russia's second-largest city after Moscow. With over 5.3 million inhabitants as of 2018, it is the fourth-most populous city in Europe, as well as being the northernmost city with over one million people. An important Russian port on the Baltic Sea, it has a status of a federal subject (a federal city).
```

[Saint Petersburg](https://en.wikipedia.org/w/index.php?title=Saint_Petersburg&oldid=955987618)

### 問題と解答の例

```
What is the former name of Saint Petersburg during the period from 1924 to 1991?
leningrad 
```

```
What is the largest city in Russia?
moscow 
```

## predict_simple_en.py

```
* is the capital city of Japan.
['tokyo', 'osaka', 'nagoya', 'kyoto', 'nagasaki']
```

## predict_simple.py

```
日本の首都は*です。
['東京', '大阪', '京都', '名古屋', '神戸']
```



