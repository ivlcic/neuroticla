če # neuroticla

Library for neural networks based text classification and processing.

## Initialize the project

### Single time initialization
```
cd ~/projects
git lfs clone https://github.com/ivlcic/neuroticla.git
cd neuroticla
python -m venv venv
source venv/bin/activate 
pip install --upgrade pip
pip install -r requirements.txt
```

### Before each new session
```
cd ~/projects
cd neuroticla
source venv/bin/activate
```

## Named Entity Recognition

### Data preparation
(single time, optional step)

Convert all Slovenian corpora files to common format
```
./ner prep sl
```
All supported languages (*sl, hr, sr, bs, mk, sq, cs, bg, pl, ru, sk, uk*) corpora has been already prepared.
See command help for additional options:  
```
./ner prep --help
```

### Data split
(single time, run before each desired data-split change)

The following command:
```
./ner split -s 80:10 sl hr sr bs mk sq cs bg pl ru sk uk
```
will produce 80%% train, 10%% evaluation and 10%% test data set size:
```
data/ner/split/
├── bg.eval.csv
├── bg.test.csv
├── bg.train.csv
├── bs.eval.csv
├── bs.test.csv
├── bs.train.csv
├── cs.eval.csv
├── cs.test.csv
├── cs.train.csv
├── hr.eval.csv
├── hr.test.csv
├── hr.train.csv
├── mk.eval.csv
├── mk.test.csv
├── mk.train.csv
├── pl.eval.csv
├── pl.test.csv
├── pl.train.csv
├── ru.eval.csv
├── ru.test.csv
├── ru.train.csv
├── sk.eval.csv
├── sk.test.csv
├── sk.train.csv
├── sl.eval.csv
├── sl.test.csv
├── sl.train.csv
├── sq.eval.csv
├── sq.test.csv
├── sq.train.csv
├── sr.eval.csv
├── sr.test.csv
├── sr.train.csv
├── uk.eval.csv
├── uk.test.csv
└── uk.train.csv
```
(this is also the default if `-s` switch is omitted)

Each language corpora is combined, shuffled and split. 
By default the data shuffle is reproducible, so be careful here!

For all options see: 
```
./ner split --help
```

### Train
The *train / eval / test* splits 
are concatenated from all languages in order as specified in command line.

```
./ner train -l 2e-5 -e 40 -b 20 xlmrb sl hr sr bs mk sq cs bg pl ru sk uk
```
For all options see: 
```
./ner train --help
```

### Single language complete example
Here is the fastest and smallest possible usage example (approx 1h on 1080 Ti):
```
# optionally prep sr.zip that contains target .csv, .conll, and analysis .json, 
./ner prep sr

# split it to 80% train, 10% eval, 10% test set.
./ner split sr

# train / fine-tune the model showing progess, with 2e-5 learning rate, for 40 epochs and with teh batch size of 20
# using HuggingFace's XLMRoberta-base pretrained model and Serbian language corpora. 
./ner train --tqdm -l 2e-5 -e 40 -b 20 xlmrb sr

# output the model evaluation against the test data
./ner test --tqdm -b 20 xlmrb-sr sr

# use the model for inference
./ner infer xlmrb-sr sr "Pa dali je to Majkel Đekson. Majke mi da jeste! I to baš tu usred Beograda!"
...
2023-08-18 16:21:55 INFO    ner.infer 67 : Pa dali je to [Majkel Đekson]-{PER}.
2023-08-18 16:21:56 INFO    ner.infer 67 : [Majke]-{PER} mi da jeste!
2023-08-18 16:21:56 INFO    ner.infer 67 : I to baš tu usred [Beograda]-{LOC}!

# train / fine-tune the model showing progess, with 2e-5 learning rate, for 40 epochs and with teh batch size of 20
# using HuggingFace's XLMRoberta-large pretrained model and Serbian language corpora. 
./ner train --tqdm -l 2e-5 -e 40 -b 20 xlmrl sr

# use the model for inference
./ner infer xlmrb-sr sr "Pa dali je to Majkel Đekson. Majke mi da jeste! I to baš tu usred Beograda!"
...
2023-08-19 06:25:14 INFO    ner.infer 67 : Pa dali je to [Majkel Đekson]-{PER}.
2023-08-19 06:25:14 INFO    ner.infer 67 : [Majke]-{PER} mi da jeste!
2023-08-19 06:25:14 INFO    ner.infer 67 : I to baš tu usred [Beograda]-{LOC}!
```

### Used NER Corpora

We keep used NER corpora in this repository just for convenience.

- [Training corpus SUK 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1747)
  
```
@misc{11356/1747,
 title = {Training corpus {SUK} 1.0},
 author = {Arhar Holdt, {\v S}pela and Krek, Simon and Dobrovoljc, Kaja and Erjavec, Toma{\v z} and Gantar, Polona and {\v C}ibej, Jaka and Pori, Eva and Ter{\v c}on, Luka and Munda, Tina and {\v Z}itnik, Slavko and Robida, Nejc and Blagus, Neli and Mo{\v z}e, Sara and Ledinek, Nina and Holz, Nanika and Zupan, Katja and Kuzman, Taja and Kav{\v c}i{\v c}, Teja and {\v S}krjanec, Iza and Marko, Dafne and Jezer{\v s}ek, Lucija and Zajc, Anja},
 url = {http://hdl.handle.net/11356/1747},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {Creative Commons - Attribution-{NonCommercial}-{ShareAlike} 4.0 International ({CC} {BY}-{NC}-{SA} 4.0)},
 issn = {2820-4042},
 year = {2022} 
}
```
- [BSNLP: 3rd Shared Task on SlavNER](http://bsnlp.cs.helsinki.fi/shared-task.html)
 
  We merged 2017+2021 train data with 2021 test data and made custom train / dev / test splits. 
  
  We also mapped EVT (event) and PRO (product) tags to MISC to align the corpus with others.
  
  You can change mappings running a custom prepare corpus step (see above).

- [Training corpus hr500k 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1183)

```
@misc{11356/1183,
    title = {Training corpus hr500k 1.0},
    author = {Ljube{\v s}i{\'c}, Nikola and Agi{\'c}, {\v Z}eljko and Klubi{\v c}ka, Filip and Batanovi{\'c}, Vuk and Erjavec, Toma{\v z}},
    url = {http://hdl.handle.net/11356/1183},
    note = {Slovenian language resource repository {CLARIN}.{SI}},
    copyright = {Creative Commons - Attribution-{ShareAlike} 4.0 International ({CC} {BY}-{SA} 4.0)},
    issn = {2820-4042},
    year = {2018} 
}
```
- [Training corpus SETimes.SR 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1200)

```
@misc{11356/1200,
    title = {Training corpus {SETimes}.{SR} 1.0},
    author = {Batanovi{\'c}, Vuk and Ljube{\v s}i{\'c}, Nikola and Samard{\v z}i{\'c}, Tanja and Erjavec, Toma{\v z}},
    url = {http://hdl.handle.net/11356/1200},
    note = {Slovenian language resource repository {CLARIN}.{SI}},
    copyright = {Creative Commons - Attribution-{ShareAlike} 4.0 International ({CC} {BY}-{SA} 4.0)},
    issn = {2820-4042},
    year = {2018} 
}
```

- [Massively Multilingual Transfer for NER.](https://github.com/afshinrahimi/mmner) nick-named WikiAnn
```
@inproceedings{rahimi-etal-2019-massively,
    title = "Massively Multilingual Transfer for {NER}",
    author = "Rahimi, Afshin  and
      Li, Yuan  and
      Cohn, Trevor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1015",
    pages = "151--164",
}
```

- [Neural Networks for Featureless Named Entity Recognition in Czech.](https://github.com/strakova/ner_tsd2016)

```
@Inbook{Strakova2016,
    author="Strakov{\'a}, Jana and Straka, Milan and Haji{\v{c}}, Jan",
    editor="Sojka, Petr and Hor{\'a}k, Ale{\v{s}} and Kope{\v{c}}ek, Ivan and Pala, Karel",
    title="Neural Networks for Featureless Named Entity Recognition in Czech",
    bookTitle="Text, Speech, and Dialogue: 19th International Conference, TSD 2016, Brno , Czech Republic, September 12-16, 2016, Proceedings",
    year="2016",
    publisher="Springer International Publishing",
    address="Cham",
    pages="173--181",
    isbn="978-3-319-45510-5",
    doi="10.1007/978-3-319-45510-5_20",
    url="http://dx.doi.org/10.1007/978-3-319-45510-5_20"
}
```

### NER Evaluation

For evaluation, we use [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval)
```
@misc{seqeval,
    title={{seqeval}: A Python framework for sequence labeling evaluation},
    url={https://github.com/chakki-works/seqeval},
    note={Software available from https://github.com/chakki-works/seqeval},
    author={Hiroki Nakayama},
    year={2018},
}
```

Which is based on
```
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
```

## News Framing Detection

### Prep
```
./nf prep aussda -p *******
./nf prep slomcor -p *******
```

### Split
```
./nf split -u manual aussda -p *******
```

### Train
By default, only a body field is used for the training, and subset of frame tags is selected based on corpora:
```
./nf train binrel --tqdm -e 20 -b 24 -u eco -p mcbert aussda_manual
./nf train binrel --tqdm -e 20 -b 24 -u eco -p xlmrb aussda_manual

./nf train binrel --tqdm -e 20 -b 24 -p mcbert aussda_manual &>train-binrel-macro-xlmrb.log
./nf train binrel --tqdm -e 20 -b 24 -p xlmrb aussda_manual &>train-binrel-macro-mcbert.log

./nf train lpset --tqdm -e 20 -b 24 -p mcbert aussda_manual &>train-lpset-macro-xlmrb.log
./nf train lpset --tqdm -e 20 -b 24 -p xlmrb aussda_manual &>train-lpset-macro-mcbert.log
```
For usage of an additional text fields you have to enumerate them:
```
./nf train binrel --tqdm -e 20 -b 24 -p mcbert -f title,body aussda_manual &>train-binrel-macro-tb-xlmrb.log
./nf train binrel --tqdm -e 20 -b 24 -p xlmrb -f title,body aussda_manual &>train-binrel-macro-tb-mcbert.log

./nf train lpset --tqdm -e 20 -b 24 -p mcbert -f title,body aussda_manual &>train-lpset-macro-tb-xlmrb.log
./nf train lpset --tqdm -e 20 -b 24 -p xlmrb -f title,body aussda_manual &>train-lpset-macro-tb-mcbert.log
```

For usage of different average metric for best model selection you have to specify it:
```
./nf train lpset --tqdm -e 20 -b 24 -m micro-1 -p xlmrb aussda_manual &>train-lpset-micro1-xlmrb.log
./nf train lpset --tqdm -e 20 -b 24 -m micro-1 -p mcbert aussda_manual &>train-lpset-micro1-mcbert.log
./nf train lpset --tqdm -e 20 -b 24 -m micro-1 -p xlmrb -f title,body aussda_manual &>train-lpset-micro1-tb-xlmrb.log
./nf train lpset --tqdm -e 20 -b 24 -m micro-1 -p mcbert -f title,body aussda_manual &>train-lpset-micro1-tb-mcbert.log

./nf train binrel --tqdm -e 20 -b 24 -m micro-1 -p xlmrb aussda_manual &>train-binrel-micro1-xlmrb.log
./nf train binrel --tqdm -e 20 -b 24 -m micro-1 -p mcbert aussda_manual &>train-binrel-micro1-mcbert.log
./nf train binrel --tqdm -e 20 -b 24 -m micro-1 -p xlmrb -f title,body aussda_manual &>train-binrel-micro1-tb-xlmrb.log
./nf train binrel --tqdm -e 20 -b 24 -m micro-1 -p mcbert -f title,body aussda_manual &>train-binrel-micro1-tb-mcbert.log
```

### Test
Epochs and pretrained model names are used just for model name computation.
```
# the same would be: ./nf test binrel --tqdm -b 24 -n mcbert.e20.b24.l2e-05.aussda_manual.f-b aussda_manual
# so it is shorter to construct model name from training params 
./nf test binrel --tqdm -e 20 -b 24 -p mcbert aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -p xlmrb aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -m micro-1 -p mcbert aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -m micro-1 -p xlmrb aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -p mcbert aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -p xlmrb aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -m micro-1 -p mcbert aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -m micro-1 -p xlmrb aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -f title,body -p mcbert aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -f title,body -p xlmrb aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -f title,body -m micro-1 -p mcbert aussda_manual
./nf test binrel --tqdm -e 20 -b 24 -f title,body -m micro-1 -p xlmrb aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -f title,body -p mcbert aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -f title,body -p xlmrb aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -f title,body -m micro-1 -p mcbert aussda_manual
./nf test lpset --tqdm -e 20 -b 24 -f title,body -m micro-1 -p xlmrb aussda_manual
./nf test majority0 aussda_manual
./nf test majority_labeled aussda_manual
./nf test random aussda_manual

# the same would be: ./nf test binrel --tqdm -b 24 -n mcbert.e20.b24.l2e-05.aussda_short.f-b aussda_short
./nf test binrel --tqdm -e 20 -b 24 -p mcbert aussda_short
./nf test binrel --tqdm -e 20 -b 24 -p xlmrb aussda_short
./nf test lpset --tqdm -e 20 -b 24 -p mcbert aussda_short
./nf test lpset --tqdm -e 20 -b 24 -p xlmrb aussda_short

./nf test binrel --tqdm -e 20 -b 24 -u eco,lab,wel,sec -p mcbert aussda_short
./nf test binrel --tqdm -e 20 -b 24 -u eco,lab,wel,sec -p xlmrb aussda_short
./nf test lpset --tqdm -e 20 -b 24 -u eco,lab,wel,sec -p mcbert aussda_short
./nf test lpset --tqdm -e 20 -b 24 -u eco,lab,wel,sec -p xlmrb aussda_short
./nf test majority -u eco,lab,wel,sec aussda_short
./nf test random -u eco,lab,wel,sec aussda_short
```


Todo 
- convert test results json to tsv/csv
- infer xmodel on other corpora