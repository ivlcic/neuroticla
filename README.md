# neuroticla

Library for neural networks based text classification and processing.

## Initialize the project

### Single time initialization
```
cd ~/projects
git clone https://github.com/ivlcic/neuroticla.git
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
(single time, optional)

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
(single time, before each desired data-split training / fine-tuning)

The following command:
```
./ner split -s 80:10 sl hr sr bs mk sq cs bg pl ru sk uk
```
will produce 80%% train, 10%% evaluation and 10%% test data set size:
```
data/ner/split/
├── sl_hr_sr_bs_mk_sq_cs_bg_pl_ru_sk_uk.eval.csv
├── sl_hr_sr_bs_mk_sq_cs_bg_pl_ru_sk_uk.test.csv
└── sl_hr_sr_bs_mk_sq_cs_bg_pl_ru_sk_uk.train.csv
```
(this is also the default if -s switch is omitted)

Each language corpora is combined, shuffled and split. The *train / eval / test* splits are concatenated from all languages in order as specified in command line.
By default the data shuffle is reproducible.

For all options see: 
```
./ner split --help
```

### Train
```
./ner train -l 2e-5 -e 40 -b 20 xlmrb sl hr sr bs mk sq cs bg pl ru sk uk
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