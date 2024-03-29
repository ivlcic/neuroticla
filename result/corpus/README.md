# EMMA 1mio corpus
(document version 1.2.1)

## Description

The corpus consists of about 10% of news articles collected by the monitoring system. It was created by choosing the most representative clients engaged in news monitoring within a specific industry or domain for each country.
The selection of clients was also influenced by the volume of news monitoring, following the principle that more coverage would enhance the corpus.
The objective was to ensure the diversity of the corpus in terms of news content.  
Subsequently, a one-year (2023) time frame was chosen considering the tasks associated with the EMMA project, which involve long-term and large-scale news analysis.  
We were also aiming for a larger corpus size because potentially using large language models usually needs large data for training or fine-tuning.

## Properties

The corpus is primarily composed of news articles from three main countries: Slovenia, Serbia, and Macedonia. 
Additionally, reflecting the socio-economic connections in the region, certain articles also stem from neighbouring countries such as Croatia, Bosnia and Herzegovina, Montenegro, Kosovo, and Albania.

The period of the collected articles is 1.1.2023 - 15.12.2023.

Mostly, ISO 639-1 language codes are used for marking news article language, but sometimes, the format is extended following the BCP 47 standard for regions and scripts.  
(for instance, the code sr-ME-Cyrl would mean the Serbian language from the Montenegro region in Cyrillic script.)  
When utilizing the corpus, it is advisable not to rely entirely on the language tag or script tag, as the language is initially inherited from the media outlet, which may employ Latin and Cyrillic scripts.  
Likewise, the ISO 3166-1 two-letter country code indicates the country of origin for articles. However, in instances where the actual country of origin is ambiguous, the country code primarily designates the location where the article is published.  
For the social media articles the non-standard arbitrary country code "GO" (for global) is used (for technical legacy reasons).

The Industry / Domain tags were chosen arbitrarily and have several problems, so they *should not be used* for research until further improved.  
The main problems are:
- Arbitrarily selected and probably biased, assigned w/o any consensus.
- Tags are assigned based on the client's field of operation rather than their field of interest.  
  (for example, the article was collected for the Energy Industry, but it has nothing to do with the Energy Industry because it contains content about government politics, which is of interest to the Energy Industry)
- The actual news content can be from a completely different domain  
  (see the example above)
- Their sole purpose was to diversify corpus content selection as much as possible.

The broadcast transcripts from TV and Radio Media have usually synthetic data added in the article body section:
 - The media outlet name is followed by the programme and the date of broadcast in the first line of the body text.
 - The speaker's name is before each spoken passage of text.  

This data is removed when computing embeddings, and the article's body section is marked with the `filtered` boolean JSON property.



## Format

The corpus comprises twelve Bzip2 compressed tar archives, each containing a month of selected news articles. SHA1 checksums are also given so that the corpus integrity can be preserved.  
Each archive contains the following directory (Year/Month/Day/unique_id.json) structure:
``` bash
2023  # year of retrieval
├── 01  # month of retrieval
│   ├── 01  # day of retrieval
│   │   ├── 00046d8f-89cb-11ed-bfff-6f1f3528f840.json  # individual article file 
│   │   ├── 0011d7a9-89ad-11ed-8e94-37c1e99ec6c5.json
│   │   ├── 0034059b-89ad-11ed-ab84-05da78aebe3b.json
│   │   ├── 003dc946-89d2-11ed-8e94-37c1e99ec6c5.json
```

Each news article data consists of:
- Unique identifier.
- Date of retrieval and publication.
- Title and body section.
- Media outlet and its country.
- Rubric, Publication Section, or Programme name.
- Precomputed embeddings of the title and body sections with the Multilingual-E5-base and OpenAI's ada-002 models.
- Text statistics.
- Arbitrary client tags unique identifiers in a flattened tree structure.

Individual news articles are stored in the following JSON format:
```json
{
  "country": {
    "name": "RS",
    // The ISO 3166-1 county code
    "uuid": "a8bc9db9-922f-34ad-86d4-dfe1305d7db1"
    // Country's unique identifier 
  },
  "created": "2023-12-13T20:09:52.182Z",
  // The date and time when an article was retrieved
  "language": "sr",
  // The BCP 47 language tag
  "rubric": {
    "name": "TV program",
    // Rubric, Publication Section, or Programme name
    "uuid": "8d304a49-5af0-48b2-8b3e-68b82fa7cf30"
    // Rubric's unique identifier 
  },
  "media": {
    "name": "Kurir",
    // Publication/Media outlet name
    "uuid": "0d5272a6-6bb7-47f8-9423-9db85badf550",
    // Media's unique identifier 
    "mediaReach": 265159,
    "type": {
      "name": "print",
      // The Type of Publication/Media outlet
      "uuid": "e45422e6-82df-3490-93e8-ab35f5f6e499"
      // The type's unique identifier
    }
  },
  "published": "2023-12-13T23:00:00.000Z",
  // The date time of publication/broadcast
  "uuid": "93432e0e-99f3-11ee-b156-8135109a7851",
  // Article's unique identifier 
  "body": {
    "text": "26 I Zabava TVprogram KURIR NAJUTICAJNUE DNEVNE NOVINE BALKANA ČETVRTAK14. DECEMBAR 2023.  ' hhh fhphhhhh hHH HHVHIHH I _J*     H rokonebo21.00LaraKrofl: IM&1 f I >1 I t h  J VjMFfH 19.55 Usijarije 11.08 Gore-dole 23.00 Trenutak iz sna 15.15 Aviondžije Pjačkaš grobnica23.05Na- [  \\ * J 1 f | j kon I HS HHHHH I I I h lflHHlH IH UUDID RTSl PINK B92 STARCRIME H DmmMmmmmm  m 06.05 Jutarnjiprogram 05.30 Novojutro 07.00 Dvaipomuškarca ru07.40D2ekTejior 09.40 084)0 Jutarrji dnevnik 114M) Praktičnažena 07.30 Kviz Štoperica detekiivi 10.40 Bui 06.00 Selo H.08 Gore-dole 12.00 Premijera 06.00 Crtani filmovi H.3S -ht r-.i  „„„„ gon.ababasečeslia 12 00 Dnevnik 12.15 Ekskluzlvno 09.30 Prvlputsocem ZT.l° r9lJ3 0,«7ejn -': n.00 Redakcga 13.00 jedandobardan 12.30 Elita 09.45 TV pnodaja: Biostile TUraiVnm U'Hi-11 3  Plli,SrMea,1ekSPreS 14,44 Trag 13,00 PrviracionaW 1ft00 Savršenrecept someruSzMnačklumo-    ... AdiTimstrcDija: ađministnacija@adriamedia.rs *381116357-025 Zabava: LidUaStoisavljević Sport: Miloš BjelinićStil: VanjaMilenković Dodatak Stil: NatašaBajat Dodata<TVE<ran: Jasmina Antonijević-Miloševićšc\"'cto s jžbc: DadoĐilasšcfprcoma: Uroš Corbić",
    "stats": {
      // text statistics in a given (body) section
      "chr": 5842,
      // The number of characters 
      "sent": 15,
      // The number of sentences
      "w_t": 1044,
      // The number of words
      "sp_t": 2154,
      // The number of Sentence-Piece encoded sub-word tokens
      "oai_t": 2774,
      // The number of cl100k_base encoded sub-word tokens
      "filter": false
      // Was the content filtered prior to embedding computation
    }
  },
  "title": {
    "text": "TV PROGRAM",
    "stats": {
      ...
    }
  },
  "embed_e5": [
    ...
  ],
  // The E5 model title + body embeddings with 'passage: ' prefix.
  "embed_oai": [
    ...
  ],
  // The OpenAi's ada-002 model title + body embeddings
  "stats": {
    // text statistics summed over both article sections
    "chr": 5852,
    "sent": 16,
    "w_t": 1046,
    "sp_t": 2160,
    "oai_t": 2776
  },
  "tags": [
    {
      "uuid": "85822f5e-e7a9-3a23-9fe9-3ec2a96230f4"
    },
    {
      "uuid": "34dd1f20-4753-3295-82be-7c1a11116149"
    },
    {
      "uuid": "229b6dca-de61-46f4-8c75-ee7f39e1ff5e",
      // Arbitrary client tag/topic
      "tags": [
        {
          "refUuid": "85822f5e-e7a9-3a23-9fe9-3ec2a96230f4",
          // Client's tag/topic group
          "tags": [
            {
              "refUuid": "34dd1f20-4753-3295-82be-7c1a11116149"
              // Client's Parent group (could be removed in the future)
            }
          ]
        }
      ],
      "type": "topic"
    }
  ]
}
```

## Version 1.1 changes
* `Keyword Expression` reverse match was added:  
  Topics are typically chosen based on the match of a Keyword Expression, as recommended by the system to a human editor. 
  However, in version 1.0, the specific sections of the text that led to the suggestion of a particular topic are not recorded. 
  To identify the relevant parts of the text, the matching process was performed again, 
  but this time only on topics that had been manually verified. 
  The `Keyword Expression` itself is recorded together with matching text spans in appropriate JSON sections. 
  Matching was conducted separately for the title and the body of the text.
  Marked matched `Keyword Expression` can now be a source for *Sentiment Target* discovery or learning.
* Simplified `tags` section:
  We removed unnecessary tags and tag information, and incorporated two JSON properties: `sentiment` and `parent`. 
  The parent property indicates the group to which the topic is associated, and it may also include the sentiment property.
* Added `IPTC` manual mapping (`kwe_iptc_map.csv`) between `Keyword Expression` and `IPTC` tags where applicable: 

### A sample with changes:
```json
{
  ...
  "body": {
    "text": "Predsednik Republike Srbije Aleksandar Vučić sutra će, povodom obeležavanja početka rada Gasnog interkonektora Srbija-Bugarska, ugostiti u Nišu\n\npredsednika Republike Azerbejdžan Ilhama Alijeva, predsednika Republike Bugarske Rumena Radeva i ambasadora Evropske unije u Srbiji Emanuela Žiofrea.\n\n  Tokom boravka u Nišu, planirani su i odvojeni bilateralni susreti predsednika Aleksandra Vučića sa predsednicima Republike Azerbejdžan i Republike Bugarske, saopštila je predsednikova Služba za saradnju s medijima.\n\n  Predviđen je i prikaz opreme i sposobnosti jedinica Vojske Srbije – protivdronskih PVO sistema, bespilotnih i upravljanih vazduhoplovnih i zemaljskih borbenih platformikao i sposobnosti padobranskih i specijalnih jedinica, dodaje se u saopštenju.\n\n  Završetkom radova na gasovodu do Bugarske omogućiće se sigurnije i stabilnije snabdevanje Srbije gasom iz različitih izvora, posebno iz Azerbejdžana.\n\n  Izgradnja novog gasovoda u dužini od 109 kilometara od Niša do Dimitrovgrada trajala je 12 meseci. Ukupna dužina dvosmernog gasovoda Srbija-Bugarska je 170 kilometara, od Novog Iskara, ublizini Sofije, do Niša.\n\n  Srbija je potpisala ugovor o interkonekciji sa Bulgartransgasom i komercijalni ugovor o snabdevanju do 400 miliona kubnih metara gasa iz Azerbejdžana do kraja 2024. godine.\n\n  Ukupna vrednost gasnog interkonektora Niš – Dimitrovgrad – Bugarska, iznosi 85,5 miliona evra. Projekat je finansiran delom kroz bespovratnu pomoć EU, u visini od 49,6 miliona evra, kao i 25 miliona evra povoljnog kredita Evropske investicione banke koji će Srbija plaćati za deo gasovoda koji prolazi kroz Srbiju.\n\n  Republika Srbija je za ovaj projekati nterkonekcije Srbija-Bugarska izdvojila 15 miliona evra i 7,5 miliona evra za pripremne radove i projektovanje.\n\n  Gas čini 13 odsto potrošnje primarne energije u Srbiji i do sada je mogao da se dobije samo sa dve ulazne tačke, preko gasovoda Balkanski tok i iz Mađarske.\n\n  UDARNA VEST.rs (Blic zvor: Tanjug)",
    "stats": {
      ...
    },
    "matches": {
      "kwe": {
        "0d776998-f284-4dbc-a72d-725b2ca19ce6": {
          "topic": {
            "uuid": "3ef00f78-759f-4906-86dc-882f0d2ac662",
            "name": "Emanuele Giaufret"
          },
          "expr": "\"emanuel* giaufre*\" OR \"emanuel* žofre*\" OR \"емануел* жофре*\" OR \"emanu* žiofr*\" OR \"емануел* жиофр*\" OR \"šef* delegacij* evrops* unij*\" OR \"шеф* делегациј* европс* униј*\" OR \"emanue* ziofre*\""
        },
        "6abd7e13-043d-4a0d-9966-7c48a145f769": {
          "topic": {
            "uuid": "3ef00f78-759f-4906-86dc-882f0d2ac662",
            "name": "Emanuele Giaufret"
          },
          "expr": "\"emanuel* giaufre*\" OR \"emanuel* žofre*\" OR \"емануел* жофре*\" OR \"emanu* žiofr*\" OR \"емануел* жиофр*\""
        },
        "fc4c593d-9450-400e-a35b-96685b65d0e8": {
          "topic": {
            "uuid": "67ebcac5-fc7d-48ab-85ce-031e785ade55",
            "name": "EIB"
          },
          "expr": "\"evrop* investici* ban*\" OR \"евро* инвести* бан*\" OR eib OR еиб OR EIB OR ЕИБ"
        }
      },
      "spans": [
        {
          "start": 277,
          "end": 293,
          "kwe": [
            "0d776998-f284-4dbc-a72d-725b2ca19ce6",
            "6abd7e13-043d-4a0d-9966-7c48a145f769"
          ]
        },
        {
          "start": 1531,
          "end": 1558,
          "kwe": [
            "fc4c593d-9450-400e-a35b-96685b65d0e8"
          ]
        }
      ]
    }
  },
  "title": {
    "text": "Sutra počinje sa radom gasni Gasni interkonektor Srbija-Bugarska! Predsednik Vučić ugostiće Alijeva i Radeva u Nišu",
    "stats": {
      ...
    },
    "matches": {
      "kwe": {},
      "spans": []
    }
  },
  "embed_e5": [
    ...
  ],
  "embed_oai": [
    ...
  ],
  "stats": {
    ...
  },
  "tags": [
    {
      "uuid": "3ef00f78-759f-4906-86dc-882f0d2ac662",
      "type": "topic",
      "name": "Emanuele Giaufret",
      "parent": {
        "uuid": "1a9482ea-2222-3f7a-93e0-0b3c0c76797b",
        "sentiment": 1
      }
    },
    {
      "uuid": "67ebcac5-fc7d-48ab-85ce-031e785ade55",
      "type": "topic",
      "name": "EIB",
      "sentiment": 0,
      "parent": {
        "uuid": "a2e17897-4ebc-3132-a41d-b9361a4c404f"
      }
    }
  ],
  "ver": "1.1c"
}
```


## Corpus index

For convenience, an index file (`EMMA_1mio-v1.0-index.cvs`) was constructed from the corpus article files for quicker and easier:
- Navigation
- Sub-corpus selection
- Statistic analysis

Lastly, the `tag_industries_map.csv` is a file which contains a mapping from clients (actually clients' topic groups) to a selected industry/domain.

## Statistics

Corpus statistics can be observed and extended by using the Jupyter Notebook `ìndex-stats.ipynb`.
