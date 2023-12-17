# EMMA 1mio corpus

## Description

Initially, the corpus was created by choosing the most representative clients engaged in news monitoring within a specific industry or domain for each country. 
The selection of clients was also influenced by the volume of news monitoring, following the principle that more coverage would enhance the corpus. 
The objective was to ensure the diversity of the corpus in terms of news content.  
Subsequently, a one-year (2023) time frame was chosen in consideration of the tasks associated with the EMMA project, which involve long-term and large-scale news analysis.
We were aiming for the larger corpus size also because potentially used LLM usually need large amounts of data for training.

## Properties

The corpus is primarily composed of news articles from three main countries: Slovenia, Serbia, and Macedonia. 
Additionally, reflecting the socio-economic connections in the region, certain articles also stem from neighboring countries such as Croatia, Bosnia and Herzegovina, Montenegro, Kosovo, and Albania.

The period of the collected articles is 1.1.2023 - 15.12.2023.

Mostly ISO 639-1 language codes are used for marking news article language, but sometimes format is extended following the BCP 47 standard for region and scripts.
(for instance, the code: sr-ME-Cyrl would mean Sebian language from Montenegro region in Cyrillic script.)
When utilizing the corpus, it is advisable not to rely entirely on the language tag or script tag, as the language is initially inherited from the media outlet, which may employ both Latin and Cyrillic scripts.
Likewise, the ISO 3166-1 two-letter country code is employed to indicate the country of origin for articles. However, in instances where the actual country of origin is ambiguous, the country code primarily designates the location where the article is published.

The Industry / Domain tags were chosen arbitrarily and have several problems, so they *should not be used* for research until further improved.  
The main problems are:
- Arbitrarily selected and probably biased, assigned w/o any consensus.
- Tags are assigned based on the client's field of operation rather than their field of interest.  
  (for example: article was collected for the Energy Industry, but it has nothing to do with the Energy Industry because it contains content about government politics which is of interest for the Energy Industry)
- The actual news content can be from completely different domain  
  (see the example above)
- Their sole purpose was to distinguish fields of interest in the corpus selection.

## Format

The corpus consists of twelve Bzip2 compressed tar archives each containing individual month of media monitoring news articles.
SHA1 checksums are also given so that the corpus integrity can be preserved.
Each archive contains the following directory (YYYY/MM/DD) structure :
```
2023
├── 01
│   ├── 01
│   │   ├── 00046d8f-89cb-11ed-bfff-6f1f3528f840.json
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
- Precomputed embeddings of the title and body sections with the E5 and OpenAI's ada-002 models.
- Text statistics.
- Arbitrary client tags unique identifiers in a flattened tree structure.

News articles are stored in the following JSON format:
```
{
  "country": {
    "name": "RS",  // The ISO 3166-1 county code
    "uuid": "a8bc9db9-922f-34ad-86d4-dfe1305d7db1"  // Country's unique identifier 
  },
  "created": "2023-12-13T20:09:52.182Z",  // Coutntry's unique identifier 
  "language": "sr",  // The BCP 47 language tag
  "rubric": {
    "name": "TV program",  // Rubric, Publication Section, or Programme name
    "uuid": "8d304a49-5af0-48b2-8b3e-68b82fa7cf30"  // Rubric's unique identifier 
  },
  "media": {
    "name": "Kurir",  // Publication - Media outlet name
    "uuid": "0d5272a6-6bb7-47f8-9423-9db85badf550",  // Media's unique identifier 
    "mediaReach": 265159,
    "type": {
      "name": "print",  // The Type of Publication - Media outlet
      "uuid": "e45422e6-82df-3490-93e8-ab35f5f6e499"  // The Type's unique identifier
    }
  },
  "published": "2023-12-13T23:00:00.000Z",  // The publication date, broadcast date time
  "uuid": "93432e0e-99f3-11ee-b156-8135109a7851",   // Article's unique identifier 
  "body": {
    "text": "26 I Zabava TVprogram KURIR NAJUTICAJNUE DNEVNE NOVINE BALKANA ČETVRTAK14. DECEMBAR 2023.  ' hhh fhphhhhh hHH HHVHIHH I _J*     H rokonebo21.00LaraKrofl: IM&1 f I >1 I t h  J VjMFfH 19.55 Usijarije 11.08 Gore-dole 23.00 Trenutak iz sna 15.15 Aviondžije Pjačkaš grobnica23.05Na- [  \\ * J 1 f | j kon I HS HHHHH I I I h lflHHlH IH UUDID RTSl PINK B92 STARCRIME H DmmMmmmmm  m 06.05 Jutarnjiprogram 05.30 Novojutro 07.00 Dvaipomuškarca ru07.40D2ekTejior 09.40 084)0 Jutarrji dnevnik 114M) Praktičnažena 07.30 Kviz Štoperica detekiivi 10.40 Bui 06.00 Selo H.08 Gore-dole 12.00 Premijera 06.00 Crtani filmovi H.3S -ht r-.i  „„„„ gon.ababasečeslia 12 00 Dnevnik 12.15 Ekskluzlvno 09.30 Prvlputsocem ZT.l° r9lJ3 0,«7ejn -': n.00 Redakcga 13.00 jedandobardan 12.30 Elita 09.45 TV pnodaja: Biostile TUraiVnm U'Hi-11 3  Plli,SrMea,1ekSPreS 14,44 Trag 13,00 PrviracionaW 1ft00 Savršenrecept someruSzMnačklumo-    H P pJ pHFJFRH j!'uu [-uisbrDije 15.19 OvojeSrbjja dnevnik 10.80 Želimdatikažem vi 20.05 Bui 21.05 FBI: Naj-  D /*  _ r ’ I V / _ 1 TT7H \" “ Rarne uz rame 15.59 Selogori, 13.30 Elita 12.00 Vesti traženiji 22.00 ViiTrent  H . 4 v4 V l   I . \\ H1 I I 9  D 14.N M jke 1 snajke a baba se češlja 14.55 Nacionalni 124)0 Tate doxtv Wh H m00 Jacaodsudbme ,700 Dnevnik dnevnik 13.30 TVprodaja-Telestar o?10 Extrume Evurest S'S Stars RTVojvodina 15.30 Nevina 13.45 Kviz Stoperica oaloocostotwar 09.50 TO,UU 17.20 Šta radite, bre 16.50 Elita 14.15 Koio sreće Air rescue 10.15 Bodyback gori, a baba se ćeslja -17,42 Beogradska 18,30 I 15.15 I I 16.55 Popodnevm dnevmk hronika 194)0 Fatalna 16.00 Vesti history 12.45 life, 1215 KvizMozaik 18,18 Okomagazin Ijubav 16.30 SportskipregledB92 Jeath&mon8y13.35Thein-17.45 Stars 18.54 Slagaliea 20.30 Osveta 17.00 Fokus ntsncenemrk 14.20 Bloo-M Maikeisnajke 19.30 Dnevnik 224)0 Braknaneviđeno 17.30 Zbornica . ITUisDOra 'liU 5« Sni'enidneVn'k 20,10 Selogori. 234)0 Trenutakizsna 18.15 Kviz Stoperica money 16.45 tneimocence ra.ao usganje a baba se Cešlja 00.00 Ellta - gledanje 18.45 Vesti network 17.35 Bloodllne de- f  u,w .  eio 2103 Četvrtkom u devet snimaka 19.00 Kuhinja tectives 18.25 Nurseswho   jl     H gori.ababase cesya 224)2 Svadba iz mog kr j'a 1945 Dvaipomuškarca kiii 19.15 Body hack 20.00 Hk ' hI  H '*85' 21.40 Vecernjidnevmk 23.05 Dnevnik 20.45 Vesti Nature’sfury21.00 Forbid-22.H) Jača od sudbine 2332 Kulturni dnevnik 21.00 FILM Posiednja bitka den history  D . HH -l  H 22.40 Rameuzrame 23.47 Dekster 23.15 Vesti HBO H ML, 23 0 Montevideo, 23.% Sportski pregled B92 06.00 Izgubjeni kraj 07.45 H i bogtevideo 23.45 Ostrvo Ijubavi - rijalrti šou Mravci 09.10 Božićna iskra m Arenafieš 10.00 NBA 11.45 PRVA HAPPY KLASIK SUPERSTAR na:Lajpcig-Jangbojs14.00 06.06 EkSkluziv 04.30 Ranojutro 06.00 FILMOperacija 06.40 Pevačica- Ligašamplcna.Crvenazve-06.25 Eksplozlv 05.50 Vestl Beorad reprlza mniona PreilSj1? 06.55 Jutro 06.00 Dobrojutro, 07.25 Dok. serjja SFRJ za 08.35 FILM Patuljci sa 1800 -1:11845 11.10 TV prodaja: Biostile Snbijo početnike naslovnih strana ga konferencje: Cenk - Čuka-TI.25 Opnoštajno 09.55 Vesti 08.55 Dok.fiimTreći 10.20 FILM Čovek bljesak rički21.00LigaEvrope:Olimpismo 10.45 Provodadžija 18.10 FILM Šolaja 12.00 FILMBeliambis pijakos - TSC 23.00 Emisja: 13.00 Vesti 11.40 TVprodaja 1155 FILMVIakbezvoznog 14.20 FILM Skoro sasvim UEL studio 23.30 liga Evro-13.10 1155 Vesti reda 16.05 igra sudbine 12.(K) Provodadžija 13.50 FILM Promeni me 15.50 FILM Uzimanje života RTV1 17.00 Ekskiuziv 1150 Ženskaroš 15.35 FILM Žena sa 17.40 FILMKonačnaanaiiza 06.30 Dobrojutro, Vojvodi-17.30 KvizPobednik 13.40 Braćnitrougao slomljenim nosom 20.00 Pevačica- nc 10.05 Paieta 11.05 Do-18.00 Vesti 14.38 Telemaster 17.20 FILM Daleko nebo repriza program 11.35 19.00 Eksploziv 14.45 Posleručka 18,45 RLMKiša 21.00 Pevačica Frankofonua 12.07 19.20 Andrija 17.20 Telemaster 204)0 RLM Partizanska 22.00 FILM Požar u Kentakiju KZT T'EtnVn iAndelka 18.08 Kviz Naslovnastrana eskadrila 23.55 FILMBumerang 14 os' zr c  0-14.32 20. igrasudbine 18.35 Provodadžjja 22.10 RLM Sex - partjjski Naukaprivredi 15.00 Agrod-21. Urgentnicentar 19.45 Aktuelnosti neprijatelj br. 1 nevnik 15.30 Graditelji No- HHHIIIII HaUH    H 22. Hotel Beograd 2130 Muzički specjjal 23.40 RLM Cefuri raus vogsada 16.00 Muzeji vo-23.00 EkSkiuziv 234)0 Ženskaroš 0128 RLMAspik- jvodine 16J0 Srpski ekran H B ISfSHHi HHHHIHHHH 23.25 Eksploziv 23.50 Braćnitrougao lošasudbina 17.00TVdnevmk 17.20Pra-   23.50 FILM Noć u starom 0135 Velika porodica 03.05 RLMKiša vi ugao 18.00 Razgiedmce 04.20 RLM u zoru enja 22.57 kuiture HHHHHHHHHHHHHHHHHHHHI JV’JflfllfB Etedakrja:Vlq|kovlćeva8, Beograđ.*S81116357-100. redakc(ja@>kurir.rs I ci= Ktitaiogiztici|,iijwji>iKucji Kurr.usiamoa  »Irflffl Prodaja: *38111635-70-31, proda]a@kurir.rsfSlatnpa; APM Print Beograd I cbr MVM 51 c'-Bl's-s«  ftrm-nrfrin l2(la-afi: f 1 MONDO Moncio Inc d.o.o. ~' '!   I V ajkoviieva 8, Ueograd Direktar: Dragan Milić Pjbiii'iing Diroctor: Darko Kaeavenda GlavniurodnilcflleksandarĐonclović Zamunicaglavnogurednfc: Branislava Majdarevic Odgavcniurudmk: Rajko Nedić MarKetmg: marketingfaiadriamedia.rs ?38t 114420-462 Šufštampanagizdanja: Ivančorbić Urednioi štampanog zdanja: Rpjko Nedić. Branislav Bjelica Dcžurm urcdnicinavvebj: Katarina LazićSimić. Miaden Radulović Pcrtar'rans rncnadžcr digitaln luanala: flieksandarKrunič Advertising: advertising( adriamedia.rs *381114420-304 Noćni uredrik štampanog zdanja: Momčilo Petrović Politika: Jelena Velinović Društvo: flleksandar Mitrović Hronika: Marjja Ivanov Piancta: Nemanja Vlačo Stars: Sandra Rilak Biznis: Jelena Skenderija Kjltura: Ljubomir Radanov AdiTimstrcDija: ađministnacija@adriamedia.rs *381116357-025 Zabava: LidUaStoisavljević Sport: Miloš BjelinićStil: VanjaMilenković Dodatak Stil: NatašaBajat Dodata<TVE<ran: Jasmina Antonijević-Miloševićšc\"'cto s jžbc: DadoĐilasšcfprcoma: Uroš Corbić",
    "stats": {
      "chr": 5842,  // The number of characters in a given (body) section
      "sent": 15,  // The number of sentences in a given section
      "w_t": 1044,  // The number of words in a given section
      "sp_t": 2154,  // The number of Sentence Piece encoded sub-word tokens in a given section
      "oai_t": 2774,  // The number of cl100k_base encoded sub-word tokens in a given section
      "filter": false  // Was the content filtered prior to embedding computation
    }
  },
  "title": {
    "text": "TV PROGRAM",
    "stats": {
      ...
    }
  },
  "embed_e5": [...],  // The E5 model title + body embeddings
  "embed_oai": [...],  // The OpenAi's ada-002 model title + body embeddings
  ,
  "stats": {  // text statistics summed over all article sections
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
      "uuid": "229b6dca-de61-46f4-8c75-ee7f39e1ff5e",  // Arbitrary client tag / topic
      "tags": [
        {
          "refUuid": "85822f5e-e7a9-3a23-9fe9-3ec2a96230f4",  // Client's tag / topic group
          "tags": [
            {
              "refUuid": "34dd1f20-4753-3295-82be-7c1a11116149",  // Client's Parent group (could be removed in the future)
            }
          ]
        }
      ],
      "type": "topic"
    }
  ]
```

An index file (EMMA_1mio-v1.0-index.cvs) was constructed from corpus, for easier:
- Navigation
- Sub-corpus selection
- Statistic analysis

Lastly the tag_industries_map.csv is a mapping from clients (actually client's topic groups) to a selected industry / domain.
