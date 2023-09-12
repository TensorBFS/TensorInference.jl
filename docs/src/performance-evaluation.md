# Performance evaluation

The graph below illustrates a comparison of the runtime performance of
*TensorInference.jl* against *Merlin* [^marinescu2022merlin], *libDAI*
[^mooij2010libdai], and *JunctionTrees.jl* [^roa2022partial][^roa2023scaling]
libraries. Both *Merlin* and *libDAI* have previously participated in UAI
inference competitions [^gal2010summary][^gogate2014uai], achieving favorable
results. Additionally, we compared against two versions of *JunctionTrees.jl*,
the predecessor of *TensorInference.jl*. The first version does not employ
tensor technology, while the second version optimizes individual sum-product
computations using tensor-based technology. The experiments were conducted on
an Intel Core i9--9900K CPU @3.60GHz with 64 GB of RAM.

```@eval
using TikzPictures

tp = TikzPicture(
  L"""
  \begin{axis}[
    axis line style={black!100},
    every axis label/.append style={black!100},
    every tick label/.append style={black!100},
    xmin={0},
    xmax={23},
    xlabel={Largest cluster size},
    xmajorgrids={true},
    ymin={0},
    ymax={1000000},
    ymode={log},
    ytick={0.001,0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0},
    ymajorgrids={true},
    ylabel={Run time speedup},
    label style={font={\footnotesize}},
    tick label style={font={\footnotesize}},
    scatter/classes={Alchemy={mark={x}}, CSP={mark={+}}, Grids={mark={asterisk}}, ObjectDetection={mark={-}}, Pedigree={mark={triangle}}, Promedus={mark={o}}, Segmentation={mark={Mercedes star}}, linkage={mark={diamond}}},
    legend style={legend columns={3}, at={(0.51,-0.4)}, anchor={south}, text={black!100}, draw={none}, fill={none}, font={\footnotesize}, column sep={1.5}}
    ]
    \addplot[
      c01,
      scatter,
      only marks,
      scatter src={explicit symbolic},
      legend image post style={black!100},
      legend style={text={black!100}, font={\footnotesize}}
    ]
      table[row sep={\\}, meta={label}]
      {
        x  y  label  \\
        4  2.8892732905905696  Promedus
        \\
        5  0.9992454677080572  Promedus
        \\
        5  2.7291490240125063  Promedus
        \\
        6  162.92710213915234  ObjectDetection
        \\
        6  143.18430666866695  ObjectDetection
        \\
        6  720.0876395098279  ObjectDetection
        \\
        6  217.1158405786517  ObjectDetection
        \\
        6  147.2467303223847  ObjectDetection
        \\
        6  170.3349686317474  ObjectDetection
        \\
        6  734.1273502948045  ObjectDetection
        \\
        6  144.59866923029833  ObjectDetection
        \\
        6  728.2931567239458  ObjectDetection
        \\
        6  138.79291257761554  ObjectDetection
        \\
        6  122.47593563280027  ObjectDetection
        \\
        6  139.85747966450083  ObjectDetection
        \\
        6  131.525522176854  ObjectDetection
        \\
        6  109.92499089134364  ObjectDetection
        \\
        6  93.72644489626983  ObjectDetection
        \\
        6  98.16016046784863  ObjectDetection
        \\
        6  119.60310364630723  ObjectDetection
        \\
        6  99.38321389928038  ObjectDetection
        \\
        6  111.86522014674959  ObjectDetection
        \\
        6  107.77481251809532  ObjectDetection
        \\
        6  145.1282395173713  ObjectDetection
        \\
        6  115.19858632266963  ObjectDetection
        \\
        6  127.8729029741368  ObjectDetection
        \\
        6  113.83264841569968  ObjectDetection
        \\
        6  104.52638074276405  ObjectDetection
        \\
        6  120.60503585256899  ObjectDetection
        \\
        6  157.24044669598547  ObjectDetection
        \\
        6  1.735522638791712  Promedus
        \\
        7  87.41255368349063  ObjectDetection
        \\
        7  61.100818944217814  ObjectDetection
        \\
        7  59.69805794504825  ObjectDetection
        \\
        7  70.73189288893803  ObjectDetection
        \\
        7  89.85785905088217  ObjectDetection
        \\
        7  58.767478375123105  ObjectDetection
        \\
        7  76.18253022438975  ObjectDetection
        \\
        7  63.31661349350759  ObjectDetection
        \\
        7  80.99589372165947  ObjectDetection
        \\
        7  73.48000793615601  ObjectDetection
        \\
        7  80.27055974802306  ObjectDetection
        \\
        7  93.61510822180446  ObjectDetection
        \\
        7  74.34768113755483  ObjectDetection
        \\
        7  91.49324265602368  ObjectDetection
        \\
        7  58.23811725652532  ObjectDetection
        \\
        7  67.8636215125312  ObjectDetection
        \\
        7  66.88380696001799  ObjectDetection
        \\
        7  102.59080909072124  ObjectDetection
        \\
        7  66.56958714507779  ObjectDetection
        \\
        7  64.88399445151073  ObjectDetection
        \\
        7  103.00893328957005  ObjectDetection
        \\
        7  47.13832914874443  ObjectDetection
        \\
        7  93.73141917401033  ObjectDetection
        \\
        7  71.55797645109784  ObjectDetection
        \\
        7  1.6560698511486507  Promedus
        \\
        8  204.72477237476792  ObjectDetection
        \\
        8  233.565856236582  ObjectDetection
        \\
        8  215.28901794948294  ObjectDetection
        \\
        8  229.87199069175702  ObjectDetection
        \\
        8  229.3782738272767  ObjectDetection
        \\
        8  212.38623403103352  ObjectDetection
        \\
        8  216.56418414636724  ObjectDetection
        \\
        8  223.96604816048549  ObjectDetection
        \\
        8  217.6183690413624  ObjectDetection
        \\
        8  196.37344051838508  ObjectDetection
        \\
        8  193.2116345753146  ObjectDetection
        \\
        8  211.63391758733627  ObjectDetection
        \\
        8  40.21682216100781  ObjectDetection
        \\
        9  2.2444568475573097  Promedus
        \\
        10  2.1682214059321203  Promedus
        \\
        10  2.395524138036038  Promedus
        \\
        10  2.4861136742851597  Promedus
        \\
        10  2.4712883148229494  Promedus
        \\
        10  2.5224688486540745  Promedus
        \\
        10  3.9094546843521214  Promedus
        \\
        11  1.1598952750759095  Grids
        \\
        11  2.1255363481394944  Promedus
        \\
        12  27.496325534646022  CSP
        \\
        12  3.9276304925464993  Promedus
        \\
        14  9.4309349986777  Segmentation
        \\
        14  8.235520718654  Segmentation
        \\
        14  7.181641313605316  Segmentation
        \\
        14  13.065928506218675  Segmentation
        \\
        14  27.239668393243996  Segmentation
        \\
        14  86.773394425308  Promedus
        \\
        15  2.5595014327825845  CSP
        \\
        15  9.494023540394732  Segmentation
        \\
        15  5.751575020073473  Promedus
        \\
        15  4.811965216436699  Promedus
        \\
        16  4.8432353347307275  Promedus
        \\
        17  15.5208026591168  Pedigree
        \\
        17  29.752075146393935  Pedigree
        \\
        17  31.313714482974373  Pedigree
        \\
        18  59.481525499227224  Promedus
        \\
        19  81.68307801261165  Promedus
        \\
        19  103.61473164461938  Promedus
        \\
        19  195.08181377958132  Promedus
        \\
        19  0.24372535034158538  linkage
        \\
        20  9.852758983135821  CSP
        \\
        20  10.157012271476857  Grids
        \\
        20  10.487716623947392  Grids
        \\
        20  7.996566553618832  Grids
        \\
        20  106.36365676541155  Alchemy
        \\
        20  144.73529291612388  Promedus
        \\
        20  73.06837243861509  Promedus
        \\
        21  113.86474560092745  Grids
        \\
        21  154.0677864901418  Grids
        \\
        21  164.04458229131768  Grids
        \\
        21  162.48832232318276  Grids
        \\
        21  331.3409948307013  Promedus
        \\
        21  90.35951391708664  Promedus
        \\
        22  267.0753196031133  Promedus
        \\
      }
      ;
    \addplot[c02, scatter, only marks, scatter src={explicit symbolic}]
      table[row sep={\\}, meta={label}]
      {
        x  y  label  \\
        4  0.04976723515430697  Promedus
        \\
        5  0.10973550584252717  Promedus
        \\
        5  0.09728915125826393  Promedus
        \\
        6  51.595787075092744  ObjectDetection
        \\
        6  50.18066801604769  ObjectDetection
        \\
        6  182.73168469378987  ObjectDetection
        \\
        6  73.58299486979577  ObjectDetection
        \\
        6  48.323981291927055  ObjectDetection
        \\
        6  58.11205129100487  ObjectDetection
        \\
        6  181.52824011098107  ObjectDetection
        \\
        6  50.31473817789115  ObjectDetection
        \\
        6  180.40017622102155  ObjectDetection
        \\
        6  47.316034211689185  ObjectDetection
        \\
        6  44.1135158535562  ObjectDetection
        \\
        6  60.49337042552758  ObjectDetection
        \\
        6  58.570380589144584  ObjectDetection
        \\
        6  45.44581442796541  ObjectDetection
        \\
        6  41.73230487235904  ObjectDetection
        \\
        6  41.714095540305344  ObjectDetection
        \\
        6  53.97256707715049  ObjectDetection
        \\
        6  41.05105202031919  ObjectDetection
        \\
        6  48.460639044535746  ObjectDetection
        \\
        6  45.9883253244037  ObjectDetection
        \\
        6  60.851776840786655  ObjectDetection
        \\
        6  49.888315306766785  ObjectDetection
        \\
        6  46.44500985213657  ObjectDetection
        \\
        6  47.872867495623325  ObjectDetection
        \\
        6  41.08513880202735  ObjectDetection
        \\
        6  50.501103323289584  ObjectDetection
        \\
        6  69.24695331746624  ObjectDetection
        \\
        6  0.10855723575770325  Promedus
        \\
        7  38.43144906854903  ObjectDetection
        \\
        7  28.824476697901407  ObjectDetection
        \\
        7  28.444098134760218  ObjectDetection
        \\
        7  33.321195122767435  ObjectDetection
        \\
        7  40.97423848801853  ObjectDetection
        \\
        7  27.053452566852428  ObjectDetection
        \\
        7  35.63704539023769  ObjectDetection
        \\
        7  30.725906566347057  ObjectDetection
        \\
        7  34.50500511476716  ObjectDetection
        \\
        7  35.85101409281535  ObjectDetection
        \\
        7  39.728908735714064  ObjectDetection
        \\
        7  39.24986878193817  ObjectDetection
        \\
        7  36.804407318047666  ObjectDetection
        \\
        7  39.00305515862831  ObjectDetection
        \\
        7  28.72695925908947  ObjectDetection
        \\
        7  33.979161951615644  ObjectDetection
        \\
        7  33.37325966632402  ObjectDetection
        \\
        7  42.73317930107966  ObjectDetection
        \\
        7  33.02674757973232  ObjectDetection
        \\
        7  31.899956589280716  ObjectDetection
        \\
        7  43.842662064464825  ObjectDetection
        \\
        7  23.018843686012243  ObjectDetection
        \\
        7  39.76117759639337  ObjectDetection
        \\
        7  23.338927808776177  ObjectDetection
        \\
        7  0.086168959904781  Promedus
        \\
        8  127.15336326780849  ObjectDetection
        \\
        8  146.7894315177654  ObjectDetection
        \\
        8  134.9880922783158  ObjectDetection
        \\
        8  147.78324479842817  ObjectDetection
        \\
        8  148.76455249133056  ObjectDetection
        \\
        8  132.68096165657985  ObjectDetection
        \\
        8  138.01146775480282  ObjectDetection
        \\
        8  143.09619087847668  ObjectDetection
        \\
        8  140.26119882812065  ObjectDetection
        \\
        8  128.06490022749406  ObjectDetection
        \\
        8  120.44440568197183  ObjectDetection
        \\
        8  133.78244741474938  ObjectDetection
        \\
        8  18.09780043602283  ObjectDetection
        \\
        9  0.14481202881592561  Promedus
        \\
        10  0.19188493884062746  Promedus
        \\
        10  0.7940853891780273  Promedus
        \\
        10  1.174243541277038  Promedus
        \\
        10  1.221189552523608  Promedus
        \\
        10  0.4403443150309808  Promedus
        \\
        10  0.11424861033800494  Promedus
        \\
        11  0.33344955953025335  Grids
        \\
        11  0.39272746458467694  Promedus
        \\
        12  6.6717372376159485  CSP
        \\
        12  0.6810795190128051  Promedus
        \\
        14  7.670703564835005  Segmentation
        \\
        14  4.498126989473214  Segmentation
        \\
        14  10.803654369852513  Segmentation
        \\
        14  7.005397987563722  Segmentation
        \\
        14  14.597150566338783  Segmentation
        \\
        14  539.1457833526977  Promedus
        \\
        15  1.4506064147708677  CSP
        \\
        15  12.69886496855729  Segmentation
        \\
        15  50.656787506219274  Promedus
        \\
        15  98.86313077858442  Promedus
        \\
        16  12.253186932761135  Promedus
        \\
        17  22.053043449344454  Pedigree
        \\
        17  4.133925837523455  Pedigree
        \\
        17  13.07915435717768  Pedigree
        \\
        18  36.82052186359802  Promedus
        \\
        19  9.897409713752268  Promedus
        \\
        19  299.63559119909456  Promedus
        \\
        19  39.49467819173893  Promedus
        \\
        19  1.1521882975029056  linkage
        \\
        20  0.6884474603364397  CSP
        \\
        20  4.392829136914002  Grids
        \\
        20  5.122665540616931  Grids
        \\
        20  3.5718247841982387  Grids
        \\
        20  5608.097560603573  Alchemy
        \\
        20  374.30465114749524  Promedus
        \\
        20  46.74698541495396  Promedus
        \\
        21  66.27290996753283  Grids
        \\
        21  91.20421069072923  Grids
        \\
        21  96.7421688523645  Grids
        \\
        21  93.84149351851156  Grids
        \\
        21  287.2876139515749  Promedus
        \\
        21  243.16961804253052  Promedus
        \\
        22  723.1816308816418  Promedus
        \\
      }
      ;
    \addplot[c03, scatter, only marks, scatter src={explicit symbolic}]
      table[row sep={\\}, meta={label}]
      {
        x  y  label  \\
        4  0.14925059102554106  Promedus
        \\
        5  0.14327283479821357  Promedus
        \\
        5  0.14939608149799202  Promedus
        \\
        6  11.65659526627016  ObjectDetection
        \\
        6  10.934355494090733  ObjectDetection
        \\
        6  37.82980519325721  ObjectDetection
        \\
        6  16.40289875241257  ObjectDetection
        \\
        6  10.545904912753045  ObjectDetection
        \\
        6  12.980514028759048  ObjectDetection
        \\
        6  39.01521697350239  ObjectDetection
        \\
        6  11.08580867500881  ObjectDetection
        \\
        6  38.252711653120734  ObjectDetection
        \\
        6  10.346665001809031  ObjectDetection
        \\
        6  11.126735619813932  ObjectDetection
        \\
        6  15.494049951417956  ObjectDetection
        \\
        6  14.732760915367122  ObjectDetection
        \\
        6  11.53986280410882  ObjectDetection
        \\
        6  10.702941741014117  ObjectDetection
        \\
        6  10.406574717062897  ObjectDetection
        \\
        6  14.04369673049766  ObjectDetection
        \\
        6  10.502890586516207  ObjectDetection
        \\
        6  12.412829707488566  ObjectDetection
        \\
        6  11.365689075173039  ObjectDetection
        \\
        6  16.202155537407094  ObjectDetection
        \\
        6  13.073740567616822  ObjectDetection
        \\
        6  11.690273215514795  ObjectDetection
        \\
        6  12.519845282109127  ObjectDetection
        \\
        6  10.416777104291414  ObjectDetection
        \\
        6  13.428594931609437  ObjectDetection
        \\
        6  18.179734226875908  ObjectDetection
        \\
        6  0.15261336274202075  Promedus
        \\
        7  9.950444014836263  ObjectDetection
        \\
        7  6.010595240211144  ObjectDetection
        \\
        7  5.874117634433757  ObjectDetection
        \\
        7  7.027357920014467  ObjectDetection
        \\
        7  10.677601496563058  ObjectDetection
        \\
        7  5.683127750944496  ObjectDetection
        \\
        7  7.660687755402418  ObjectDetection
        \\
        7  7.750348431627795  ObjectDetection
        \\
        7  10.104386143409299  ObjectDetection
        \\
        7  9.012504890658171  ObjectDetection
        \\
        7  9.923901057575499  ObjectDetection
        \\
        7  11.82454003591559  ObjectDetection
        \\
        7  9.253082975711099  ObjectDetection
        \\
        7  11.928863766302216  ObjectDetection
        \\
        7  7.098854534330338  ObjectDetection
        \\
        7  8.38465606047645  ObjectDetection
        \\
        7  8.181924699338992  ObjectDetection
        \\
        7  12.963559485537495  ObjectDetection
        \\
        7  8.42625819018594  ObjectDetection
        \\
        7  8.055206951184065  ObjectDetection
        \\
        7  12.67120214876367  ObjectDetection
        \\
        7  5.736560607385999  ObjectDetection
        \\
        7  11.872915779937728  ObjectDetection
        \\
        7  7.637733293591384  ObjectDetection
        \\
        7  0.1501605655060659  Promedus
        \\
        8  61.080281524101416  ObjectDetection
        \\
        8  70.35564083670576  ObjectDetection
        \\
        8  65.0646316297441  ObjectDetection
        \\
        8  69.84386950179862  ObjectDetection
        \\
        8  71.58183896948957  ObjectDetection
        \\
        8  63.7884017460642  ObjectDetection
        \\
        8  65.20744445120046  ObjectDetection
        \\
        8  67.97030465542156  ObjectDetection
        \\
        8  67.51435207678718  ObjectDetection
        \\
        8  59.79339462912366  ObjectDetection
        \\
        8  58.3632483853267  ObjectDetection
        \\
        8  64.52509673009989  ObjectDetection
        \\
        8  6.6031313636307924  ObjectDetection
        \\
        9  0.17319117221512492  Promedus
        \\
        10  0.2275553381629561  Promedus
        \\
        10  0.34290590657479375  Promedus
        \\
        10  0.26612528467086627  Promedus
        \\
        10  0.26450177736681896  Promedus
        \\
        10  0.2765103745635581  Promedus
        \\
        10  0.16767739558528016  Promedus
        \\
        11  0.5053784414906094  Grids
        \\
        11  0.43471037094094744  Promedus
        \\
        12  3.2777122536034278  CSP
        \\
        12  0.49076268896461656  Promedus
        \\
        14  2.9943352124985916  Segmentation
        \\
        14  2.976943847123529  Segmentation
        \\
        14  2.141426816782795  Segmentation
        \\
        14  2.6652580560344696  Segmentation
        \\
        14  3.9746761898446628  Segmentation
        \\
        14  3.21084948699377  Promedus
        \\
        15  1.2469295150627302  CSP
        \\
        15  5.149044429436219  Segmentation
        \\
        15  3.8124963256954456  Promedus
        \\
        15  2.537133473152447  Promedus
        \\
        16  3.775445361215019  Promedus
        \\
        17  9.301937517261578  Pedigree
        \\
        17  8.562852153703005  Pedigree
        \\
        17  9.148128988567924  Pedigree
        \\
        18  23.261101521631982  Promedus
        \\
        19  15.394222824706796  Promedus
        \\
        19  28.431795623899347  Promedus
        \\
        19  26.896570718586002  Promedus
        \\
        19  0.07020498289629841  linkage
        \\
        20  1.783234446980031  CSP
        \\
        20  10.667015866609162  Grids
        \\
        20  12.296281482726384  Grids
        \\
        20  8.374407026764578  Grids
        \\
        20  14611.571676660204  Alchemy
        \\
        20  79.66400246926366  Promedus
        \\
        20  25.29833290997319  Promedus
        \\
        21  29.74750830183507  Grids
        \\
        21  40.4017737662268  Grids
        \\
        21  42.91628608886999  Grids
        \\
        21  41.57373577358043  Grids
        \\
        21  64.2427265803294  Promedus
        \\
        21  59.1002175804972  Promedus
        \\
        22  489.2030533478738  Promedus
        \\
      }
      ;
    \addplot[c04, scatter, only marks, scatter src={explicit symbolic}]
      table[row sep={\\}, meta={label}]
      {
        x  y  label  \\
        4  0.3746078056579958  Promedus
        \\
        5  0.39770357077308965  Promedus
        \\
        5  0.41727978430655255  Promedus
        \\
        6  8.83552576158785  ObjectDetection
        \\
        6  8.299956982765655  ObjectDetection
        \\
        6  17.147306935218477  ObjectDetection
        \\
        6  12.4763787103948  ObjectDetection
        \\
        6  7.960469693967034  ObjectDetection
        \\
        6  10.043202942993146  ObjectDetection
        \\
        6  17.496546030332993  ObjectDetection
        \\
        6  8.44298635263838  ObjectDetection
        \\
        6  16.894073327243657  ObjectDetection
        \\
        6  7.880243683887004  ObjectDetection
        \\
        6  2.0183448286845977  ObjectDetection
        \\
        6  2.703325201115473  ObjectDetection
        \\
        6  2.578704214284112  ObjectDetection
        \\
        6  2.3629595702908177  ObjectDetection
        \\
        6  1.8591955679998455  ObjectDetection
        \\
        6  2.085629178405993  ObjectDetection
        \\
        6  2.4181815142043153  ObjectDetection
        \\
        6  1.9071323770919433  ObjectDetection
        \\
        6  2.18884397411018  ObjectDetection
        \\
        6  2.0829565816441864  ObjectDetection
        \\
        6  2.823271360548588  ObjectDetection
        \\
        6  2.3190901230876806  ObjectDetection
        \\
        6  2.114790847714359  ObjectDetection
        \\
        6  2.1998911928515423  ObjectDetection
        \\
        6  1.9547104262217916  ObjectDetection
        \\
        6  2.28573262672064  ObjectDetection
        \\
        6  3.1903787215020563  ObjectDetection
        \\
        6  0.4301565999745164  Promedus
        \\
        7  1.423330760130917  ObjectDetection
        \\
        7  1.2323356884087273  ObjectDetection
        \\
        7  1.2958865509943351  ObjectDetection
        \\
        7  1.538061382446274  ObjectDetection
        \\
        7  1.4282314336936388  ObjectDetection
        \\
        7  1.276502804170552  ObjectDetection
        \\
        7  1.6217129504182368  ObjectDetection
        \\
        7  1.5174838312779253  ObjectDetection
        \\
        7  1.0880128637926745  ObjectDetection
        \\
        7  1.7887919360067777  ObjectDetection
        \\
        7  1.9907848253611329  ObjectDetection
        \\
        7  1.2544119172477382  ObjectDetection
        \\
        7  1.7668603858883114  ObjectDetection
        \\
        7  1.2399949754557344  ObjectDetection
        \\
        7  1.3835187267375706  ObjectDetection
        \\
        7  1.6235673933455617  ObjectDetection
        \\
        7  1.5773721466574788  ObjectDetection
        \\
        7  1.392974353859367  ObjectDetection
        \\
        7  1.5860136495801964  ObjectDetection
        \\
        7  1.5757672831986314  ObjectDetection
        \\
        7  1.3996556369753421  ObjectDetection
        \\
        7  1.1042203066582073  ObjectDetection
        \\
        7  1.2451586576052875  ObjectDetection
        \\
        7  6.699133654829738  ObjectDetection
        \\
        7  0.42853681041529246  Promedus
        \\
        8  10.80442952184336  ObjectDetection
        \\
        8  12.348299160463945  ObjectDetection
        \\
        8  11.36380045230718  ObjectDetection
        \\
        8  12.339629389305586  ObjectDetection
        \\
        8  12.538207380503694  ObjectDetection
        \\
        8  11.536405764488366  ObjectDetection
        \\
        8  11.869135922221297  ObjectDetection
        \\
        8  12.272664551601661  ObjectDetection
        \\
        8  12.149553633329234  ObjectDetection
        \\
        8  10.540852443952083  ObjectDetection
        \\
        8  10.242382204270152  ObjectDetection
        \\
        8  11.268995261030028  ObjectDetection
        \\
        8  1.7940610647262243  ObjectDetection
        \\
        9  0.44475356965738383  Promedus
        \\
        10  0.46736320536957465  Promedus
        \\
        10  0.4729010373413127  Promedus
        \\
        10  0.44346673227335254  Promedus
        \\
        10  0.44400421393689843  Promedus
        \\
        10  0.45492971841053587  Promedus
        \\
        10  0.4482841822729594  Promedus
        \\
        11  0.4017198206803196  Grids
        \\
        11  0.4971727593791871  Promedus
        \\
        12  0.6649791291508202  CSP
        \\
        12  0.5156988876045215  Promedus
        \\
        14  0.6264304751755524  Segmentation
        \\
        14  0.6863121836610862  Segmentation
        \\
        14  0.4702560990041832  Segmentation
        \\
        14  0.5785308939447695  Segmentation
        \\
        14  0.8785863166752916  Segmentation
        \\
        14  0.8819076854069103  Promedus
        \\
        15  0.3589231378284274  CSP
        \\
        15  0.8743514115666701  Segmentation
        \\
        15  0.6753917031340068  Promedus
        \\
        15  0.5692032451693735  Promedus
        \\
        16  0.6384060242066736  Promedus
        \\
        17  0.9754385698118018  Pedigree
        \\
        17  0.8817231999502103  Pedigree
        \\
        17  0.9576650538556629  Pedigree
        \\
        18  1.343051308292903  Promedus
        \\
        19  0.7330364775397645  Promedus
        \\
        19  1.2205566050153949  Promedus
        \\
        19  1.2596161623374778  Promedus
        \\
        19  0.002966625858796674  linkage
        \\
        20  0.7902712105015884  CSP
        \\
        20  0.7650048643625871  Grids
        \\
        20  0.8463123491738426  Grids
        \\
        20  0.5946012792113868  Grids
        \\
        20  22.90457157917797  Alchemy
        \\
        20  6.446709444016422  Promedus
        \\
        20  0.9810786697091944  Promedus
        \\
        21  2.3092430706405267  Grids
        \\
        21  3.196210414120729  Grids
        \\
        21  3.3833152403024003  Grids
        \\
        21  3.272489208513286  Grids
        \\
        21  3.3983549404745084  Promedus
        \\
        21  2.6398777210603632  Promedus
        \\
        22  57.17884340544315  Promedus
        \\
      }
      ;
      \draw[dashed, black] ({rel axis cs:1,0}|-{axis cs:0,1}) -- ({rel axis cs:0,0}|-{axis cs:0,1});
      \legend{{Alchemy},{CSP},{Grids},{ObjectDetection},{Pedigree},{Promedus},{Segmentation},{linkage}}
      \node 
      [text={black!100}, draw={black!100}, fill={white}, font={\scriptsize}]  at 
      (5.5,30000)
      {\shortstack[l] { $\textcolor{c01}{\blacksquare}$ libDAI \\ $\textcolor{c02}{\blacksquare}$ Merlin \\ $\textcolor{c03}{\blacksquare}$ JunctionTrees.jl-v1  \\ $\textcolor{c04}{\blacksquare}$ JunctionTrees.jl-v2}};
    \end{axis}
    % This command controls the size of the standalone doc
    % I'm using it in order to display the shadow
    % https://tex.stackexchange.com/a/424023/23046
    \useasboundingbox (-18mm,64mm) rectangle (76mm,-26mm); 
    \begin{pgfonlayer}{background}
      \draw[fill=white!80,blur shadow={shadow blur steps=5,shadow xshift=-1mm},fill=white!00] 
      ([xshift=-16mm,yshift=-24mm]current axis.south west) rectangle 
      ([xshift=5mm,yshift=4mm]current axis.north east);
    \end{pgfonlayer}
  """,
  options="transform shape, scale=2.0",
  preamble="\\input{" * joinpath(@__DIR__, "assets", "preambles", "performance-evaluation") * "}",
)
save(SVG("performance-evaluation"), tp)
```

```@raw html
<img src="../performance-evaluation.svg"  style="margin-left: auto; margin-right: auto; display:block; width=1500">
```

The benchmark problems are arranged along the x-axis in ascending order of
complexity, measured by the induced tree width. On average,
*TensorInference.jl* achieves a speedup of 11 times across all problems.
Notably, for the 10 most complex problems, the average speedup increases to 63
times, highlighting its superior scalability.

## References

[^marinescu2022merlin]:
    Radu Marinescu. Merlin. 2022. [Online]. Available: [https://www.ibm.com/opensource/open/projects/merlin/](https://www.ibm.com/opensource/open/projects/merlin/) [Accessed: 11 September 2023].

[^mooij2010libdai]:
    Joris M. Mooij. libDAI: A Free and Open Source C++ Library for Discrete Approximate Inference in Graphical Models. *Journal of Machine Learning Research*, 11:2169-2173, Aug 2010. [Online]. Available: [http://www.jmlr.org/papers/volume11/mooij10a/mooij10a.pdf](http://www.jmlr.org/papers/volume11/mooij10a/mooij10a.pdf).

[^gal2010summary]:
    Gal Elidan and Amir Globerson. Summary of the 2010 UAI approximate inference challenge. 2010. [Online]. Available: [https://www.cs.huji.ac.il/project/UAI10/summary.php](https://www.cs.huji.ac.il/project/UAI10/summary.php) [Accessed: 11 September 2023].

[^gogate2014uai]:
    Vibhav Gogate. UAI 2014 Probabilistic Inference Competition. 2014. [Online]. Available: [https://www.ics.uci.edu/~dechter/softwares/benchmarks/Uai14/UAI_2014_Inference_Competition.pdf](https://www.ics.uci.edu/~dechter/softwares/benchmarks/Uai14/UAI_2014_Inference_Competition.pdf) [Accessed: 11 September 2023].
