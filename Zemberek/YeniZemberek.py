from typing import List, Tuple
from zemberek import TurkishMorphology, TurkishTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from stop_words import get_stop_words
from collections import Counter


# Zemberek ile Türkçe metinleri köklerine ayırma ve kelime türlerini belirleme
def analyze_text(text: str, morphology: TurkishMorphology) -> List[Tuple[str, str]]:
    tokenizer = TurkishTokenizer.DEFAULT
    tokens = tokenizer.tokenize(text)
    analyzed_tokens = []
    for token in tokens:
        # Her bir token için ayrı ayrı analiz yapımı
        results = morphology.analyze(token.content)
        for result in results:
            analyzed_tokens.append((result.get_stem(), result.format_string()))
    return analyzed_tokens




# Metin ön işleme fonksiyonu
def preprocess_text(text: str, morphology: TurkishMorphology) -> str:
    if text is None:
        return ""  # None değeri alırsa boş bir dize döndür
    text = text.lower()  # Metni küçük harflere dönüştürme
    text = "".join([char for char in text if char.isalpha() or char.isspace()])
    
    # Durma kelimelerini çıkarma kısmı
    turkish_stopwords = get_stop_words("turkish")
    words = text.split()
    words = [word for word in words if word not in turkish_stopwords]
    
    analyzed_tokens = analyze_text(" ".join(words), morphology)
    lemmas = [lemma for lemma, pos in analyzed_tokens]
    return " ".join(lemmas)




def prepare_data(
    corner_texts: List[Tuple[str, str]], morphology: TurkishMorphology
) -> Tuple[List[str], List[str]]:
    texts, authors = zip(*corner_texts)
    preprocessed_texts = [preprocess_text(text, morphology) for text in texts]

    return preprocessed_texts, list(authors)





# Köşe yazılarını ve yazarlarını içeren eğitim verisi (örnek)
corner_texts = [
    (
        """Evet bugün, Ege Üniversitesi Eczacılık Fakültesi’nin emekli öğretim üyelerinden değerli bilim insanı Prof. Dr. Levent Kırılmaz’ın, özellikle gençler için çok ilginç ve yararlı bulduğum düşüncelerini paylaşacağım:
    Bugüne kadar eğitim kurumlarımızda, karar verici anlayış, gençlerimizin ödevlerden ve sınavlardan aldıkları notlar olmuştur. Ne yazık ki gençlerimizin günlük hayatlarındaki sorunlarına çözüm bulabilmeleri, aileleriyle-arkadaşlarıyla ve çevreleriyle doğru, güzel ve iyi ilişkiler içinde olabilmelerini sağlayacak kişisel, duygusal ve sosyal becerilere eğitim sistemimizde yeterince yer verilmemiştir. Grigoriy Petrov’un söylediği gibi: “Eğer gençliğin ruhunu bakımsız bir tarla gibi bırakırsak, orada yabani otlar ve dikenler biter!.. 
    Artık günümüzde üniversite öğrenimi ile edinilmiş akademik bilgilerin “yaşam” için tek başına yeterli olmadığı kuşku götürmez bir gerçek. Üniversitelerin öncelikli görevi, gençlerimize sadece mesleki konularda bilgiler vermek değil, onları hayata hazırlayacak “hayata dair” bilgileri de sunmak, tecrübe ile donatmak ve topluma faydalı “etkin insan” olarak yetiştirmek olmalı.""",
        "Uğur Dündar",
    ),
    (
        """Ulaştırma Bakanı Abdulkadir Uraloğlu’nun, bakanlığına yüksek hızlı demiryolu hatları ve limanlar inşa eden Rönesans Holding’in özel uçağıyla Almanya’nın Leipzig kentindeki “Ulaştırma Forumu”na gitmesi, günlerdir kamuoyunda konuşuluyor.Haklı tepkiler üzerine Meclis’e bilgi veren Uraloğlu; firma ile bakanlığı arasında imzalanan sözleşmenin 31. maddesinin bu imkanı sağladığını belirterek, şunları söylüyor:
    Bakanlığa büyük iş yapan ilgili firmanın sözleşmesinde 31. maddede var. Ulaştırma alanındaki eğitim, sempozyum ile ilgili masraflar, bilabedel (bedelsiz) olmak üzere taraflarından karşılanır diye. Burada ne bir uçak kiralaması, ne de devletin ilave bir ücret ödemesi var. Böyle bir imkanla biz 4 günlük programı iki günle sınırlandırdık. Yani ne kiralama, ne ödeme, ne de müteahhit firmaya borçlu kalma vardır.""",
        "Uğur Dündar",
    ),
    (
        """Duydunuz mu? Suudi Arabistan 12 yıl aradan sonra Suriye’ye, başkent Şam’a büyükelçi atadı.Oysa aynı Suudi Arabistan, emperyalizmin  “Arap Baharı” adı altında pazarladığı Büyük Ortadoğu Projesi’nin Suriye’yi bölüp parçalama operasyonu başladığında Türkiye ile birlikte ABD’ nin en büyük destekçilerinden biriydi.
    Peki Suudi Arabistan, Suriye’nin parçalanmasını isteyen ABD’ye rağmen bu adımı atarken, Türkiye, çoktan yapması gereken benzer hamleyi neden sürekli erteliyor?
    Üstelik Türkiye ile Suriye ilişkileri normalleştiğinde; PKK’nın garnizon devlet kurması engelleneceği gibi, her geçen gün daha büyük sosyal ve ekonomik sorun olmaya başlayan sığınmacılar da ülkelerine döner.Böylece ülkemiz iki beka tehdidini bertaraf etmiş olur.Türkiye’yi yönetenlere Suriye Krizi’nin patlak verdiği tarihten bu yana uygulanan politikanın, ülkemizin Cumhuriyet tarihi boyunca yaptığı en büyük dış politika yanlışı olduğunu bir kez daha hatırlatmakta yarar var.Bunu ben söylemiyorum.Tüm öngörüleri doğru çıkan emekli Büyükelçi Şükrü Elekdağ, krizin ilk gününden itibaren dile getiriyor ve aşağıya alıntıladığım  “Bu vahim yanlıştan dönün” uyarısında bulunuyor:""",
        "Uğur Dündar",
    ),
    (
        """Dün bu köşede okudunuz. Suç örgütü üyelerinden birinin iddiasına göre; Ayhan Bora kaplan ile bir polis şefi, iki kez Ankara’dan İstanbul’a birlikte gitmişler. Dönüşlerinde Kaplan, 10’ar kilo kokain getirmiş!..Suç örgütü lideri Kaplan, emniyet, yargı ve siyaset üçgeninde kurduğu bağlantılarla kısa sürede Ankara’nın gece hayatına ve uyuşturucu trafiğine hakim olmuş.

Onun Ulus’ta telefon satışı yaptığı küçük bir dükkanla başlayan serüvenini okurken, yıllar öncesine, Türkiye’de ilk Alkol ve Madde Bağımlılığı Tedavi Merkezi”nin (AMATEM) açıldığı 1983 yılına gittim.Eski Sağlık Bakanlarından merhum Dr. Yıldırım Aktuna, Bakırköy Ruh ve Sinir Hastalıkları Hastanesi Başhekimliğine atandığında hastane, insanlık suçlarının işlendiği, sözcüklerle anlatılamayacak kadar kötü koşulların hakim olduğu bir toplama kampı gibiydi.""",
        "Uğur Dündar",
    ),
    (
        """Başarılı, cesur, yurtsever soruşturmacı gazeteci kardeşim Timur Soykan, yargılaması süren Ayhan Bora Kaplan olayındaki skandallar zincirini gözler önüne seren, bu arada benim de son dönemde neden iftira yağmurunun hedefi olduğumu ortaya koyan çok önemli bir araştırmaya daha imza atmış.Ayhan Bora Kaplan Olayı, sadece bir devlet krizini değil, devletin çürüdüğünü gözler önüne seriyor. MHP Genel Başkanı Devlet Bahçeli operasyonu kumpas olarak nitelendirdi. Ancak Cumhurbaşkanı Erdoğan darbe ve kumpas iddialarına katılmadı. Hatta bürokratik vesayet diyerek belki de darbe diyenleri işaret etti.Kaplan’ı gözaltına alan polisler geçen günlerde tutuklandı. Suç örgütü lideri Ayhan Bora Kaplan’a operasyon yapan polis müdürlerine siyasi iktidara darbe suçlamaları yöneltildi ve tutuklandılar. Acaba polis mi siyasilere operasyon yapacaktı yoksa mafya mı polise operasyon düzenledi? Sinan Ateş davasında hedef olan MHP ortağı AKP’ye karşı hamle mi yapıyor? Eski İçişleri Bakanı Süleyman Soylu, İçişleri Bakanı Ali Yerlikaya’yı mı hedef alıyor? Sorular çoğaltılabilir. Ama Ayhan Bora Kaplan olayındaki skandallar sadece bir devlet krizini değil, devletin çürüdüğünü gözler önüne seriyor. Bir mafya operasyonuna aşağıdaki 20 büyük skandal sığıyor:""",
        "Uğur Dündar",
    ),
    (
        """Dünkü yazımda daha önce AKP’de olup, son seçimlerde CHP’ye geçen belediyelerin büyük borçlar bıraktığını belirtmiş ve şu soruyu yöneltmiştim:Acaba borç rekortmeni belediye hangisi? Denizli mi, yoksa Balıkesir mi?..Çok geçmeden cevap geldi.

Balıkesir Büyükşehir Belediye Başkanı seçilen Ahmet Akın’a yakın kaynaklar, borç yükünün 15 milyar lirayı geçtiğinin belirlendiğini belirttiler.

Bu rekor rakam, Denizli’den 4 milyar TL daha fazla!..

O nedenle Ahmet Akın’ın işi zor.

Ancak hemşerileri, çok sevdikleri ve çalışkan bir siyasetçi olan yeni başkanın hem şeffaf bir yönetimle bu inanılmaz borç yükünü azaltacağına, hem de özlemini duydukları hizmetleri getireceğine inanıyorlar...Dün yaşanan bir başka olay da çok düşündürücü.

O da Seyhan’ın CHP’den seçilen yeni Belediye Başkanı Oya Tekin’e kurulan montajlı kumpası, Cumhurbaşkanı Başdanışmanı Oktay Saral’ın paylaşmış olması!..

Gerçi görüntünün montajdan ibaret olduğu ortaya çıkınca hem paylaşımını sildi hem de görevinden istifa etti.

Ancak olay, bazı AKP yandaşlarının seçimlerin propaganda döneminde sık sık baş vurdukları montaj, kumpas ve asparagas alışkanlıklarını sürdürdüklerini kanıtlaması açısından son derece vahim...""",
        "Uğur Dündar",
    ),
    (
        """Cumhuriyet Halk Partisi’ni (CHP) uzun yıllar sonra ilk kez birinci parti yapan seçim sonuçları dikkatlice analiz edildiğinde, seçmenin iktidara birçok mesaj verdiği görülüyor.Şöyle ki:

- Ekonomiyi bir an önce düzelt. “Faiz sebep, enflasyon sonuçtur” diyerek iktisat biliminde hiç yeri olmayan söylemlerle içinden çıkılmaz hale getirdiğin enflasyonun ezdiği geniş halk yığınlarını, en başta da emeklileri düşün. Onların sırtlarında taşımaktan iki büklüm oldukları ve yaşamdan soğudukları ağır hayat pahalılığı yükünü azalt. Milli gelirin dağılımındaki adaletsizliği gider, uçurumu ortadan kaldır. Hazine’nin kıt kaynaklarını yandaşların kayrıldığı ihaleler yerine; üretime, istihdama, yüksek teknoloji ürünleri ihracatına, kısacası döviz getirici yatırımlara yönlendir. Yapısal reformları erteleme. Tarım ve hayvancılıkla uğraşan köylüye, çiftçiye teşvik sağla. Öğretmenleri tekrar köy okullarına gönder. Bunları yaparken sadece sana oy verenleri kayırma, eşitlikçi ve adil davran. Yoksulluğu yöneterek iktidarını sürdürmenin çözüm getirmediğini, asıl yapman gerekenin yoksulluğu ortadan kaldırmak olduğunu göz önünde bulundur!..Ülkemizi hedef alan yabancı istilasına son ver. Gençlerin geleceklerini yabancı diyarlarda ve el kapısında aramalarına “dur” diyecek tüm önlemleri al. Yabancı istilası denilince akla gelen ilk isimlerden biri olan Bolu Belediye Başkanı Tanju Özcan’ın yeniden ve rekor sayılabilecek bir oyla seçilmesinin ardındaki nedenleri iyi oku. Gençleri kucakla, onlara eğitimde ve her alanda fırsat eşitliği sağla!.. Her ile bir üniversite açmakla eğitime kalitenin gelmediğini, bunun üniversite tabelası sayısını artırmaktan ve öğrencilerin ellerine birer kağıt parçasından ibaret mezuniyet belgeleri tutuşturmaktan öte bir işlevi olmadığını gör. Uluslararası geçerliliği olan tek bilimsel makalesi bulunmayan kişileri, yandaşın diye rektör yapmaktan vazgeç. Eğitimin içine düştüğü kalitesizlik sarmalını hiç vakit geçirmeden ortadan kaldırmak için gereken tüm bilimsel ve çağdaş önlemleri al!..Halkı kutuplaştırıp kamplara ayırarak oylarını konsolide etme gayretinin, seni sürgit iktidarda tutmaya yetmediğini artık anla. İttifakına oy vermeyenleri birer hasım, hatta düşman gibi gösterme gayretinden uzaklaş. Milleti yeniden kucaklaştırmaya gayret et! Halkın hür iradesine saygı duy ve seçimle işbaşına gelen yerel yöneticileri şu veya bu nedenle görevlerinden uzaklaştırmayı -yasal zorunluluk olmadıkça- asla deneme. Merkezi iktidarın gücünü kullanarak yerel yöneticilerin ellerini kollarını bağlayacak girişimlerde bulunmanın, senin adaylarının seçilmelerini sağlamadığı gibi, bu tür adaletsizliklerin tıpkı bumerang gibi dönüp seni vurduğunu gör!..Yurttaşların anayasadan aldıkları demokratik haklarını kullanmalarına izin ver. Anayasa’dan söz etmişken; Anayasa Mahkemesi’nin kararlarına saygı duy ve o kararları vakit geçirmeden uygula. Baskıcı yönetim anlayışını ve yargıyı senin gibi düşünmeyenlere karşı bir sopa olarak kullanma alışkanlığını artık geride bırak… Bu ülkede bir zamanlar düşünce ve basın özgürlüğünün var olduğunu hatırla!..""",
        "Uğur Dündar",
    ),
    (
        """Sözcü yazarı Uğur Dündar bu hafta 'Atatürk’ün dehasını ve büyüklüğünü bir gün herkes kabul edecek!..' başlıklı yazısını kaleme aldı.Çalışması 18 yıl sürdü...

1900-2000 yılları arasında 200 farklı ülkede yaşamış 1941 lideri araştırdı.

ABD Başkanı Roosvelt’ten, İngiltere’nin unutulmaz Başbakanı Sir Winston Churchill’e, Çin’in lideri Mao’dan, Küba devrimini yapan Fidel Castro’ya kadar uzayan geniş bir yelpazedeki 377 lider arasında en büyük puanı Atatürk’e verdi.Liderleri incelerken “Yeni ülke yaratma, savaş kazanma, toprak kazanımı/kaybı, iktidarda kalma süresi, askeri başarı, sosyal mühendislik, toplumu olumlu yönde değiştirme, iyileşen ekonomi, devlet adamlığı, ideoloji, ahlaklı örnek oluşturma, yolsuzluğa karışmamak, politik miras” gibi özelliklerine baktı.
Araştırmasını doğrulatmak için 80 kişilik bağımsız bir çalışma grubuna ayrıca benzer bir puanlama yaptırdı.


Her ikisinde de tüm liderler arasında birinciliği Atatürk aldı...Araştırmadaki önemli ölçeklerin başında “Hiç yoktan bir şey var etmek, örneğin sıfırdan bir devlet kurmak” geliyordu. Büyük Önder Atatürk burada en yüksek puana ulaştı. Çünkü bitmiş, yok olmuş bir Osmanlı’dan, yeni, modern Türkiye’yi yaratmış, Cumhuriyet’i yoktan var etmişti.

Atatürk Sevr Antlaşması ile kaybedilen toprakları Lozan Antlaşması ile kazandığı için yine, en yüksek puanı elde etti... Bir diğer olgu ise dış destek olmadan ve hukuksuzluğa başvurmadan uzun yıllar iktidarını korumasıydı. Büyük Önder hem bu alanda, hem de askeri başarıda diğer liderler arasından sıyrılıp, birinci çıktı... 

Çanakkale’de başlayıp Kurtuluş Savaşı ile doruğa çıkan askeri başarısıyla yine en yüksek puana layık görüldü...Harf, kıyafet, medeni kanun, kadına seçme-seçilme hakkı, laiklik gibi devrimlerin yanı sıra, ekonomik kalkınma hamlesi, devlet adamlığı, diplomatik ilişkileri geliştirip komşular ve diğer ülkelerle iyi geçinmesi, ideolojisi, siyasi mirası, ahlaki yönden örnek olması yani yozlaşmaması, çalıp çırpmaması, hanedan gibi hareket etmemesi gibi özellikleriyle de rakipsizdi.


Bir başka deyişle “Efsane lider” deyimi yerini bulmuştu.Bunları hayatı boyunca Türkiye’ye hiç gelmemiş olan ABD’li Psikiyatri Profesörü Arnold Ludwig, yıllarını adadığı “Political Greatness Scale” araştırması hakkında bilgi sunarken söylüyor...

Ayrıca unutturmak, itibarsızlaştırmak için Atatürk’ü ve eşsiz eseri Cumhuriyet’i hedef alan yalan ve iftira saldırılarını da şöyle yorumluyor:""",
        "Uğur Dündar",
    ),
    (
        """Sözcü yazarı Uğur Dündar bu hafta 'Tek kişilik en büyük siyasi parti!..' başlıklı yazısını kaleme aldı.Siyasallaşan yargı 14 Aralık 2022 günü, İstanbul Büyükşehir Belediyesi Başkanı Ekrem İmamoğlu’na 2 yıl 7 ay 15 gün hapis cezası vermiş ve siyaset yapmasını yasaklamıştı.

Ben bu tür kararların bumerang etkisi yaptığına inanırım.

Kısa bir süre için istenilen sonucu sağlamış gibi görünse de, bir süre sonra döner ve bu kararı alanları vurur.Peki geçmişte cezaevinde yatmış ve siyaseten yasaklanmış bir kişi olan Cumhurbaşkanı Erdoğan “bumerang gerçeğini” deneyimlemiş ve en iyi bilen bir politikacı olarak neden kendi iktidarında böyle bir karara ihtiyaç duymuş ve siyasi harakiriyi göze almış olabilir?


Cevabı çok basit:

Zorda, hem de çok zorda kaldığı için!..İktidarın İmamoğlu’nu siyaset sahnesinden silmek ve İstanbul Büyükşehir Belediye Başkanlığı koltuğuna kendi adamını oturtmak için siyasallaşan yargı eliyle aldığı bu karar, aynı zamanda Tayyip Erdoğan’ın Cumhurbaşkanlığı yarışında en çekindiği adayın İmamoğlu olduğunun da itirafı niteliğinde.


O halde “Altılı Masa’nın” yapması gereken; İstinaf Mahkemesi ve Yargıtay aşamalarını beklemeden bu resti görmek ve vakit geçirmeden Ekrem İmamoğlu’nu Cumhurbaşkanı adayı ilan etmek olmalı...

Gerisini iktidar düşünsün!..”Okuduğunuz satırları, bu köşede, Ekrem İmamoğlu’na cezanın verildiği ve Türkiye’nin cumhurbaşkanlığı ve milletvekili genel seçimlerine koştuğu günlerde yazdım.


Ancak Kemal Kılıçdaroğlu “Millet İttifakı” partilerine adaylığını dayattı ve sonuçta seçimi kaybetti.

Şimdi yine bir seçim var: Yerel yönetim seçimleri...""",
        "Uğur Dündar",
    ),
    (
        """Sözcü yazarı Uğur Dündar bu hafta 'Almanya’daki gurur tablosu...' başlıklı yazısını kaleme aldı.Son 15 günde, söyleşi ve imza günlerimiz için iki kez Almanya’ya gittim.

İlk durağım Frankfurt oldu. Salı günü de Kuzey Ren-Vestfalya Eyaleti’nin Gelsenkirchen-Schalke kentindeki etkinliğimize katılmak üzere Düsseldorf’a uçtum.

Özellikle son yolculuğumdan bazı çarpıcı gözlemlerimi sizlerle paylaşmak istiyorum.Örneğin Düsseldorf Havalimanı’na inip, pasaport kontroluna doğru yürürken, yanıma yaklaşan iki yolcu ile kısa sohbetlerimiz oldu.


İkisi de Türkiye’de meslek edinmiş bu Türk yolcular, Almanya’ya yerleşmek üzere geldiklerini söylediler.

Nedenini sorduğumda ikisi de artık demokratik, yargısı bağımsız ve ekonomisi istikrarlı bir ülkede özgürce ve korku duymadan yaşamak, çocuklarını geleceğe burada hazırlamak istediklerini söylediler.Gelsenkirchen’e ilk gelen Türkler, ağırlıklı olarak Zonguldak’daki kömür ocaklarında çalışan işçiler arasından seçilmişler.

Ruhr bölgesine yayılan kömür madenleriyle en ağır işlerde çalışmak için, sanki at satın alır gibi, dişleri kontrol edilerek, göğüslerine çarpı işareti konularak Almanya’ya getirilen (70’li yılların başlangıcı) ilk işçi kafilesinden bu yana, onlarla ilgili haberler, belgeseller yapıyorum.

Birinci kuşağın gurbetçileri, öylesine çok çalıştılar ki; Almanların 50 yılda çıkarabileceği kömürleri 15-20 yılda yeryüzü ile buluşturdular.

Ocaklar kapatılınca, sağlıklarını yüzlerce metre derinlikte bırakarak toprak üstüne çıkabildiler!..


Kazandıkları markları da, (Almanya’nın o yıllardaki para birimi) yemeyip, içmeyip yurda gönderdiler.

Türkiye ekonomisi döviz üretmekte zorlanınca o marklarla can simidi oldular.""",
        "Uğur Dündar",
    ),
    (
        """Sözcü yazarı Uğur Dündar bu hafta 'LÖSEV mucizesine imza atan Dr. Üstün Ezer, ağır yaralı Hatay’ı da ayağa kaldırmaya talip oldu...' başlıklı yazısını kaleme aldı.Küçük bir muayene odasından milyonlara uzanan hikayesini on binlerce hayata dokunarak ilmek ilmek işleyen LÖSEV (Lösemili Çocuklar Sağlık ve Eğitim Vakfı) çeyrek asrı geride bıraktı. Kurulduğu 1998 yılında 35 lösemili çocuğu tedavi ederek hizmete başlayan LÖSEV, bugün 75 bine yakın aileye, 6 milyonu aşkın gönüllüsü ile Türkiye’nin ve dünyanın en büyük gönüllü ağını oluşturarak yardım elini uzatıyor. Her geçen yıl bağışçılarından personeline, kayıtlı hastalardan gönüllülere, hizmet ağını kat be kat büyüten LÖSEV, bu süre içinde ülkemizin gurur duyulacak sağlık kurumlarından biri haline geldi.LÖSEV, lösemi ve kansere karşı istikrarlı bir şekilde sevgi, iyilik, umut mücadelesi veriyor.Sağlık hizmetlerinde alanında en yetkin sağlık kuruluşlarından biri olan Lösante Hastanesi teknolojinin gerektirdiği yeniliklerle lösemi tedavisinde yüzde 94 oranında başarıya imza atıyor. Hedef ise yüzde 100’e ulaşmak.Yoksulluğun yarattığı, o derin çaresizliğin kollarına hiçbir çocuğun bırakılmadan, minicik bedenlerindeki kocaman mücadelelerine rehberlik etmeyi, sevgiyle kucaklamayı ve en önemlisi iyileştirmeyi görev ediniyor. 1998’den bu yana yürüttüğü çalışmalarla toplumsal farkındalığa, dayanışmaya, iyiliğin yayılmasına ve sosyal politikalara katkıda bulunuyor.İnsanlığa ve bilime evrensel düzeyde katkıda bulunmayı amaçlayan LÖSEV, lösemi ve kanser mücadelesi veren çocuklarla gençlerin ücretsiz tedavilerini üstlenmenin yanı sıra, onları LSV Eğitim Kurumları, yaz kampları, “Canım Kardeşim Projesi” ve eğitim bursları ile destekleyerek; araştırmaya, eleştirmeye, öğrenmeyi öğrenmeye, yaratıcı fikirler geliştirmeye özendiriyor.


Böylece üst düzey ahlaki değerlerle donatılmış, aklı ve bilimi esas alan, toplumsal sorumluluklarının bilincinde, Atatürk’ün ilkelerini kendine şiar edinmiş, ülkesini seven ve ülkesi için mücadele eden gençlerin yetişmesine katkı sunuyor.Erzurum’un Pasinler İlçesi...

İki aile arasındaki taşlı sopalı kavga ihbarı üzerine, Jandarma Kıdemli Başçavuş Mustafa Yaşar ve emrindeki askerler olay yerine gidiyorlar.

Beraberlerinde bir ambulans da götürüyorlar.

Jandarma erleri yaralıları ambulansa taşırlarken E.K. isimli kişi Mustafa Yaşar’a bir taş savuruyor.


Taşın kafasına isabet etmesi nedeniyle sol elmacık kemiğinde 20 kırık oluşuyor, sol gözü parçalanıyor, burnu 3 yerden kırılıyor, kafatası çatlıyor ve beyni su topluyor.

Yani ölümün eşiğinden dönüyor.

Hemen hastaneye yetiştirilen Kıdemli Başçavuş aradan geçen 7 ay içinde peş peşe 5 ayrı ameliyat oluyor.

Ve 25 Ocak 2024 günü yapılan duruşmada saldırgan E.K. (Adam şüpheli değil sanık. Aylardır yargılanıyor ama açık yazılması gereken ismi, haberlerde hâlâ  E.K. olarak geçiyor!) adli kontrol koşuluyla salıveriliyor!..""",
        "Uğur Dündar",
    ),
    (
        """Aşağıda okuyacağınız yazı, 1993 yılının karlı 24 Ocak sabahı, Ankara’da, evinin önündeki özel aracına yerleştirilen patlayıcılarla şehit edilen büyük araştırmacı gazeteci dostum Uğur Mumcu tarafından, 16 Nisan 1981’de Cumhuriyet gazetesindeki köşesinde yayımlandı.

Aradan gecen 43 yılda onun “ülkemizde en yaygın, en uzun süre ayakta kalmış, en tehlikeli ve en yoğun siyasal destekli gizli örgüt” olarak tanımladığı “mafya” yok edilemediği gibi, giderek uluslararası bir boyut kazandı. Türkiye yabancı mafya babalarının, uyuşturucu ve fuhuş baronlarının hesaplaşma arenasına dönüştü. 

İstanbul ve Ankara başta olmak üzere büyük kentlerimizle tatil cenneti yörelerimizde, polisiye filmlere konu olabilecek, kara paraların paylaşımı kaynaklı korkunç cinayetler işlendi.


İçişleri Bakanlığı değişikliğinden sonra aralıksız yapılan operasyonlardan anlıyoruz ki dünyada kırmızı bültenle aranan ne kadar suç örgütü liderleri varsa kapağı Türkiye’ye atmış.

Ülkemiz dünyanın azılı kriminalleri için adeta genel kurul toplantısı yapabilecekleri korunaklı bir alan olmuş!..Şehit edilişinin 31. yıldönümünde sevgi, saygı, özlem ve rahmetle andığımız Mumcu’nun 43 yıl önce bugünleri anlattığı ve mafyanın siyasi bağlantılarına dikkat çektiği o yazısı:

“Önceki gece televizyonda Uğur Dündar’ın “Günlerin Getirdiği” adlı programını herhalde izlemişsinizdir. Dündar, bu programda gerek Gaziantep’teki kaçakçılığı, gerekse İskenderun’da “Soğukoluk” adı verilen fuhuş ve eğlence örgütünü gözler önüne serdi. Bu program nedeniyle TRT yönetimi ile Uğur Dündar’ı yürekten kutlarız.İzlediğimiz program genel olarak “mafya” olarak andığımız yeraltı dünyasının, yoksul genç kızları ne gibi yollarla ağlarına düşürüp, bu kara yazgılı insanları nasıl kullandıklarını unutulmaz görüntülerle ortaya koydu.Kabul edelim ki Türkiye’de yıllarca, yasa ve devlet tanımayan yeraltı dünyasının kuralları egemen oldu. Uyuşturucu madde, silah, sigara ve yedek parça kaçakçılığından başlayıp, eğlence ve fuhuş ticaretine kadar uzanan suç örgütleri, siyaset dünyası ile içlidışlı ilişkilerle ayakta kalmayı becerdi. Silah kaçakçılarından gazino patronlarına kadar uzanan bir kirli çizgi, Türk mafyasının parmak izlerini yansıtmaktadır.Uğur Dündar’ın programında izledik. Gazino patronlarının “acenta” dedikleri bir “fuhuş örgütü” çeşitli yollarla genç kızları aldatıp kaçırıyor. Ve bu genç kızlar bu örgüt tarafından uyuşturucu haplarla bayıltılıp, Soğukoluk’taki  fuhuş ve eğlence merkezine getiriliyor. Artık buradan kaçıp kurtulmak olanaksızdır. Gizli dehlizler, mağaralar yapılmış, muhafızlar ve eşkıya kiralanmıştır.""",
        "Uğur Dündar",
    ),
    (
        """Cumhuriyet tarihimizin en vahim iç politika yanlışlarından biri iktidarın, Bölücü Terör Örgütü PKK ile silah bıraktırmadan, üstelik Meclis’ten kaçırarak “Çözüm Süreci”ni başlatması, dış politika da ise ABD’nin peşinden rejimi devirmek için Suriye’ye girmesi ve sonrasında yanlışta ısrar etmesidir.

Bakın tüm öngörüleri doğru çıkan emekli Büyükelçi Şükrü Elekdağ, bundan 11 yıl önce bu tespiti nasıl dile getiriyor:

“Cumhuriyet döneminde dış politikada en büyük hatalar deyince akla ilk önce 1961’de Jüpiter füzelerini Türkiye’de konuşlandırma kararı ve 1979’da Rogers Planı’nın kabulü gelir. Ancak bunların doğurduğu zarar, AKP’nin Suriye politikasının Türkiye’yi karşı karşıya bıraktığı risk ve tehditlerle karşılaştırıldığında devede kulak kalır!..”Geride bıraktığımız 11 yılda yaşadıklarımızı şöyle bir düşünün.

Terör eylemlerinde verdiğimiz can kayıplarını... Suriye sınırına PKK başta olmak üzere IŞİD gibi terör örgütlerinin yerleşmeleriyle ortaya çıkan beka tehdidine karşı yapmak zorunda kaldığımız askeri operasyonları... Sayısını unuttuğumuz şehitlerimizi... Vatanlarından kaçarak ülkemize yerleşen milyonlarca Suriyeliyi... Sınırlarımızın kevgire dönüp, sığınmacı istilasına açık hale gelmesini... Bunların neden olduğu vahim sosyal, ekonomik ve demografik sorunları... Ekonominin belini doğrultması için harcanması gerekirken, sığınmacılara akıtılan milyarlarca doları... Uğradığımız maddi ve manevi zararların yanı sıra, büyük kentlerde asayişi sağlamanın giderek zorlaşmasını, gözünüzün önüne getirin...Hep sorduk, sormaya devam edelim:

İktidar ABD’nin Suriye’ye girişini destekler ve Büyük Ortadoğu Projesi’nin (BOP) eş başkanlığını yürütürken, hedefin Suriye topraklarında PKK’ya bir garnizon devlet kurmak ve onu İsrail ile komşu yapmak olduğunu bilmiyor muydu?


Buna Rusya’nın da ses çıkarmayacağından, hatta İsrail’in güvenliği için bu devletçiğe örtülü destek vereceğinden haberdar değil miydi?""",
        "Uğur Dündar",
    ),
    (
        """Anadolu, yıllar süren savaş sonunda yıkıntıya dönmüştü. Son atımlık cephanesini de İstiklal Savaşı’nda harcamıştı. Askeri zafer kazanılmış, sıra enkaza dönen ülkeyi kalkındırma savaşına gelmişti.

O süreçte 1929 Dünya Ekonomik Buhranı patlak verince, işler daha da zorlaşmıştı.

Gazi Mustafa Kemal Atatürk, halkın sıkıntılarını yerinde görüp dinlemek için sık sık yurt gezisine çıkıyordu.6 Mart 1930, Antalya...

Gün boyu halka dertleşen Atatürk kaldığı odada koltuğa yığılıyor. Çok yorgundur. Elleri titreyerek sigarasını yakar ve yanındaki Genel Sekreter Hasan Rıza Soyak’a, şu sözlerle içini döker.

“Bunalıyorum çocuk, büyük bir acı içinde bunalıyorum. Gittiğimiz her yerde devamlı dert, şikayet dinliyoruz. Her taraf derin bir yokluk, maddi manevi perişanlık içinde. Ferahlatıcı pek az şeye rastlıyoruz, memleketin gerçek durumu bu işte. Bunda bizim bir günahımız yoktur.Uzun yıllar, hatta asırlarca dünyanın gidişinden aymaz, bir takım bilinçsiz yöneticilerin elinde kalan bu cennet memleket; düşe düşe şu acınacak duruma düşmüş. Memurlarımız henüz istenilen düzeyde ve kalitede değil. Çoğu görgüsüz, yetersiz ve şaşkın. Büyük yeteneklere sahip zavallı halkımız ise kendisine kutsal inanç şeklinde telkin edilen bir sürü temelsiz görüşlerin etkisi altında uyumuş, kalmış...Bu arada beni en çok üzen şey nedir bilir misin? Halkımızın aklında kökleşmiş olan, her şeyi başta bulunandan beklemek alışkanlığıdır. Bütün iyilikleri bir kişiden, yani şimdi benden istiyor, benden bekliyor. Ama sonuçta ben de bir insanım birader, sihirli bir gücüm yok ki...Yeri geldikçe her yerde tekrar ediyorum. Bütün bu dertlerin, bütün bu ihtiyaçların giderilmesi, her şeyden önce bilgili, geniş düşünceli, azimli, gönlü tok ve uzmanlık sahibi (liyakatli) adam meselesidir. Sonra da zaman ve imkan meselesidir.Bu itibarla önce kafaları ve vicdanları yıpranmış, geri, uyuşturucu düşüncelerden temizleyeceksin. İşlerin uzmanı, idealist ve enerjik insanlardan kurulu, düzenli, her parçası yerli yerinde, modern bir devlet makinesi kuracaksın. Sonra bu makine, halkın başında ve halkla beraber durmadan çalışacak, maddi, manevi her türlü doğal yetenek ve kaynaklarımızı harekete geçirecek, işletecek, böylece memleket ileriye, refaha doğru yol alacak. İleri milletler düzeyine erişmek işini bir yılda, beş yılda hatta bir nesille tamamlamak da imkansızdır.

Biz şimdi o yol üzerindeyiz. Kafileyi hedefe doğru yürütmek için insan gücünün üstünde gayret sarf ediyoruz. Başka ne yapabiliriz ki?..”""",
        "Uğur Dündar",
    ),
    (
        """Önceki akşam Cumhuriyet Halk Partisi’nin (CHP) Meclis Grup Başkanı ve Genel Başkan Adayı Özgür Özel’i, SÖZCÜ Tv’de değerli meslektaşlarım İpek Özbey ve İsmail Saymaz ile soru yağmuruna tuttuk.

Özel’in verdiği bazı cevapları dinlerken, şaşkınlıktan ne diyeceğimizi bilemedik.Şu rezalete bakar mısınız?

Partiyi Meclis’te temsil eden en yetkili kişi, Genel Başkanın danışmanlarının önemli bir bölümünün varlığından, seçimlerden sonra medyada çıkan haberler sayesinde haberdar olmuş!..

Hele Atatürk’e ve Kılıçdaroğlu’na hakaretler eden bir kişinin Atatürk’ün kurduğu partide danışman olmasını, dahası kendisine özel bir oda ayrılmasını, içine hiç mi hiç sindirememiş.""",
        "Uğur Dündar",
    ),
    (
        """17 Ağustos 1999 daki Büyük Marmara Depremi korkunç pençeleriyle vura vura gelmiş, binaları, araçları, önüne çıkan her şeyi çiğneyip parçalayarak, enkaza çevirerek, bir kenara...17 Ağustos 1999 daki Büyük Marmara Depremi korkunç pençeleriyle vura vura gelmiş, binaları, araçları, önüne çıkan her şeyi çiğneyip parçalayarak, enkaza çevirerek, bir kenara atmıştı.

Gece 03.00 de başlayan ve 45 saniye süren deprem canavarının ilerlemesiyle Kocaeli, Sakarya, Yalova, İstanbul, Bolu, Eskişehir, Zonguldak, Bursa çok ağır hasar almıştı. Yirmi bine yakın insan ölmüş, yüz binlerce kişi de yaralanmıştı. O gece Türkiye, tarihin en ürkütücü yıkımlarından birini yaşamıştı…Depremde, o ana kadar çok güvenilen insani yardım kuruluşumuz Kızılay da enkaz altında kalmıştı.

Günlerce; “Kızılay nerede?” diye sordu insanlar. On binlerce kişi yaralıydı ve hemşire, doktor, ilaç, yiyecek, içecek, giyecek, çadır ihtiyacı vardı…

Fakat Kızılay, en olması gereken zamanda, orada yoktu!.. """,
        "Uğur Dündar",
    ),
    (
        """İktidarın aldığı bir kararla, bundan böyle ev sahibi-kiracı ihtilaflarında mahkemeden önce arabulucular devreye girecek.

1 Eylül tarihinden itibaren başlayan bu uygulamada, uzlaşı sağlanamaması halinde ihtilaf mahkemeye taşınacak.

Özeti bu olan yeni gelişmeyle ilgili ayrıntıları okurken, yıllar önce yaşadığım çarpıcı bir mülk sahibi- kiracı çatışmasını hatırladım.

Anlatayım:""",
        "Uğur Dündar",
    ),
    (
        """Genç bir Türk subayının 25 Nisan 1915 sabahı, Çanakkale-Conkbayırı’nda, 57’nci Piyade alayına: ‘Ben size taarruzu değil, ölmeyi emrediyorum.’ demesiyle başladı her şey.Sekiz sene sürecek...Genç bir Türk subayının 25 Nisan 1915 sabahı, Çanakkale-Conkbayırı’nda, 57’nci Piyade alayına: ‘Ben size taarruzu değil, ölmeyi emrediyorum.’ demesiyle başladı her şey.

Sekiz sene sürecek amansız mücadelenin fitili bu sihirli cümleyle ateşlendi. Zira o subay, dahili ve harici düşmanlar tarafından başına ve vatanına gelebilecek her türlü musibeti kusursuz ferasetiyle öngörmüş, bunlara karşı tedbirlerini daha otuz dört yaşında cephelerde almıştı. Amerika’nın mandası olmayı düşünenler de vardı, İstanbul’u pazarlayıp paçayı kurtarmak isteyenler de…


Ama O, yönünü vatanına döndü…Çanakkale’de taarruzu değil, tam bağımsızlık için ölmeyi emreden inancın tezahürüyüz hepimiz. Yarbay Mustafa Kemal’in gözlerinde parlayan o inancı saygıyla, minnetle selamlıyorum.""",
        "Uğur Dündar",
    ),
    (
        """Üzerimde emeği bulunan meslek büyüklerimle ilgili bir “vefa” kitabı hazırlıyorum.Bu amaçla anı defterlerimi, geçmişin gazete küpürlerini ve ses kayıtlarını karıştırırken, 1987...Üzerimde emeği bulunan meslek büyüklerimle ilgili bir “vefa” kitabı hazırlıyorum.

Bu amaçla anı defterlerimi, geçmişin gazete küpürlerini ve ses kayıtlarını karıştırırken, 1987 yılı Sedat Simavi Gazetecilik Ödülü’nü birlikte kazanma onurunu yaşadığım, ülkemizde soruşturmacı gazeteciliği başlatan basın şehidi Uğur Mumcu’nun benimle ilgili yazılarını buldum.

Bunlardan birine şöyle başlamış:Üzerimde emeği bulunan meslek büyüklerimle ilgili bir “vefa” kitabı hazırlıyorum.

Bu amaçla anı defterlerimi, geçmişin gazete küpürlerini ve ses kayıtlarını karıştırırken, 1987 yılı Sedat Simavi Gazetecilik Ödülü’nü birlikte kazanma onurunu yaşadığım, ülkemizde soruşturmacı gazeteciliği başlatan basın şehidi Uğur Mumcu’nun benimle ilgili yazılarını buldum.

Bunlardan birine şöyle başlamış:""",
        "Uğur Dündar",
    ),
    (
        """Adeta dörtnal koşar gibi, ne çabuk geçti yıllar…Geçen gün, birkaç ekip arkadaşımla ARENA’da 30, meslekte de 54 yılımın geride kalmasını kutladık.Dile kolay; yarım asrı aşkın bir...Adeta dörtnal koşar gibi, ne çabuk geçti yıllar…

Geçen gün, birkaç ekip arkadaşımla ARENA’da 30, meslekte de 54 yılımın geride kalmasını kutladık.

Dile kolay; yarım asrı aşkın bir süre…

Hakikate adanmış bir hayat…

Yaşadıklarımızı konuşurken, hayatımın dönüm noktaları, bir film şeridi gibi gözlerimin önünden akmaya başladı…""",
        "Uğur Dündar",
    ),
    (
        """Geçmişte (Cumhuriyet gazetesi öncesi) yazıp da yayımlamadığım bir yazı buldum zulada. Bu yazıyı köşeli paranteze aldıktan sonra tamamlayacağım. Bilginize...

[Bu düşmanca bir programıdır. Başbakanın “İmam hatipler milletin gözbebeği olacaktır” yobaz talimatına gönderme yapılan broşürde “Yeni dönemde Türkiye’nin gözbebeği olacak olan imam hatip liselerine kayıtta geç kalmayın” uyarısı yapılıyor.

Sonra sıra velilere geliyor: “Çocuklarımız imam hatip ortaokullarını bitirdiğinde; hem yüce kitabımız Kuranıkerim’i öğrenecek hem de Anadolu veya öğretmen lisesine gidebilecek hem Hz. Peygamber’in hayatını öğrenebilecek hem de fen lisesi veya imam hatip lisesine gidebilecek.”Çocuklar kesinlikle gerçek Kuranıkerim’i öğrenemeyecekler çünkü öğretmenleri de Kuranıkerim Arapçasını bilmezler. Çocuklar laik okullarda öğrendikleri İngilizce ve Fransızca kadar Arapça öğrenemeyecekleri için Kuran’ı anlamalarına olanak yoktur. Ancak onu ezberleyecekler ve saptırılmış Türkçe mealini okuyacaklar. Ama buna karşın tamamı softa ve yobaz olacak. Said Nursi ve Fethullah Gülen türü meczupların kölesi olacaklar; kafa ve ruh sağlıklarını yitirecekler. Dahası, ana-babalarından nefret edecekler, kadını ve erkeği anlama yeteneğinden yoksun olacakları için de sağlıklı bir yuva kuramayacaklar.

“İmam hatip okullarının ‘Sosyal, beşeri ve fen bilimleri ile birlikte İslami ilimleri aynı müfredat altında göstermesi bakımından Türkiye’ye özgü bir tecrübe’ olarak tanıtıldığı bilgi notunda ise ‘Gençlerimize bu okullarda değerler eğitimi verildiği için kötü alışkanlıklar yok denecek kadar azdır’ görüşü dikkat çekiyor.”

Bu cümle, laik okulların bile isteye AKP hükümeti tarafından sabote edildiğinin itirafıdır.

Broşürün sözünü ettiği “Türkiye’ye özgü tecrübe” Said Nursi’nin Van’da kurmayı hayal ettiği “Medresetüzzehra” ucubesini örnek almaktadır.

Sonuç olarak: AKP hükümeti pedagoji bilimine aykırı altı kaval üstü şeşhane bir öğretim sistemi içinde Türk gençliğini özgür ve bilimsel düşünceden uzak, kumanda aletiyle güdülebilir bir insan sürüsü haline getirmek istemektedir. ÖNDER’in broşüründe de itiraf edildiği gibi uygar dünyada böylesine kaçık ve budala bir okul ve öğretim sistemi bulunmamaktadır.

Düşünsenize, gerçek beden eğitimi, müzik, resim derslerinden yoksun bırakılan çocuklarımız “din dersi” adı altında haftada sekiz saat irtica, hurafe ve üfürükçülük dersi alacaklar. Ağustos ayında yapılacak atamalar sonunda 8 bin başlık din kültürü ve ahlak bilgisi öğretmeni açığı kapatılmaz ise müftülük, imamlık ve Kuran kursu öğreticiliği yapan (ama pedagoji sertifikası bulunmayan) ilahiyat fakültesi mezunları ders verecek. Ve bu yetkisiz ve yeteneksiz kimselerin mezun ettiği zavallı çocuklar dünya ile yarışacaklar (!).

Böyle bir uygulama karşısında bütün “aklı başında Türkiye”nin ayağa kalkması, isyan etmesi, uygulamayı yargıya götürmesi gerekmektedir. Ama tıpkı Osmanlı gibi Türkiye de uyuyor. Bütün dikkatini iki ve üç sayfalara vermiş olan basın çürümeye devam etmektedir.]

Yazının başında bir başbakandan söz edildiğine göre yazı R.T. Erdoğan’ın başbakanlık döneminde (2003-2018) yazılmış olmalı. Bu yazıyı bulmam çok iyi olmuş. Çünkü Google Alerts, Yeni Akit gazetesinde hakkımda yayımlanan (1.6.2024) bir yazıyı gönderdi. Yazı, 31 Mayıs günü gazetemizde yayımlanan “Einstein imam hatipte okusaydı” üzerine.

“Ölünce cesedinin yakılmasını vasiyet eden Cumhuriyet yazarından, İHL’lerle ilgili alçakça sözler! Tüh senin suratına.Ölünce cesedinin yakılmasını vasiyet eden Cumhuriyet gazetesi yazarı Özdemir İnce, yine boyundan büyük laflar etti. İmam hatip okullarına saldırıp, bilim insanlarının bu okullarda yetişemeyeceğini savunan İnce, İHL açılmasını ‘soytarılık’ olarak nitelendirme aymazlığında bulundu. İnce, rezil ötesi yazısında, özetle şu herzeleri yumurtladı:

'AKP aklı, bütün başarılı öğrencileri imam hatip okulları adlı medreselerinde toplamak için türlü desiselere başvuruyor. Böylece dindar ve kindar bilim insanı, felsefeci, yazar ve sanatçıları yaratarak tek eksilerini (!) kapatacaklar!'”

Gerçekten de bırakın entelektüel ürünleri, İHL’den mezun iyi bir futbolcu, basketbolcu, voleybolcu ya da atlet var mı? Mimarın, bestecinin, ressamın sözünü bile etmiyorum! """,
        "Özdemir İnce",
    ),
    (
        """31 Mart 2024 günü yapılan yerel seçimlerin CHP’nin başarısıyla sonuçlanması üzerine kimilerinin çocukların sokakta “Dondurma, dondurma!” diye şımarıklık yapmasından ilham almışçasına “Erken seçim, erken seçim!” diye tutturmasını merak ve hayretle izlemekteyim. Bu erken seçim tiryakileri “Erken seçim isterük!” deyu CHP’ye ve partinin yeni genel başkanı Özgür Özel’e baskı yapmaktalar. Özgür Özel de “Halk isterse!” diye mazeret beyan etmekte. Ben fakir, sizin de bilip takdir edeceğiniz gibi bu işlerden ezbere anlamam. Anlamadığım işlerde anayasa ve yasalara bakarım. Bu inatçı ısrar üzerine anayasayı açıp baktım, erken seçim nasıl olur diye. Şöyle bir madde var anayasada:Madde 116-(Değişik: 21/1/2017- 6771/11 md.)

Türkiye Büyük Millet Meclisi, üye tamsayısının beşte üç çoğunluğuyla seçimlerin yenilenmesine karar verebilir. Bu halde Türkiye Büyük Millet Meclisi genel seçimi ile Cumhurbaşkanlığı seçimi birlikte yapılır. """,
        "Özdemir İnce",
    ),
    (
        """AKP aklı, bütün başarılı öğrencileri imam hatip okulları adlı medreselerinde toplamak için türlü desiselere başvuruyor. Böylece dindar ve kindar bilim insanı, felsefeci, yazar ve sanatçıları yaratarak tek eksilerini (!) kapatacaklar! Değil günümüzün en başarılı çocuklarını, Thales, Pisagor, Arşimet, Newton, Kopernik, Galileo, Darwin, Einstein, Stephen Hawking; Biruni, Cezeri, Harezmi, İbni Haldun, İbni Sina, Ali Kuşçu, Farabi, Cahit Arf, Hulusi Behçet, Aziz Sancar vb. gibi geçmişin ve günümüzün dâhilerini bile AKP’nin imam hatiplerine doldursanız sonunda ancak dinbaz, kötü siyasetçi ve haramzade müteahhit yetiştirebilirsiniz. Çünkü günümüzün sünnetçi Gazalileri tarafından akılları ve imgelem güçleri sünnet edilir. Akıl ile inanç, çiftleşmezler.


Bilimin kaynağında kör inanç ve vahiy yoktur, akıl vardır. Bilim ve sanatın Tanrı ve dinle doğrudan, dolaylı-dolaysız bir ilişki ve sorunu yoktur ama din adamıyla (ruhbanla) ve dinbazla sorunu vardır. Tanrı ve din “insan” değildir ama rahip, haham, imam ve dinbaz “insan”dır. Hadi meslekleridir diye rahip, haham ve imamı idare edelim ama dinbaz  neyin nesi, kimin fesi oluyor? Dinbaz, dinin yozlaşmış, kokuşmuş materyalist ürünüdür. İnsanları uyutmak için imam hatip okulları açar, çiftlik ve inek banklar kurar. Tanrı ve din, bilimsel akıldan gocunmaz ama dinbaz, engizisyon mahkemeleri kurup bilim insanlarını ateşlikte yakar, zulmeder; (İskenderiyeli Hypatia, Sokrates, Pisagor, Galileo, Kopernik, Newton, Darwin, Lavoisier, Bruno, Bacon vb.) Hallac-ı Mansur gibi derisini yüzer, türlü çeşitli zulümler eder. (İbni Sina, Suhreverdi, El Kindi, Farabi vb.) """,
        "Özdemir İnce",
    ),
    (
        """1 Mayıs günü Saraçhane’den simgesel Taksim Meydanı’na yürüyen emekçileri televizyonda izlerken 1966 yılı 1 Mayıs’ında Paris’te Saint-Michel Bulvarı’nda yoldaşlarla birlikte “Muguets, demandez messieux-dames, muguets” diye diye müge sattığımızı anımsadım. Beyaz müge çiçeği baharın ve devrimin simgesidir. Pek genç de değildim 29 yaşımdaydım ve Türkiye’de yapmam olanasız bir gösteriye katılmak istemiştim.

Paris Komünü’nden devşirerek Taksim’i artık “devrim”in simgesi olarak ilan ediyorum. Paris Komüncüleri (Communards) 1871 yılının 18 Mart-28 Mayıs tarihleri arasında Paris Komünü adlı bir sosyalist oluşumla iktidara geldi. Bizim ilk Meclis hükümetimizin (23-24 Nisan 1920) ilham kaynağıdır. Komünün yıkılmasının ardından komün üyelerinin yaklaşık 20 bini ilk hafta içinde infaz edilmiş, yaklaşık 7 bin 500 kişi hapse atılmış veya denizaşırı sömürgelere sürgüne gönderilmiştir. 1880 yılında genel af ilan edilse de önde gelen komün üyeleri af kapsamına alınmamıştır. Paris Komünü, devrimci emek hareketleri için son derece önemlidir. Paris Komüncülerinin Kiraz Zamanı (Le Temps des cerices) adlı şarkısı vardır. Şiiri 1866’da Jean Baptiste Clément yazmıştı. Bu şarkıyı en iyi Yves Montand söyler. Taksim Komüncülerini televizyonda izlerken onu dinledim. (Google’da bulabilirsiniz.) Bir gün gelecek hükümetin Taksim Komüncülerini engelleyen polisleri de devrimcilere katılacaktır.

Benim de Kiraz Zamanı (1968 May Şiir Ödülü) adlı bir kitabım vardır. Aşağıda okuyacağınız şiir benim Kiraz Zamanı’mdır.KAVUN ACISI
Bu kavun acısı gelecektir

bu kavun acısı geçecektir

demir tavını bulacaktır

ağır kuru ve gebe bir sesle

çekiç örse vuracaktır

karımın devsel yeşil gözleri

öfkenin şiirini yazacaktır



Kavun acısı

kışın ilk sesidir camlarda

yazın boş bir okul avlusunda

birikmesidir,

unutulmuş bir kalemdir öğretmen

masasında

gülen ayvadır ağlayan nardır

bir umut sürgünüdür Dicle boyunda

kavun acısı gelecektir

kavun acısı geçecektir

kırağı gibi dalların üzerinden

bir al turna gibi tüfeğin önünden

su gibi damlayacaktır

ve dağlayacaktır yalım gibi

kavun acısı geçecektir

kiraz zamanı gelecektir



Çünkü

saat çalışır ve tamamlar günü

bir kan damlar kaldırımın üzerine

bir daha bir daha damlar

acı yağmur suyuna karışır

bir adam durur direğin dibinde

boynu kıldan ince bir adam

saat vurur yürek atar kan damlar

atar sigarasını adam ezer böcek gibi

atar sigarasını adam ezer yazgı gibi

atar sigarasını adam, çünkü

bir yerlerde beyaz mügeler açmaktadır

incir sütü biber gibi yakmaktadır

ak döşekler diken gibi batmaktadır

dağlar dağlar dağlar çağırmaktadır



Türkünün yurdu insanın yüreğidirtürkünün yüreği insanın belleğidir

onlar senin türkünü anlamazlar

türkün bütün sularda yıkanmıştır

bütün otların ince tadını bilir

bütün zindanları özgürlüğe çevirmiştir

onlar senin türkünü anlamazlar

çünkü onlar

gak deyince et

guk deyince su isteyen

Anka’dırlar

Kavun acısı geçecektir

kiraz zamanı gelecektir

bu kütük çiçeğe duracaktır

karım devsel yeşil gözleri

öfkenin şiirini yazacaktır.

""",
        "Özdemir İnce",
    ),
    (
        """Türkiye’nin en iyi gazetesi benim yazdığım Cumhuriyet gazetesidir. Lise öğrenciliğimde Cumhuriyet ve Dünya gazetesini satın alıp okurdum, spor haberleri için Hürriyet gazetesine bakardım. Hürriyet gazetesini ilk kez 7 Ağustos 1948 günü satın almışım, 12 yaşımda. Birinci sayfasında Londra Olimpiyat Oyunları’nda grekoromen stilinde olimpiyat şampiyonu olan Mersinli Ahmet ile Mehmet Oktav’ın fotoğrafları var. Takım halinde ikinci olmuşuz.

Cumhuriyet gazetesi gözdem ve öğretmenimdir ama başka gazetelerin iyi işlerini görmeme engel değildir bu. Örneğin Sözcü gazetesinin birinci sayfasını çok beğenirim. 17 Mayıs 2024 günlü birinci sayfası bana bu yazıyı esinledi.

Manşet: “Diyanet her gün yiyor et!” Kafiyeli! Din adamlarının yemek listesi bu yazıyı ilham etti. Güney Amerika’nın Katolik rahiplerinin dışında dünyanın geri kalan ülkelerindeki din adamları, kapitalizm ve sömürüden yanadır. Birinci sınıf beş yıldızlı otellerde toplantı yapıp buralarda yatan ve çatal kaşık şakırdatan Diyanet İşleri Başkanlığı (DİB) imamlarının pahalılıktan ve müzmin açlık ile yoksulluktan şikâyetçi olup durumu eleştirdiklerini hiç duydunuz mu? Adamlar sanki bir sömürge ülkesinde kolonyal düzenin temsilcileri...DİB ya da kısa adıyla Diyanet, 3 Mart 1924 tarihinde Şeriyye ve Evkaf Vekâleti’nin yerine kurulan, İslam dininin inançları, ibadet ve ahlak esasları ile ilgili işleri yürütmek, din konusunda toplumu aydınlatmak ve ibadet yerlerini yönetmekle görevli kurumdur. Mustafa Kemal Atatürk’ün emriyle 429 sayılı kanunla Türkiye Cumhuriyeti Başbakanlığı’na bağlı bir teşkilat olarak kurulmuştur. 9 Temmuz 2018’de Türkiye Cumhuriyeti Cumhurbaşkanlığı’na bağlanmıştır.

Anayasanın 136. maddesinde, ‘Genel idare içinde yer alan Diyanet İşleri Başkanlığı, laiklik ilkesi doğrultusunda, bütün siyasi görüş ve düşünüşlerin dışında kalarak ve milletçe dayanışmayı ve bütünleşmeyi amaç edinerek özel kanununda gösterilen görevleri yerine getirir’ hükmü yer almaktadır.Mevcut DİB Kuruluş ve Görev Yönetmeliği’nde yukarıdaki ilkeler yer almaktadır. Sözcü gazetesi mayıs ayının yemek listesini yayımlamış. Afiyet şeker, lop lop et olsun. Dünkü (23 Mayıs) menüsü şöyle: Mantar çorbası, fırın köfte, nohut salatası, keşkül. Listede ana yemek olarak ekşili köfte, orman kebabı, Ankara tava, etli nohut, tavuk döner, çiftlik kebabı, soslu misket köfte, etli taze fasulye, tavuk külbastı, hünkâr beğendi, sebze graten, çiftlik köfte, tavuk şinitzel, etli taze fasulye, et döner, fırın köfte, patlıcan musakka, etli kuru fasulye, püreli kebap, piliç Topkapı, kıymalı ıspanak, karışık ızgara. Ayrıca çorba çeşitleri, ara sıcaklar ve dört çeşit son tıkıntı var. Bu listeyi okusun, halkın bir yeri şişer vallahi!

Diyanet son dört ayda 31.8 milyar lira harcamış! 17 Mayıs 2024 tarihli Sözcü gazetesinde Deniz Ayhan’ın haberi şöyle:

‘Yiyin efendiler yiyin’

“Milyarlarca liralık bütçesi ve lüks makam araçlarıyla gündemden düşmeyen DİB’nin üst düzey yöneticileri için hazırlanan öğle yemeği listesi ortaya çıktı. Vatandaşın ucuz et için zifiri karanlıkta kuyruklara girip saatlerce kuyruk beklediği, çocukların ete ulaşamadığı bir ülkede DİB’nin üst düzey yöneticileri için hemen her gün etli yemek çıkıyor. Sebze yemekleri de kıymalı yapılıyor. Diyanet’in 17 Mayıs, yani bugünkü sofrasında ‘mısır çorbası, çiftlik köfte, bulgur pilavı, meyve’ olacak.”

Yemek maliyetlerinin bir kısmı bütçeden karşılanırken DİB Başkanı Ali Erbaş, bir günlük yemek için 67.5 TL ödüyor. DİB başkan yardımcıları, Din İşleri Yüksek Kurulu başkanı, Diyanet Akademisi başkanı, genel müdürler, rehberlik ve teftiş başkanı, strateji geliştirme başkanı da bu mönü için 67.5 TL ödüyor. Diyanet’te çalışan 4/D’li işçiler radyo TV personeli ve KOMAŞ personeli ile başkanlık personeli 100 TL ödüyor. Misafirler için ise yemek 110 TL. Aşçı, bekçi, hizmetli, şoför, yönetmen, 4/B sözleşmeli personel bu yemek için 30 TL, başkanlık müşaviri, baş müfettiş, hukuk müşaviri, avukatlar da 42 TL ödüyor.

“Sosyal adalet”e göre az kazananın az, çok kazananın da az kazanandan fazla ödemesi gerekmez mi? Gerekir ama DİB’de demek ki dinsel adalet uygulanıyor, sosyal adaletin yeri yok.

""",
        "Özdemir İnce",
    ),
    (
        """Adonis, “Kitap, Hitap, Hakikat”1 adlı kitabının 143. sayfasında şöyle yazar: “Araplar olarak çoğumuz vatan ile dini birleştiririz. Bu nedenle milliyetçilik ile dindarlığı da birbirine karıştırırız. Ama vatan ‘tek’ dindarlık ‘çok’tur.”Bu cümleyi okuyunca aynı sayfanın boşluğuna alışkanlığım gereği, ben de şunları yazmışım: “Din vatan değildir. Vatanın olmadan dinin olabilir. Vatan sana din verebilir ama din sana vatan veremez!”

İslamın ilkesi olan “ümmet” düşüncesi de vatanı yok sayar. Vatan yoksa, vatanın yoksa “hiç kimse”sin artık. Belki dinin vardır ama din vatan değildir. AKP’nin Cumhuriyetin fabrikalarını, limanlarını, madenlerini kolayca satmasının nedenini arıyorsanız o neden buradadır. Türkiye’yi vatan saymıyor. “Biz neyi satacağımızı çok iyi biliriz!” diyor. Peki neden satıyorsun, satmasını çok iyi bildiğin şey vatanın önemli bir parçası değil mi?


Bu yazıyı defterlerimden birine de aktarmışım. Sayfanın bir yerinde “Sanayinin gelişmesinde din adamlarının bir dirhem katkısı yoktur. İnsanlık bilim ve fen ile insan olup dünyayı değiştirdi” diye yazmışım.Bir ulusun bağımsız ve egemen olarak üzerinde yaşadığı yeryüzü parçasına ve onun havası ile karasularına vatan denir. Bir kimsenin doğup büyüdüğü; bir milletin hâkim olarak üzerinde yaşadığı, barındığı, gerekirse uğrunda canını vereceği toprak. Bir kimsenin yerleştiği yere de vatan denir. Vatan ile yurt aynı anlamdadır. """,
        "Özdemir İnce",
    ),
    (
        """Televizyonda izlemiştim. Şimdi bir kez daha kontrol ettim. Doğru anımsıyorum. Yıl 1994. Yerel yönetim seçimleri yapılmış, İstanbul Büyükşehir Belediye başkanlığını, solun aymazlığı yüzünden, Refah Partisi’nin yüzde 25.19 oy alan adayı Recep Tayyip Erdoğan kazanmış.

Refah Partisi’nin genel başkanı Necmettin Erbakan daha seçim sonuçları resmen açıklanmadan R.T. Erdoğan’ın elinden tutarak İstanbul Büyükşehir Belediye başkanının kapısına dayanmış. Ama ellerinde seçim kurulunun mazbatası yok!

Televizyon yayınından o sahne: Necmettin Erbakan ile R.T. Erdoğan bu beklenmedik devlet kuşunun etkisiyle gülümsemekte... Hâlâ makam sahibi Prof. Dr. Nurettin Sözen güler yüzlü, rahat! “Madem öyle, Sayın Erdoğan Seçim Kurulu’na gider mazbatayı alır gelir, biz de devir teslimi yaparız!” diyor. Bu ne uygar zarafet!Cumhuriyet devletini kuran CHP ile bu anıt yapıyı yıkmaya kalkışan AKP’nin 31 Mart 2024 yerel seçimlerinden sonra yaptıklarını karşılaştırınca aradaki uygarlık farkı açıkça görülmekte. Sanki bir ülkeyi düşman istila etmekte ve yenilgiye uğrayanlar havaalanlarını, limanları, köprüleri havaya uçurarak imha etmekte, değerli belgeleri yakmakta... Bunları yapan kimler? 1994 yılında SHP’li Nurettin Sözen’in kapıdan kovmayıp saygıyla makamına aldığı “Milli Görüş” simsarları...

Seçim kaybeden AKP belediyelerinin, belediye başkanlarının yaptıklarına bakın. AKP’nin adayı Murat Kurum, seçim çalışmaları kapsamında Kadıköy Rıhtımı’nda kurduğu iftar çadırını seçimden sonra kaldırmış. Oysa seçim sonuçlarına bakmayıp devam etselerdi gülünç olmazlardı. Nasıl olsa hesabı yeni belediye ödeyecekti. AKP nedense bu türden incelikleri düşünemiyor...

Benim her derde deva olan, geleceğin “gerçekleşecek” falına bakan kitaplarım vardır. 2016 yılının haziran ayında Tekin Yayınevi tarafından yayımlanan Din İman Masa Kasa da bunlardan biridir. Aslına bakarsanız, 19 Temmuz 1983 günü kurulan Refah Partisi ile 14 Ağustos 2001 tarihinde kurulan AKP İslamcılığının özet tarihi, din ve imanı kullanarak masa (iktidar) ve kasayı (parayı) ele geçirme kavgasının öyküsüdür. Din İman Masa Kasa’nın bir bölümünün adı “Ver Türbanı Yağmala Kasayı”dır ki bu öykünün anafikrini oluşturur.

İslamcı siyasetin “yağmacı” zihniyet dünyasına giriş yaptıktan sonra adını andığım kitabımın önsözünden alıntı yapmamak olmaz:

“Din ve imanın masa ve kasa ile bir araya gelmemesi gerekir ama din adamları ile siyasetçiler işe karışınca, bir araya gelmenin ötesinde din ve iman, masa ve kasanın hizmetine giriyor. Şimdi sözünü ettiğim yazıyı okuyalım:

7 Ağustos günü yayımlanan ‘Sünni Din Bezirgânları Artık Özgür’ başlıklı yazımı okuyan bir okur ‘Sünni madrabazların CHP’nin tek parti döneminde uğradığı zulmün (!) ne menem bir zulüm olduğuna açıklık getirdi:

Bir toplantıda din madrabazlarından biri CHP’nin tek parti döneminde uğradıkları zulmü konuşmacıya laf atarak hatırlatmış. Bunun üzerine konuşmacı laf atana sormuş:

‘Hangi ibadeti yapmak istedin de yapamadın? Namaz mı kılamıyordun, hacca mı gidemiyordun?...’

Madrabaz, konuşmacıyı yanıtlamış: ‘İbadeti yasaklamaya gücünüz yetmez. Siz bizi masadan ve kasadan uzak tutuyorsunuz.’

Müthiş bir yanıt. Hiç duymamıştım. Okur devam ediyor:

‘Yani tüm dertleri masaya ve -özelikle de- kasaya yanaşmakmış. Bunu yapamadıkları için gerçekten de ‘zulüm’ gördüklerine, acı çektiklerine inanıyorum. Düşünsenize, kasa orada, başkaları yanaşmış (örneğin ANAP, DYP) ama bunlar yanaşamıyor. Bu zulüm değil mi, onlar açısından?’”Monarşik ve teokratik Osmanlı’nın iktisat anlayışı, İslam öncesi ve sonrasının baskın, yağma ve ganimet üleşmek yönteminin devamı olarak değerlendirilebilir. Mekke döneminin kutsal savaşları kervan basıp ganimet paylaşmaktan ibarettir ki üleşmenin yöntemi kutsal metinlere bile girmiştir.

Arap-İslam töresine göre Müslümanlar tarafından yönetilen İslam ülkesi darülislam olarak adlandırılır. Kâfirler tarafından yönetilen ve halkı gavur olan memleketlere darülharp denir ki malı mülkü yağmalanabilir, insanları köle yapılabilir. Kadınları ise ister köle diye sat, ister hareme al. Hepsi helaldir!

Günümüzün siyasal İslamına göre üçüncü bir durum vardır: Kendisinin iktidarda olmadığı her yer darülharptir. İktidara geçince artık her şey helaldir. 1994’ten bu yana iktidar olan AKP belediyelerinin milyonlarca liralık leblebi, çekirdek faturalarına şaşmak gerekmez!

""",
        "Özdemir İnce",
    ),
    (
        """Okumanızı önerdiğim yazı 7 Mart 2002 günü Hürriyet Pazar’da “Tevrat, 137. Mezmur” adıyla yayımlanmıştı. İsrail’in Gazze’de yürüttüğü savaş görünürde Hamas’a (İslami Direniş Hareketi) karşı; ama öldürülen insanlar “Hamas” adlı insanlar değil Filistinliler; yıkılan, bombalanan yerleşim yerleri ve binalar Hamas denen terör örgütüne ait değil Filistinli zavallı insanların; dolayısıyla bu saldırı Hamas’tan çok Filistintilere karşı. 2002 yılının mart ayında gene İsrail saldırıya geçmiş, o zaman adını verdiğim yazıyı yayımlamıştım. Daha iyisini yazamayacağımı düşündüğüm için aynı yazıyı okumanıza sunuyorum.Çevredeki kavaklara/ Lirlerimizi astık./ Çünkü orada bizi tutsak edenler bizden ezgiler,/ Bize zulmedenler bizden şenlik istiyor,/ “Siyon ezgilerinden birini okuyun bize!” diyorlardı.

Nasıl okuyabiliriz RAB’ın ezgisini/ El toprağında?/ Ey Yeruşalim, seni unutursam,/ Sağ elim kurusun./ Seni anmaz, Yeruşalim’i en büyük sevincimden üstün tutmazsam,/ Dilim damağıma yapışsın!

Yeruşalim’in düştüğü gün,/ “Yıkın onu, yıkın temellerine kadar!”/ Diyen Edomluların tavrını anımsa, ya RAB.

Ey sen, yıkılası Babil kızı,/ Bize yaptıklarını/ Sana ödetecek olana ne mutlu!/ Ne mutlu senin yavrularını tutup/ Kayalarda parçalayacak insana!]

Tevrat’ın Mezmurlar (Zebur) bölümü bir ilahi ve dua kitabıdır. Bu ilahiler birkaç sınıfa ayrılabilir: Övgü ve tapınma ilahileri, ağıtlar; yardım, korunma ve kurtuluş için edilen dualar; bağışlanmak için yalvarışlar; Tanrı’nın kutsamalarına karşı şükran ilahileri; düşmanın cezalandırılması için dilekler. Bu dualar kişi ve ulus adına edilirdi.


137. Mezmur neden yazıldı, neyi anlatıyor? """,
        "Özdemir İnce",
    ),
    (
        """Kemal Kılıçdaroğlu, Meral Akşener ve Ekrem İmamoğlu...

Üçünü de kişisel olarak tanımam. Ancak Kemal Kılıçdaroğlu, Hürriyet gazetesinden atıldığımda “geçmiş olsun” telefonu etmişti. Meral Akşener de Hürriyet’te yazdığım dönemde bir kez telefon etti, annesinin ısrarı üzerine beni aramıştı... Ekrem İmamoğlu ile aramızda herhangi bir ilişki olmadı...

Cumhurbaşkanlığı adaylığı konusunda, 3 Şubat 2023 tarihli ve Ben de İmzalıyorum başlıklı yazıda şunları yazdım: “Kimin cumhurbaşkanı olacağı mutabakat metninde yer almıyor ama ben bu yazıda kimin aday olması gerektiğini de yazacağım: Millet İttifakı’nın fikir babası ve hamalı kim ise o aday olur: Fikir babası ve ‘mayın katırı’ Kemal Kılıçdaroğlu değil de öteki 5 genel başkandan biri olsaydı doğal aday o olurdu.”Meral Akşener: MHP’den istifa etti, nedeni henüz beni ilgilendirmez ama MHP’den mitoz bölünmeyle ayrıldığı için MHP’den farklı olması mümkün değil. Genler ortak. Bunun belirtileri altılı masa döneminde sık sık görüldü. MHP’den ters zamanda ayrılmıştı. Kurduğu partiyi örgütleyecek zamanı yoktu. Kadın içgüdüsüyle, komşudan tuz biber ister gibi, “Komşu komşu hu hu!” diye seslenerek CHP Genel Başkanı Kemal Kılıçdaroğlu’ndan 15 milletvekili ödünç istedi ve aldı. Bu sayede 2018 seçimlerine girme hakkı kazandı. Sıfır sermaye ile CHP ile ortaklık kurdu, CHP’nin kendi alın teriyle kazandığı belediyelerin ortağı oldu. Ancak ve sanki, “İYİ Parti olmasaydı CHP’nin hali dumandı (!)” durumu söz konusu... Bu kostaklanma da MHP’den miras kalan gen...

Bu ruh halini ve bu zihinsel yapıyı siyaset bilimsel yorumlarla betimlemek, çözümlemek mümkün değil, Kafka-Dostoyevski karışımı bir romanın işi bu. Biz henüz yazılmamış olan romanın sayfalarını çevirelim ve İYİ Parti’nin MHP tarzı şovenliği bırakmadan altılı masanın kurulmasına gelelim. Masa kuruldu ve İYİ Parti, HDP’nin belediye seçimlerindeki katkısıyla kazanılan zaferleri kendi defterine yazdı ve Asena’nın partisi olarak bu partiyi derhal aforoz etti. Bu otomatik tepkiyi anlamak mümkün... Soydur çeker! Ama payandası olan CHP’nin iç işlerine karışması kabalığına ne demeli? Görgü ve geleneklere göre, kendisi istemedikçe, bir partinin genel başkanınından başka kim “Herhangi bir şey” olabilir? Kimse! Üstelik son on yıl içinde bunun iki somut örneği varken? Bu ne görgüsüzlük, bu ne kabalık! Önce Ankara Büyükşehir Belediyesi başkanı, ardından ve ısrarla İstanbul’unki... Kemal Kılıçdaroğlu’nun nesi var, ne kusuru var? Parti kurarken yardımını istemek için adamın ayağına gideceksin, adam engin bir siyasal sezgi ve öngörüyle sana istediğin yardımı sunacak, seçime girip parti olacaksın, CHP sayesinde bitin kanlanacak amma velakin iş “cumhurdamatlık”a gelince adamın cüzamlı olduğunu keşfedeceksin! Oh ne âlâ memleket... Kal neymiş, milletin niyetini, verimkâr mı yoksa değil mi, öğrenmek için araştırma, soruşturma yapmak lazımmış... Bu çok önemli taharri için düğünden üç gün önceye kadar beklemek de neden? İYİ Parti ailesi horantanın, mahallenin, milletin kararını çoktan bilmekte ama onun kararı bir iç güveyi bulmak, burnuna halka takıp güdebileceği bir damat aramakta... Bunu 4 Şubat 2023 Cumartesi günü TELE1’in Forum Hafta Sonu programında söyledim,

“Meral Akşener güdebileceği bir aday arıyor” dedim. Böyle bir ahmak aday bulamaz(lar), bulurlarsa artık dünyanın sonudur! Hâlâ bulamamış ki İYİ Parti’nin profesörlerinden Ümit Özlale 6 Şubat 2023 tarihli Cumhuriyet’te “Tayyip Erdoğan kaybedecek, kazanacak adayı bulmalıyız” diyor. Geçenlerde de “Kendimi iyi ifade edemedim” diyordu galiba TELE1’de. İyi düşünemeyenler kendilerini iyi ifade edemez zaten. İyi düşünmek için bir dili iyi bilmek gerekir. Be kardeşim, Tayyip Erdoğan’ın kaybedeceği kesinse neden kazanacak aday arıyorsunuz, Kılıçdaroğlu da kazanır, öyle değil mi? Kazanacak aday seçim sandığında belli olur, Prof. Dr. Ümit Özlale “Önemli olan adayın ismi değil. Doğru adayın belirlenmesi” diyor. Monsieur de la Palisse de bir dostu için “Ölmeden on dakika önce yaşıyordu” demesiyle tarihe geçmişti Fransa’da...  """,
        "Özdemir İnce",
    ),
    (
        """Bilimci olması gereken bir profesörün “Ecel öldürür deprem vesiledir” dediğini duyunca benim de dilimin ucuna yazımın adı geldi. Başağrısı ölüme bahane olur da deprem olmaz mı, elbette olur. Bina yaptıranların, yapanların, yapımına ve iskânına izin verenlerin sorumsuzluğu, gevşekliği yanında yazımın adı çok ciddi kalır.

17 Ağustos 1999 İzmit-Gölcük depreminden başlayarak, 30 Ekim 2020 depreminden geçerek ve 6 Şubat 2023’ün 10 ilimizi ezen felakete gelinceye kadar ülkemizin bilimcileri, deprem bilimcileri, yerbilimcileri neredeyse kapı numarasına kadar olacak depremin adresini vermişler; 1500 bilmem kaçtan bu yana bölgede deprem öfkesi birikiyor, dikkat, demişler... Demişler ama ha bilimciler konuşmuş ha bizim Göde Omar’ın eşeği yellenmiş... Aynı bölgenin güneyinde askere hücum emri vermek için keçinin kuyruğuna bakan Osmanlı paşası gibi...

Daha fazlasını yazamayacağım, hepiniz 6 Şubat’tan bu yana olan biteni televizyonlardan izliyorsunuz... Eğer TELE1, Halk TV gibi televizyonun dışında kalan yandaş ve besleme televizyonlardan izlemişseniz felaketin boyutlarını kesinlikle anlayamazsınız! Hükümet yetkililerinden, AFAD yetkililerinden masallar dinlediniz, Başyüce’nin dünya liderleriyle yaptığı telefon konuşmalarını, “Başınız sağ olsun!” mesajlarını öğrenmişsinizdir.


Gerçek başka, çok başka, sanki Hatay ili Türkiye sınırları içinde değilmiş gibi... Her yerde bir düzensizlik, başıbozukluk, dağınıklık, basiretsizlik var ama sanki Hatay “Bizden değil” gibi... İskenderun’u deniz basmış, liman cayır cayır yanmakta...

Kaç gündür bir delikanlı “Yenge, yenge, duyuyorsan cevap ver!” diye bağırmakta bir enkazın başında, cevap aldı mı acaba?

“Biz AKP’liyiz, burası CHP belediyesi, AKP’liler buraya yardım ederse çok iyi olur!”

Pazartesi günü TELE1’i izlemekteyim, 14.30’dan itibaren... Sahneler ve anlar:

Bir kadın haykırmakta: “Devlet nerede, AFAD nerede? Ama kızım işte orada, enkazın altında ağlamakta...” 

Aslında “Hükümet nerede?” diye haykırması gerek kadının... Devlet aygıtının harekete geçmesi için hükümetin yetenekli olması, becerikli olması ve devlet canavarını uyandırması gerek!

Vatandaşın şu soruyu sorması gerek, haykırması gerek: Bu on ilin ve bütün ülkenin depremden en az etkilenmesi için gereken her türlü önlemin alınmış olması mı gerekirdi, yoksa Çanakkale Köprüsü mü, İstanbul’a yeri yanlış nazenin, fiyakalı havaalanı mı, İstanbul Kanalı procesi mi, yoksa onlarca halk bahçesi mi?...

Arada sevimsiz bir AFAD yetkilisi televizyon ekranlarından depremzedeleri taciz ediyor, ölü ve yaralı bilançosu okuyor ve son olarak da banka IBAN numarası veriyor!!! Bre adam, 20 yıldır toplanan deprem vergileri nerede? Millet can derdinde, herif para... Yıkıntıların başında hiçbir kurtarma ekibi yok... Ne vinç ne kepçe ne demir kesecek elektrikli testere...

Adana-Gaziantep otoyolunun bir bölümü göçmüş, yıkılmış, kırılmış... Böyle bir olayın depremcede bir adı var ama aklıma gelmiyor, artık siz anlayın... Fay mı? Göl üzerine yapılan havaalanı da neredeyse ikiye katlanmış ortasından... Yahu hükümet kardeş, İstanbul ve Hatay havaalanlarının yeri için sizi bilimciler kaç kez uyarmadı mı?

Bunları yazan ben ve benzerlerim, nankör ve merhametsiz değiliz! Ekranlarda gördüğüm manzara sanki atom bombası yemiş Hiroşima, Nagazaki manzarası... Yıkılan bina sayısı sanırım 15 binden aşağı olmayacak... Böyle bir yıkımla hiçbir ülke baş edemez, edemez!!! Evet ama hastaneler, belediyeler, okullar, kamu binaları yıkılmış... Bu nasıl olur, bu nasıl olur?

Namertten bir müteahhit bir yıl önce 15 katlı bir bina yaptırmış, güya 9 kuvvetinde depreme dayanıklıymış, elinde sertifika varmış... Ve daireleri iki buçuk milyon (2.500.000 TL) liraya satışa çıkarmış... Bina 6 Şubat zelzelesinde (yer sarsıntısında) un ufak olmuş... Bre Allah’tan korkmaz milletten utanmazlar, 2.5 milyonu sokakta mı buldunuz?

Bunlar kentlerde olanlar! Köylerde, kasabalarda kim bilir ne türlü facialar var? Ama bakarsınız Cumhur İttifakı seçimden önce son 20 yılın 10’uncu imar affını da çıkarır! """,
        "Özdemir İnce",
    ),
    (
        """Basından aktarıyorum: “Son dakika: İYİ Parti’de Kılıçdaroğlu istifası... Cihan Paçacı, genel başkan yardımcılığı görevinden istifa etti.

İYİ Partili Cihan Paçacı, genel başkan yardımcılığı görevinden ayrıldı. İYİ Partili Paçacı’nın CHP lideri Kılıçdaroğlu’nun söz konusu cumhurbaşkanlığı adaylığına ilişkin açıklamaları çok konuşulmuştu.

Altılı masanın toplantısı devam ederken İYİ Parti’den istifa haberi geldi. İYİ Partili Cihan Paçacı, genel başkan yardımcılığı görevinden ayrıldığını duyurdu.İYİ Partili Paçacı, bugün Habertürk’ten Nagehan Alçı’ya konuşmuş; CHP lideri Kılıçdaroğlu’nun adaylığıyla ilgili sözleri tartışma konusu olmuştu. Paçacı şu değerlendirmelerde bulunmuştu:”

“Biz başından itibaren ‘kazanacak aday’ diyoruz. Zaten herkes bu tanımda mutabık olacaktır. Şöyle bir şey var: Sokakta Kemal Bey’e itiraz görüyoruz, ‘Dürüst değil mi’ diye soruyoruz, ‘Dürüst’ diyorlar. ‘Devlet tecrübesi yok mu?’ diyoruz ‘Var’ diyorlar. ‘E o zaman?’ ‘Ama olmaz...’

Sokaktaki bu itirazı İYİ Parti olarak görmezden gelemeyiz. Parti yetkili kurulları şu anda Kemal Bey’i onaylayacak noktada değil ama bu nihai olarak böyle devam edecek anlamına da gelmiyor. Meral Hanım konuyu yetkili kurullara getirecek, orada nihai karar alınacak.”

Paçacı istifa kararını şu sözlerle açıkladı:

“Ülkemizin ve milletimizin umudu olan altılı masanın cumhurbaşkanı adaylık sürecine dair, bir gazeteciye verdiğim demecin, maksadı aşan yorumlara neden olduğunu görüyorum.

Demokrasimize nefes aldıracak bir süreçte büyük emek harcayan, sayın genel başkanımızı ve partimizi, aynı zamanda, altılı masayı oluşturan sayın genel başkanların iradelerini koruyup kollamak amacıyla; İYİ Parti’deki kurumsal ilişkiler başkanlığı görevimden istifa ediyorum.”

Çok güzel! Ama bu istifada çok yaşamsal bir sorun var: Kemal Kılıçdaroğlu’nun aday olması konusunda her şey tamam “Ama sokak ‘Olmaz!’ diyor” demek ne demek? Kılıçdaroğlu ne kusuru var da sokak “Olmaz” diyor? Sokağın tamamı mı, yoksa İYİ Parti’nin tabanı mı? İYİ Parti tabanının böyle bir mızıkçılık yapacağını hesaba katmadan mı masaya oturdu? Tabanda böyle bir mızıkçılık eğilimi varsa parti neden bu mızıkçı tabanı ikna etmek yoluna gitmiyor? Tabanını ikna etmek gücünden yoksun bir yapıya parti denir mi?

Cumhuriyet Halk Partisi, Türkiye Cumhuriyeti’ni kuran ve 1946 yılında (Demokrat Parti’nin kurulması) çok partili demokratik düzenin önünü açan ve 1950 seçimlerinden sonra iktidarı seçimi kazanan Demokrat Parti’ye teslim etmiş bir partidir.

CHP mevcut anayasaya ve kurumlarına, Siyasal Partiler Yasası’na tam anlamıyla bağlı bir partidir.

Ve Kemal Kılıçdaroğlu bu partinin genel başkanıdır. Açıkça ifade edilmese de İYİ Parti tabanı adı geçen kişinin dini inanç tarzına (mezhebine) muhalefet etmektedir. Ancak, biline ki bu muhalefet TC anayasasının tamamına ve Partiler Yasası’nın aşağıdaki maddelerine karşıdır.

SİYASİ PARTİLERLE İLGİLİ YASAKLAR
Madde 78

Siyasi partiler:

b) Bölge, ırk, belli kişi, aile, zümre veya cemaat, din, mezhep veya tarikat esaslarına dayanamaz veya adlarını kullanamazlar.Madde 81

Siyasi partiler:

a) Türkiye Cumhuriyeti ülkesi üzerinde milli veya dini kültür veya mezhep veya ırk veya dil farklılığına dayanan azınlıklar bulunduğunu ileri süremezler.

Madde 83

Siyasi partiler, herkesin dil, ırk, renk, cinsiyet, siyasi düşünce, felsefi inanç, din, mezhep ve benzeri sebeplerle ayırım gözetilmeksizin kanun önünde eşit olduğu prensibine aykırı amaç güdemez ve faaliyette bulunamazlar.

Cihan Paçacı’nın sözünden çıkamadığı “Sokak” CHP Genel Başkanı Kılıçdaroğlu’nun kimliğine karşı çıkarak anayasa ve siyasi sartilerle ilgili yasaklara göre suç işlemektedirler. Cihan Paçacı görevinden istifa etmek suretiyle aynı suça katılmaktan kurtulmuş değil. Ancak önemli bir sorun daha var: Böyle yasal kusurları olan bir seçmen kitlesini Cumhuriyet ülküleri doğrultusunda eğitemeyen bir parti nasıl Cumhuriyetçi bir parti olabilir? Demokratik cumhuriyet masasına oturan bir partinin yöneticisi tabanın hassasiyetinden söz edemez. Tek hassasiyet anayasanın ikinci maddesinde yazmaktadır.

Can sıkacak bir soru: Cihan Paçacı İYİ Parti’nin milletvekili aday listesinde olacak mı?  """,
        "Özdemir İnce",
    ),
    (
        """Bu yazı da altılı masa üyesi, Gelecek Partisi’nin “türban meftunu” genel başkanı Ahmet Davutoğlu ve genel başkan yardımcılarıyla ilgilidir.

***

Nur suresi, 31. ayetin türbanla ilgili olmadığını onlarca yazıyla kanıtladım. Kitap olur! Bugün bir kez daha kanıtlayacağım. Söz konusu ayetin en doğru çevirisi Muhammed Bin Hamza’nın 15. yüzyılın başlarında yaptığı çeviridir (Kültür Bakanlığı Yayınları, 1975).“Dakı eyit mu’mine avratlara; Örtsünler gözlerinin bir nicesin, dakı saklasınlar ferçlerini. Dakı göstermesinler bezeklerini... Dakı bıraksunlar derinceklerini göncükleri üzre...” (24:31)

Dakı: Ve. Eyit: Söyle. Ferç: Kadının ve erkeğin avret mahalli (cinsel organı). Bezek: Süs, ziynet. Derincek: Başörtüsü. Göncük: Yaka.

Günümüz Türkçesi ile anlam: “Ve söyle inanan kadınlara: Gözlerini (harama bakmaktan) sakınsınlar, ve saklasınlar cinsel organlarını. Ve göstermesinler zinetlerini (süslerini)... Ve yakaları (göğüsleri) üzerine bıraksınlar (indirsinler) başörtülerini (hımarlarını)...”

31. ayetin bu çeviri kadar harbisi yoktur. 15. yüzyılın saf ve temiz Türkü yalana dolana sapmadan cinsel organın Arapça adı soyadı olan “ferç”i (çoğulu: füruç) aynen alıp çevirisine koymuş. Arapçası da harbi zaten. Peki “Ayşe’ye söyle kapıyı kapatsın!”ın anlamı ne? Demek ki kapı açıkmış... Peki “Saklasınlar cinsel organlarını” demek ne demek? Demek ki cinsel organlar açıkta. Nasıl açıkta? Araplar, kadınıyla, erkeğiyle demek ki henüz don giymiyormuş o tarihte. Don giymedikleri için entarinin eteklerine dikkat edilsin isteniyor. 

Çeviri yapan yobaz hocalar “ferç” sözcüğünü kullanmamak için yerine “ziynet” gibi metaforlar kullanmışlar. Don işi tamam, gelelim “hımar” işine. Hımarla baş mı yoksa çıplak göğüsler mi örtülüyor?

Bu konuda epeyce hadis ve rivayet var. Bunlardan aklıma en yatanı bilginize sunuyorum: İbni Kesir, Mukatil bin Hayyan’dan, o da Cabir bin Abdullah el-Ensari’den rivayet ediyor:

“Esma binti Mirsed’in Beni Harise mevkiinde bir hurmalığı vardı. Kadınlar oraya etek giymeden, göğüsleri, saçları ve ayaklarındaki halhalları açık olarak giderlerdi. Esma ‘bu görünüşünüz ne kadar çirkindir’ derdi. Bunun üzerine, ‘Mümin kadınlara da söyle: gözlerini (harama bakmaktan) sakınsınlar...’ (Nur Suresi / 31. Ayet) ayeti nazil oldu.”

Nur Suresi 31. ayetin ikinci kilit sözcüğü “hımar”ın ne anlama geldiğini öğrenip önce kendisiyle tanışalım. Sözcüğün İslamla ilişkisi 31. ayetin inmesiyle başlamıştır. Hımar, İslamdan belki de binlerce yıl önce Arabistan topraklarında kadın, erkek, çocuk, yaşlı, Musevi, Hıristiyan, Mecusi, putatapar ve her inançtan halkın güneşe, rüzgâra, kuma, toza ve toprağa karşı başını örtüp kendisini koruduğu örtüdür. Öyle ki Hz. Muhammed’e Kuran’ın ilk ayeti nazil olduğu gün kendisinin ve eşi Hz. Hatice’nin başında mutlaka hımar vardı ve herhangi bir kutsallığı yoktu.

İslamcıların iddia ettiği gibi hımarın İslamla hiçbir ilişkisi yoktur. Kadınlar ve erkekler Müslüman olduklarında başlarında zaten binlerce yıldır hımar (örtü) vardı. Öte yandan İslamdan önce Arap kadınlarının göğüs bölgelerini örtmedikleri bilinmektedir. Kadınların İslama girmeleriyle birlikte giyim alışkanlığı rahatsızlık yarattığı için söz konusu ayet inmiş (yukarıda yazdığım hadis). Bu nedenle ayet kadınlara başlarındaki hımar denen uzun şeyi (yani örtüyü) artık ayıp sayılan çıplak göğüslerin üzerine indirmelerini (salmalarını) buyurmaktadır. Bu buyruktan sonra hımar (örtü) kadınların başlarını güneşe, rüzgâra ve kuma karşı korumaktan başka artık çıplak göğüslarini de kapatacaktır. Hımar denen şey günümüzde de Arabistan ve havalisinde kullanılmaktadır. Yazdığım örtünme âdetini bilmeyenler hiç kuşkusuz, işlerine geldiği için Tanrı’nın ve Kuran’ın kadınların başlarını örtmeyi buyurduğunu iddia ederler. İşportacıların “ikizlere takke” diye sattıkları her boy sutyeni bir yana bırakalım, Afrika’nın bazı bölgeleri dışında artık göğüslerini açık bırakan kadın yoktur. Başını örtmek isteyen kadın için türlü türlü olanak var: Şapka, bere, eşarp, yağlık vb.Yukarıda yazdıklarım, tanıklıklarım; Suriye kökenli şair ve filozof Ali Ahmad İspir (Adonis), Tunuslu profesör rahmetli Abdelwahab Meddeb ve Prof. Tahar Bekri, Faslı şair Abdelwahab Laabi gibi yazar, şair, filozof, sosyolog, tarihçi ve düşünürlerle yaptığım görüşmelere dayanmaktadır.

İslamcıların el çabukluğu marifetiyle kutsal anayasaya sokuşturmak istedikleri türban çaputunun gerçek öyküsünü okudunuz. Altılı masada da yüksek sesle okunmasını tavsiye ederim.

***

8 Ocak 2023 tarihli yazımda eksik kalan bir noktayı tamamlamak istiyorum: “DEVA Partisi Genel Başkanı Babacan cemaatlere yönelik ise ‘Din ve inanç topluluklarının örgütlenme özgürlüğü önündeki tüm engelleri kaldıracağız’ vaadinde bulunmuş.”

Bay Babacan, anlaşılan, anayasamızın 174. maddesinin koruması altında olan 677. sayılı “tekke ve zaviyelerle türbelerin kapatılmasına dair yasa”nın kaldırılmasını mı istemektedir. Bay Babacan, tek başına seçime katıldığı zaman Kürtlere anadilde öğrenim hakkı verilmesini ve 677. sayılı yasanın kaldırılmasını isteyebilir, altılı masada otururken isterse sabotajcı olur.

""",
        "Özdemir İnce",
    ),
    (
        """Bilim, dinsel inançlara terbiye dahilinde saygılı davranır ama bu türden inançların hatırına da bilim olmaktan vazgeçmez. Bilim nedir? İlk anlamıyla toplanmış bilginin bir yöntem olarak kullanım tarzıdır. Nesnel bilginin bulunduğu her yer ve alanda bilim yapılır. Bilim çoğuldur. Bilimin amacı, tahminler ve işlevsel uygulamalar yaparak dünyayı ve onun olgularını (fenomenlerini) bilgi temelinde anlamak ve açıklamaktır. Hem edinilen bilgiler hem de onu elde etmek için kullanılan yöntemler ve bilimsel veya katılımcı araştırmalar sırasında kullanılan kanıtlar açısından eleştiriye açıktır.

Somut verilere ve deneysel bilgiye dayanmadıkları nedeniyle bilim ve bilimci (yani bilim insanı) inançlarla uğraşmaz. İlgi alanının dışındadır. Ama dinsel bağnazlıkla donanmış inanç kıskanç ve diktatoryal olduğu için özgür ve bağımsız bilime düşmandır. Tuhaf bir durum: Bilim, dinsel inanca karşı saygılı ama dinsel inanç bilime düşmandır onu yok etmek ister..

Bu durumun en iyi örneği ülkemizin medrese kılıklı bir üniversitesidir: Van Yüzüncü Yıl Üniversitesi. Bu üniversitede neredeyse her yıl laikliğin hedef alındığı, Said Nursi’nin övüldüğü Uluslararası Bilimin Işığında Yaratılış Kongresi yapılır. 


Cumhuriyet gazetesi muhabiri Sefa Uyar’ın (12 Kasım 2022) yazdığına göre yapılan son toplantısının sonuç bildirgesinde, “Materyalist felsefenin, inançsızlığı yaşam tarzı haline getirdiği” iddia edilirken kongrede “Tabiat olaylarının Allah’ın isim ve sıfatlarıyla açıklandığı” belirtilmiş. Evrim, “Herhangi bir delile ve bilimsel veriye dayanmayan bir felsefi görüş” imiş. “Atomu ilah seviyesine çıkaran” ders kitaplarının güncellenmesi gerektiği savunulmuş. 

Materyalist felsefenin inançsızlığı yaşam tarzı haline getirmesinden size ne? Evrim hiçbir bilimsel kanıta dayanmayan bir palavra imiş!... Bunları hamamda kendileri çalıp kendileri söylerken iddia etmişler. 

İrlanda Başpiskoposu Ussher, Tevrat’ın Yaratılış (Tekvin) bölümündeki açıklamalar aracılığıyla bazı hesaplamalar yapmış ve dünyanın MÖ 4004 yılında yaratıldığını söylemekteymiş. Bazıları da dünyanın yaşının yaklaşık 10.000-20.000 yıl civarında olduğunu iddia etmekteymişler. Yaratılış’ta Tanrı’nın evreni ve insan dahil canlıları altı günde yarattığı, yedinci gün dinlendiği yazar ama bilime göre yaratılış milyarlarca yıl sürmüştür.

Van’lı teologlar Darwin’den önce bu hususla ilgilenseler çok daha iyi olur! Olmaz mı? Ussher’a göre insan 23 Ekim sabahı saat 9’da yaratılmış. Ben Bertrand Russell’ın yalancısıyım.¹

AKP saltanatında MS 600’lü yılların ikliminde yaşadığımız için, bilime ve evrime karşı duranları inandırmanın devenin iğne deliğinden geçmesinden daha zor olduğunu bilecek yaşa geldim. Yolcu yolunda gerek. Benim itirazım bu türden mürteci toplantıların yapıldığı yer. Neden Nurcu bir tarikatın izbesinde değil de bir Cumhuriyet üniversitesinde yapılıyor olması.

Yoksa Said Nursi’yi âlim sayarak göklere çıkaran, materyalist felsefenin inançsızlığı yaşam tarzı(!) haline getirmesine karşı olan bu ulemaya okumaları için birkaç kitap önerebilirdim. Telos Yayınevi’ni yönettiğim sırada, Fransa’da yayımlandığı yıl 1996’da ülkemizde yayımladığım Dünyanın En Güzel Öyküsü’nü² okumalarını salık verirdim. Bununla kalmaz, günümüzden 7 milyon yıl önce yaşamış “ön insan” Australopitekus kemikleri bulunurken, bilimciler 3 milyon yıl önce yaşamış Lucy Ana’nın kemiklerini incelerken³ neden âdemoğullarının o dönemlerden kalma iskeletleri ortalıkta yok diye de sormuyorum.

Ancak 7 Aralık 2022 tarihli BirGün gazetesinde (s.7) okuduğuma göre aynı türden mürteci programlar İstanbul Ataşehir İlçe Milli Eğitim Müdürlüğü’nün talimatı ile okullarda da yapılmaktaymış. Bilim düşmanlığının başında bir kadı var! Baş Kadı MEB’e şikâyet edilse ona madalya verir. Günümüz ders kitaplarının “Atomu ilah seviyesine” çıkardığını hiç sanmam ama “atom”a inanmayanlara Hiroşima ve Nagazaki’yi anımsatmak istemem.

1 Bertrand Russell, Din ile Bilim, YKY, s.37.


2 Şimdi Türkiye İş Bankası Kültür Yayınları tarafından Dünyanın En Güzel Tarihi yayımlanmış. """,
        "Özdemir İnce",
    ),
    (
        """Kuran’da, Tevrat (Eski Ahit) ve İncil’den (Yeni Ahit) epeyce alıntı vardır. Bu, ya tam alıntı olarak ya da edebiyat kuramında açıklandığı şekliyle başka bir bağlamda kullanılmasıyla olmaktadır. Bu yönteme edebiyatta “metinlerarası ilişki” (intertextualité) denir.

“İğne deliği” öznesi Matta İncil’inde yer alan bir ayet, Kuran’da başka bir bağlamda kullanılmaktadır.

Matta 19:23: “İsa öğrencilerine, ‘Size doğrusunu söyleyeyim’ dedi. “Zengin kişi Göklerin Egemenliği’ne zor girecek. Matta 19:24: Yine şunu söyleyeyim ki devenin iğne deliğinden geçmesi, zenginin Tanrı Egemenliğine girmesinden daha kolaydır.” *Araf suresi, 40. ayet: “Bizim âyetlerimizi yalanlayıp onlara kibirlenenler var ya! İşte göğün kapıları açılmayacak ve deve, iğne deliğinden geçinceye kadar onlar Cennet’e giremeyeceklerdir. İşte suçluları böyle cezalandırırız.” **

İncil’de zenginlerin cennete girmesinin imkânsız olduğunu yazarken Kuran’da Tanrı’nın ayetlerine inanmayarak kibirlenenlerin cennete giremeyeceğini yazmakta. Bu nasıl iş? Bu bir başka yazının konusu.Cuma günleri Hürriyet gazetesinde “Bugün Cuma” köşesinde yazan Naci Öncel de yukarıda benim sözünü ettiğim ilişkiyi ele almış. 14 Ekim 2022 günü “İğne Deliği” başlıklı makalesinde şöyle yazıyor:

[Başarısızlıkta faturayı başkalarına kesen insan nefsi, başarıları hemen kendi hanesine yazmaya meyillidir. Üstelik egomuz başarı, yüksek mevki ve zenginlik sahibi olunca daha da güçlenir. Hemen her konuda “Öyle yapılmaz, böyle yapılmalı” demeye başlarız. Öyle ki varlıklı bir kişi, kâinatın işleyişini bile beğenmez hale gelebilir.

İncil bu hali şöyle tarif etmiş: “Devenin iğne deliğinden geçmesi, zengin birinin ‘Tanrı egemenliği’ne girmesinden (iman etmesinden) daha kolaydır(Matta, 19:24).” İslam’a göre kudretli bir kişi, firavun örneğinde olduğu gibi “Ben sizin en yüce tanrınızım”(Naziat, 24)” diyecek kadar kibre kapılabilir. Kuran bu konuda “Kibirlenenler deve iğne deliğinden geçinceye kadar cennete giremeyecektir (Araf, 40)” diyerek son noktayı koyar. Yani kibir, özeleştiri önündeki en güçlü engeldir.]

İki kutsal kitapta Tanrı’nın cennetine kimler giremiyor? Zenginler (İncil) ve Tanrı’nın ayetlerine inanmayanlar (Kuran). Araf suresi 40. ayette sözü edilen insanlar Tanrı ayetlerine inanmadıkları için mi yoksa bundan dolayı kibirlendikleri için mi cennete giremiyorlar. Kibirlenmeseler cennete girebilecekler mi? Naci Öncel, Kuran’da ana eylemin “inanmamak” olduğunu yanlış yorumladığı için yanlış yorum yapmakta. Tanrı, o insanları kibirlendikleri için değil ayetlerine inanmadıkları, “Müslüman” olmadıkları için cezalandırıyor. Ayrıca, “Ben sizin en yüce tanrınızım” (Naziat, 24) diye hava atan firavunun bu yazıda işi ne? “Din” kıssalarından yararlanarak yazı yazanlar hep böyledirler. Ayetler ve bunlarda geçen sözcükleri, amaçlarına göre alıp kolaj yaparlar. 

Yazar ilgisiz alıntılarını, ıkınmalarını şu cümleyi yazmak için yapmakta: “Başarısızlıkta faturayı başkalarına kesen insan nefsi, başarıları hemen kendi hanesine yazmaya meyillidir. Üstelik egomuz başarı, yüksek mevki ve zenginlik sahibi olunca daha da güçlenir. Hemen her konuda ‘Öyle yapılmaz, böyle yapılmalı’ demeye başlarız. Öyle ki varlıklı bir kişi, kâinatın işleyişini bile beğenmez hale gelebilir.”

İncil ve Kuran’ın ayetleri ne diyor, Hürriyet’in yazarı ne diyor?! İncil’deki ayetin “zengin”i Kuran’da neden “Tanrı’nın ayetlerine inanmayanlar”a dönüşüyor? Kim yapmış bunu? Hürriyet yazarı bu konuda bir yorum yazarsa hep birlikte müstefit oluruz (yararlanırız).

""",
        "Özdemir İnce",
    ),
    (
        """10 Mart 2022 günü saat sabahın yedisi, perşembe... Yazı masamdan bakıyorum, karşı evlerin damları görünüyor, hepsi kar beyazı. Kar “Elif, Elif!”(1) diye diye yağmakta ve terasın duvarı üzerinde birkaç martı dolaşmakta... Dün (9 Mart Çarşamba), meteorolojinin tahminlerine dayanarak okullar pazartesiye kadar tatil edildi. Televizyonun dediğine göre İstanbul Valisi, İstanbul Büyük Şehir Belediye Başkanı’nı davet ederek birlikte kriz toplantısı yapmış. İBB Başkanı da toplantı düzenleyerek alınan önlemleri açıkladı. 

Pencereden bakarken bunları düşündüm ve meteorolojinin tahminlerine güvenip uyarak gereken önlemleri alan yetkili ve sorumlulara “Aferin!” dedim, alkışlayarak. Ama nedense mel’un aklıma Mersin Lisesi’ndeki (1949-1955) tarih öğretmenimiz Salih (İdikut) Ağa’nın sözleri geldi. Osmanlı paşaları keçi kuyruğuna bakarak hava tahmini yaparlarmış. Bilgisayarın başındayım ya Hacı Google’a “Keçi kuyruğuna bakarak hava tahmini yapmak” diye yazdım. Karşıma Cevat Kulaksız’ın bir yazısı(2) çıkmazmı… Bizim Salih Ağa’nın söyledikleri meğer gene doğruymuş…

Yazıyı okuyalım ve ardından kıssadan hisse çıkaralım: “Yıl 1839’dur. II. Mahmut ile Mısır Valisi Mehmet Ali Paşa kuvvetleri arasında meydana gelen ‘Nizip Savaşı’nda Osmanlı ordusuna Hafız Ahmet Paşa komuta etmektedir. Hafız Ahmet Paşa tecrübesizdir ve de komutasındaki generallerden çok, ulemanın düşüncelerine önem vermektedir. Osmanlı ordusunda danışman olarak Prusyalı üç kurmay subay bulunmaktadır.

Bunlardan biri de Moltke’dir. Moltke’ye göre Nizip’te Osmanlı ordusunun önemli birliğini yayalar (piyade) teşkil etmektedir. Erlere çarçabuk bazı şeyler öğretilebilmiştir. Subaylar ise subaylıktan hiçbir şey anlamamaktadırlar.

Mısır kuvvetlerinin başında İbrahim Paşa bulunmaktadır. Onların da durumu pek iyi değildir. Sayı bakımından iki ordu aşağı yukarı eşittir. Nizip alanında bu iki ordu harp kurallarına uygun olarak yerleştirilmiştir.

Prusyalı kurmay subaylar Osmanlı ordusunun Mısırlıları yenecek bir durumda iken hemen muharebeye girişilmesi için Başkomutan Hafız Paşa’ya tavsiyelerde bulunmuşlardır. Ancak ordu içinde bulunan ulema, o gün cuma olduğundan harp yapılmasının şeran caiz olmadığını ileri sürmüşlerdir.

Bir gün sonra Prusyalı subaylar bir gece baskını yapılmasını önermişler. Ulema bu defa da haydut gibi ansızın gece baskını yapılmasının padişah askerlerinin şanına yakışmayacağını bildirmişlerdir. ‘Peki, ne zaman hücum edeceğiz’ sorusuna da. ‘Keçinin kuyruğundan gelecek işareti bekleyeceğiz’ cevabını vermişlerdir. Osmanlı komutanı, yarın yağmur yağıp yağmayacağını bir neferin keçinin kuyruğuna bakarak yağmurun yağacağını veya yağmayacağını söyler. Mareşal Moltke bu cahilce uygulama karşısında şaşar kalır. ‘Keçi kuyruğundan hava raporu alan orduya benim yapacağım bir şey yok’ diyerek hayretini gizleyemez; ümitsizliğe kapıldığı için o yıl ayrılır gider.

(…)

Buna rağmen, Moltke yazdığı anılarına, ‘Benim taarruz planımı müdafaada durdurabilecek bir tek millet vardır, o da Türklerdir’ demiştir.”(3)Mareşal Moltke’nin son cümleyi neden yazdığını şimdi bilemeyiz: Gönül almak mı, yağ çekmek mi yoksa öte dünyaya inanmanın verdiği saçma cesaretin saptanması mı… ben bilemem. Siz ne dersiniz onu da bilemem… Ama son 24 saat içinde, devlet ricalinin meteoroloji bilimine inanıyor olmaları… Bu umut verici! Ne var ki hava durumu konusunda bilime inanan hükümet ve Saray ricalinin başka konularda bilimin kurallarından çok “keçinin kuyruğu”na inanmaları var.Uçuşlara kar engeli: “THY, İstanbul’da beklenen kar yağışı nedeniyle İstanbul ve Sabiha Gökçen havalimanlarında 10 Mart’ta sefer iptalleri yapıldığını duyurdu. Buna göre İstanbuI Havalimanı’nda 114’ü iç hat, 71’i dış hat toplam 185 sefer iptal edildi. Sabiha Gökçen’den ise 20 Anadolujet seferi iptal edildi. THY, 11 Mart’ta da İstanbul Havalimanı’ndan 179, Sabiha Gökçen’den de 34 olmak üzere 213 seferini iptal etti…”Ne olacak şimdi, iki meydanın işletmecilerine tazminat ödenecek mi, ödenecekse kim ödeyecek? Doğa mı, yoksa kim? Ödenecekse, bana sorarsanız “havaalanı” yerine “Hava Alanı” yaptıran AKP şirketi ödemeli!

Haaa, bir de Zihni Sinir procesi var: Üstü ve pistleri kapalı hava meydanı yaptırmak! """,
        "Özdemir İnce",
    ),
    (
        """“Ne olacak şimdi” netameli bir sorudur, akla hemen “Ne yapmalı”yı getirir. Bu soru da Vladimir İlyiç Lenin’i çağırır. Şimdi onu rahat bırakalım!

Sanki bütün partiler birbirine benzermiş gibi, iktidar koalisyonuna göre birbirine benzemez altı siyasal parti bir yuvarlak masa çevresinde oturunca “Şimdi ne olacak” diye sorulmaya başlandı. Armudun sapı üzümün çöpü!  İlkin Dövletlü Devlet Bey, kılınçtan keskin diliyle güzellemeler döktürmeye başladı: Kal neymiş, masanın gizli ayağı HDP imiş... HDP’nin gizli ayak değil de aleni (açık) bacak olmasının ne zararı var? Yuvarlak masaya yedinci, sekizinci partiler de oturabilir. Daha fazla da, size ne! Üstelik HDP en azından MHP kadar yasal ve ondan daha büyük oy oranına ve milletvekiline  sahip. HDP de aldığı oy oranında “milli irade”yi temsil etmekte. 

Efendim, Dövletlü Devlet Beylerin masası, müjdeler olsun ki köşeli imiş. Masa masadır. Poker masası da yemek masası da masadır. Neyse! Ama şu “Bunların altısını toplasanız bir etmez” deyişine takıldım. Altı tane biri yan yana yazıp toplasanız altı (1+1+1+1+1+1 = 6) altı eder. Biri altı ile çarpsanız (1x6 = 6) eder. Bu altı partinin oy toplamı ise yüzde 51.30; AKP + MHP = 33.1, veee HDP’nin oyu yüzde 12.9.


Vaziyetin durumu işte böyle!Kimi siyasetçiler ve gazete yazıcıları ve de özellikle televizyon hatipleri, özellikle de onlar (önce Millet İttifakı’na, şimdi de altı partiye) kasıla kasıla sizin “Ortak porgramınız ne, ekonomik programınız ne” diye sormaktalar. Bu kılçıklı soruyu ben yanıtlayacağım: Şimdilik bir toplumsal ve ekonomik plan ve programları yok, olmayabilir. Olamaz zaten! Ama bir siyasal planları var. Neymiş o? Şu: Tek adam zorbalığına son vermek ve usulüne uygun, adına yaraşır bir parlamenter düzen kurmak için bir ortak cumhurbaşkanı adayına hep birlikte oy vermek. Yukardaki yüzdeye, HDP’den ve şuradan buradan gelecek oyları eklesen en azından yüzde 55-60 oy eder ki AKP adayı nal toplar.Önümüzdeki milletvekili seçimlerine her parti kendi kesesinden, kendi hesabına girecek.

Sanırım, muhalefetin milletvekilleri, AKP ve MHP’nin milletvekillerinden fazla olur. Anayasayı değiştirmeye yeter mi yetmez mi, o ayrı hesap. 

Hükümeti, Millet İttifakı ve yandaşlarının seçim kazanan adayı cumhurbaşkanı sıfatıyla kuracak. Bakanlar kimler olacak, cumhurbaşkanının siyasi görevi belli zaten, demokrasi; ekonomik program da kısa zamanda ortaya çıkar. Şimdiden yapılsa olmaz mı? Olmaz! Çünkü nazar değer (!) Israrla ortak program soranlara bundan daha anlamlı bir cevap olamaz.Gardaş, sen televizyonda dandun edeceğine, ilkin R.T. Erdoğan’ı Saray’ından göndermek için yekin biraz. Saray’a değil, Çankaya Köşkü’ne yeni kiracı getirmek için kolları sıva! En iyi konuşmacı, en belagat sahibi, en etkili hatip yarışmasında birinci sensin zaten!Eğer bir erken seçim yapılmaz ise Türkiye’nin siyasal seçimi bundan sonraki genel seçimde (2028) ortaya çıkacak. Haaa, yeni cumhurbaşkanı, kabine kurmaktan feragat edip TBMM’deki partilerden birinin, tek başına ya da koalisyonla hükümet kurmasını isteyebilir mi? Valla benim şair aklım buna yetmez! Ya da bakarsınız, Demokrat Cumhurbaşkanı, hükümet kurma görevini bir parti genel başkanına verir.

Hedef böyle işte: Ortak akılsızlığın yerine demokratik, laik, eşitlikçi (sosyal) hukuk devletini getirmek!

Gerçek şu ki Türkiye birkaç yıl daha cumhurbaşkanlığı hükümet rejimi ile yönetilecek; ancak bu yönetim R.T. Erdoğan’ın yönetim tarzına hiç mi hiç benzemeyecek; cumhurbaşkanı, kanun hükmünde kararname yetkisini pek çok az kullanacak; kuvvetler ayrılığı 24 saat içinde restore edilecek; yeni üniversite açılmayacak, üniversitelere rektör atanmayacak; ama Boğaziçi Üniversitesi’nde poyraz esecek; mülakat sınavları hemen kalkacak; partizan atamalara son verilecek; yap-işlet-kazıkla yöntemi kesinlikle son bulacak... İçerde ve dışarda onurumuzu tekrar kazanacağız!


Millet İttifakı’nın cumhurbaşkanı adayı kim mi olacak? Özdemir İnce ile Göde Omar’ın olmayacağı kesindir. Ama Millet İttifakı, adam gibi bir adam bulacaktır elbette. Siz (kimseniz) kendinize bakın! """,
        "Özdemir İnce",
    ),
    (
        """Âdem ve Havva ile ilgili iki yazıyı şarkıcı (şantöz) ve güfte yazarı Sezen Aksu’nun hedef olduğu, tasarlanmış saldırıdan esinlenerek yazdım. Ve bu ilgi, bana Dr. Abdullah Cevdet üzerine kaleme aldığım ve henüz yayımlamadığım yazı dizisini hatırlattı. Büyük Cumhuriyet devrimcisi, benzersiz entelektüel, yazar ve çevirmen Dr. Abdullah Cevdet (1869-1932) üzerine 1 Ocak 2021 günlü Cumhuriyet gazetemizde “Dr. Abdullah Cevdet” başlıklı bir yazı yayımlamıştım. Bu yazıda o yazıdan da yararlanacağım. Dr. Abdullah Cevdet’i ve bütün Cumhuriyet devrimcilerini hedef alan kafa günümüzde de aynı kafa; sıfır numara cahil olduğu için “cahil” sözcüğünün çok geniş eşanlam kapsadığını kavrayamayan kafa!Dr. Abdullah Cevdet, geleceği de düşünerek çok önemli çeviriler yaptı. (Çeviri yapmayan ulusların gelişmeleri durur ve aklı kurur.) Doktor, bu gerçeği çok iyi bildiği için bir çeviri kitaplığı yarattı ve İçtihad adlı bir dergi yayımladı. Prof. Dr. Mustafa Gündüz, Abdullah Cevdet’in İçtihad dergisinde yayımlanan yazılarından bir seçme yaparak İçtihad’ın İçtihadı (Lotus Yayınevi, 2008) adıyla yayımladı. Hararetle tavsiye ederim!


Dr. Abdullah Cevdet, Vittorio Alfieri’nin Della Tiranide adlı kitabını İstibdad adıyla çevirmiş ve 1908 yılında Kahire’de yayımlamıştı. Kitap, Osmanlı dünyasında fırtına gibi esti. Çeviri doğrudan II. Abdülhamit’i hedef almıştı. Bu nedenle Cumhuriyet mürtecileri ondan nefret ederler. Basında ve internette Dr. Abdullah Cevdet hakkında araştırma yapanlar, onun Türk toplumunu medenileştirmek için Avrupa’dan damızlık erkek getirilmesini önerdiği iftirasını öğrenirler. Ama bu iftirayı bozan bir yazı da var (Sefa Kaplan, Hürriyet, 17.08.2005): 

“ ‘Avrupa’dan damızlık erkek getirtelim’ dediği gerekçesiyle adı Ankara’daki bir sokaktan silinen Abdullah Cevdet’in sırrı çözüldü.

Abdullah Cevdet, Mustafa Kemal’le yaptığı bir görüşmede, verimi artırmak için tarımla uğraşan göçmenlerin(1) Türkiye’ye getirilmesinin fayda sağlayacağını söylüyor. Ama haber ertesi gün Tasvir-i Efkâr’da, ‘Avrupa’dan damızlık celbini isteyen var’ manşetiyle yer alıyor. Abdullah Cevdet gazeteye tekzip gönderiyor, kendi dergisi İçtihat’ta böyle bir şey söylemediğini yazıyor ama dedikoduları engelleyemiyor. Öyle ki cenaze namazı bile büyük tartışmalara sebep oluyor.

‘Avrupa’dan damızlık erkek getirelim’ dediği gerekçesiyle Ankara Çankaya’da bir sokağa verilen ismi değiştirilen Dr. Abdullah Cevdet’in, böyle bir söz etmediğine dair ifadeler netleşiyor. Mustafa Kemal tarafından 1925 seçimlerinde Elazığ (Elaziz) milletvekili olması istenen Abdullah Cevdet, Çankaya’ya çıkarak Cumhurbaşkanı ile görüşüyor. Görüşme sırasında, Mütareke Dönemi’nden beri üzerinde ısrarla durduğu tarımda verimlilik bahsine değiniyor Abdullah Cevdet. Daha sonra da Mustafa Kemal’e, ‘Avrupa ülkelerinin özellikle tarımla uğraşanlarından getirilecek göçmenlerle ülkede nüfus artışı ve tarımsal gelişme sağlanması konusu’ndaki görüşlerini anlatıyor. Bu konuda tek nitelikli çalışmayı yapan ve halen Princeton Üniversitesi’nde öğretim üyesi olan Prof. Şükrü Hanioğlu’na göre, ‘Artık son faaliyetlerini sürdürmekte olan dinci çevreler bu beyanatı saptırarak kendisinin Avrupa’dan damızlık erkek getirmeyi arzuladığını’ iddia ediyorlar. (Kaynak: Bir Siyasal Düşünür Olarak Doktor Abdullah Cevdet ve Dönemi, Üçdal Neşriyat, Ankara, 1981, s.387.)

Bu nedenle, Abdullah Cevdet’in sözleri, dönemin muhafazakâr gazetelerinden Tevhid-i Efkâr’da çarpıtılan bir başlık ve yorumla yer alıyor. ‘Avrupa’dan damızlık adam celbini isteyen de var’ manşetiyle okuyucuya duyurulan haber-yorum şöyledir:

‘...Abdullah Cevdet Bey’in, bu sözlerini işittikten sonra, Elaziz’de bu adama rey değil, selam bile verecek Türk ve müslüman çıkmayacağına şüphe etmiyoruz (...) Fakat damızlık Alman ve İtalyan erkekleri getirip Türk kadınlarıyla izdivaç ettirmek ve onların kanını kanımıza karıştırmak isteyebileceğini doğrusu hatırımıza bile getirmezdik... Liberallik ve laiklik yapacağım diye her gün hezeyan kusan bu adamı Millet Meclisi’ne sokmak değil, Toptaşı’na tıkmak lazım gelir...’


Haber-yorumun yayımlanmasından sonra Abdullah Cevdet, Tevhid-i Efkâr’a tekzip gönderir, Akşam ve İçtihad’da meselenin aslını anlatır ama dedikoduları engellemesi mümkün değildir artık. Öyle ki, 1932 yılında kalp krizinden öldüğünde yapılan ilk tartışma, cenaze namazının kılınıp kılınmayacağına ilişkindir. Bazıları, dinsiz olduğu için cenaze namazının kılınmamasını, bazıları da Hıristiyan mezarlığına gömülmesini ister. Uzun tartışmalardan sonra, Müslüman bir anadan doğduğu için cenaze namazı kılınacak ve cenazesi Müslüman mezarlığına defnedilecektir.”

***

“Altı kaval üstü şeşhane” derler ya bunların da vücutları 2022 yılında ama kafaları 13. yüzyılda! """,
        "Özdemir İnce",
    ),
    (
        """İçişleri Bakanı Süleyman Soylu, Bursa AKP İl Başkanlığı toplantısında “Sadece bizim yaptıklarımıza bakmayın. Biz kendimiz yapmıyoruz. Biz inanıyoruz ki bize yaptıran Allah’tır, bize yaptıran Allah’tır, bize yaptıran Allah’tır!” ifadelerini kullanmış.

Tehlikeli ve anlamsız bir açıklama. Çünkü İslamın çıktığı yerde Müslümanlar, İslamın son din, Peygamber Hz. Muhammed’in son peygamber, Kuran’ın da son kutsal kitap olduğuna inanırlar. İçişleri Bakanı Süleyman Soylu’nun ağzından çıkan anlamsız sözler bu “statüko”ya da aykırıdır. AKP’nin, bu partinin genel başkanı R.T. Erdoğan’ın ve de Süleyman Soylu’nun 20 yıldır biriken vukuatlarının sebebi Allah ise yandık ki ne yandık!“Sadece bizim yaptıklarımıza bakmayın. Biz kendimiz yapmıyoruz. Biz inanıyoruz ki bize yaptıran Allah’tır, bize yaptıran Allah’tır bize yaptıran Allah’tır!” cümlesini ancak suçlular, suçları altında ezilenler kullanır.

Karısını, nişanlısını canavarca öldüren ve “Hep telefonla konuşuyordu hâkim bey, çok kıskandım, kendimi kaybettim, gerisini hatırlamıyorum, çok seviyordum pişmanım” diye kendisini savunmaya çalışan herif, artık Süleyman Soylu’nun izinden giderek “Siz benim yaptığıma bakmayın, bana bu cinayeti işleten Allah’tır. Alın yazım böyleymiş. Yazan da yaptıran da Allah’tır” derse kendisine “Affedersin, kusura kalma kardeş, biz yanlış anlamışız, başka bir şey sanmıştık” mı diyecek Süleyman Bey’in polisi, Abdulhamit Bey’in savcısı? Ve bu duruma muttali olan yani ıttıla kesp eden Umumi Reis Beyefendi bir Başyüce olarak ne yapacak? Ne yapacak, her şey yazılan, yazılmış olan ve yazılırken uygulanmakta olan akıldışı senaryoya uygun. Öyle deel mi?İnsanların dünyasında böyle şeyler olur, olmuştur. Oyuncak bebeğini parçalayan çocuk “Ben yapmadım, ellerim yaptı” diyebileceği gibi yaptığı işi bir arkadaşına da gönderebilir. Böyle şeyler yetişkinlerde olduğu zaman iş gelip kimlik bölünmesine kadar dayanır. Tıp bu ruh hastalığını şöyle açıklamakta: “Dissosiyatif kimlik bozukluğu türünde, kişi birbirinden farklı kimlik karakterlerini, aynı anda yaşar. Kişilik sayısı 5-10 arası değişebilmektedir. Ve bu türde kişi bir karakterden diğer karaktere ani bir geçiş yapar. Ve karakterler arası geçişte kimlikleri hatırlayamaz.”

Ama bu örnek Süleyman Soylugillerin durumunu açıklamıyor. Ya bu durumda Soylugillerin dünyevi varlıklarının içinde kendilerini yönlendiren bir kutsal irade merkezi var, eylemlerini bu yönlendiriyor. Ya da Karagöz gölge oyununda olduğu gibi, Karagöz ve Hacivat’ı ustanın parmakları yönlendiriyor.

İşin içinde Hasan Sabbah’ın Haşhaşi cenneti de var. Kafa bulanlar, doları Kılıçdaroğlu’nun yükselttiğini sanmakta kalmayıp iddia etmekte. İşin tuhafı acemi televizyoncular “Kılıçdaroğlu nasıl oluyor da doları yükseltiyor” sorusunu sormuyorlar.Şimdi biz Allah’ın Erdoğangillere, Soylugillere  yaptırttığı işlere ve yolsuzluklara bakalım. Önce “batan geminin malı gibi sattıkları”: Bir vatandaş CİMER’e son 18 yılda satılan fabrikalar ve kurumların listesini sormuş. Gelen listede satılan kurumlar, yerler ve fabrikaların listesi şöyle:

Termik Santrallar (8 adet), Hidroelektrik Santrallar (8 adet), Şeker Fabrikaları (10 adet), Tekel Binaları (15 adet), Sümer Holding’e bağlı şirketler (9 adet), limanlar (11 adet), diğer satılan şirketler (19 adet). Bu listede, Cumhuriyetin 1936-1984 arasında 48 yılda kurduğu ve AKP’nin sekiz yılda haraç mezat sattığı kâğıt fabrikalarının öyküsü yok. Şimdi bir sayfa kâğıt üretilmediği için yayıncılık ölmek üzere.


Peki, bu satışlardan gelen milyarlarca lira ya da dolara ne oldu? Parayı ne yaptıkları bilinmiyor: Allah’ın izniyle ya zimmete geçirildi ya da kumara basıldı.

YAP İŞLET DEVRET yöntemiyle yaptırılan ve Türkiye’nin 25-30 yıllık hazinesine ipotek koyan işler yaptırılmasaydı ülke ve vatandaşlar “hiçbir şey” yitirmezlerdi. Tamamı kirli pay (hisse) amacıyla yapılmıştır:

Osmangazi Köprüsü, Avrasya Tüneli, Yavuz Sultan Selim Köprüsü, Ankara Garı, İstanbul Yeni Havalimanı, şehir hastaneleri (10 adet), 1915 Çanakkale Köprüsü, Zafer Havalimanı (28 yıllığına işletilecek olan havalimanında verilen yolcu garantisi sayısı neredeyse Kütahya, Afyon ve Uşak’ın nüfusuna eşit).

Gerçekten bu işleri bunlara yaptıran böyle bir somut Tanrı var ise kurtuluş yok, demokrasilerde böyle bir Tanrı’dan da hesap sorulur! """,
        "Özdemir İnce",
    ),
    (
        """28 Ekim 2021 tarihli Cumhuriyet gazetemizden, “AKP’li Şenliklioğlu deistlere saldırdı” başlıklı bir haber kesmiştim. Adı geçen yazıcı, ateist ve deistlerin ana ve babalarıyla evlendiklerini ileri sürüyordu. Bir boş zamanımda internete başvurdum, söyledikleri bu kadar değilmiş. Okuyalım:

“Ölümüne AK Partiliyim” diyen yazar Emine Şenlikoğlu, katıldığı bir programda deistlere ilişkin olay yaratacak açıklamalarda bulundu. Şenlikoğlu, “Deizmde bir adam kızıyla evlensin hiçbir sakınca yoktur” dedi.Ateizmin “öldüğünü” öne süren Şenlikoğlu, konuyu “ensest ilişki”ye bağlayarak şunları söyledi:“Ateizmi öldürdük biz. Şimdi deizmi öldüreceğiz az kaldı. Çünkü ateizm çok mantıksız. Baktılar ateizm bitiyor, deizmi hortlattılar. Dinsiz de demiyorlar, deist diyorlar. Dinsizlik halkımızda kötü bir isimle anılır ya gençler sıcak baksın diye deist diyorlar. Ateizm ve deizm hayvanlık âlemi gibidir. Deizmde bir adam kızıyla evlensin hiçbir sakınca yoktur. Hangi ülkede hatırlamıyorum, bir adam köpeği ile nikâh yaptı.”

Bazı deistlerin ensest ilişkiyi kabul etmediğini dile getiren Şenlikoğlu, “Ama senin inandığın inanç buna karşı değil. Onlar anne babalarıyla evlenebilirler. Deizmde ahlak yoktur. Deizmin ateizmin kendisi bozuk” dedi. Şenlikoğlu, İrem Beyhan’ın “İnançsız bir insan iyi olamaz mı?” sorusuna da “Deizmin kendi içinde çok kibar hanımefendi, beyefendiler olabilir. Ama deizm çok ahlaksız. Her deist ya da ateist kötü olacak diye bir kaide yok. Ben sistemden bahsediyorum. Ben kişilerden bahsetmiyorum. Deist ahlaklı olabilir ama deizm ahlaksız” şeklinde yanıt verdi. """,
        "Özdemir İnce",
    ),
    (
        """“Ateizmi öldürdük biz. Şimdi deizmi öldüreceğiz az kaldı. Çünkü ateizm çok mantıksız. Baktılar ateizm bitiyor, deizmi hortlattılar. Dinsiz de demiyorlar, deist diyorlar. Dinsizlik halkımızda kötü bir isimle anılır ya gençler sıcak baksın diye deist diyorlar. Ateizm ve deizm hayvanlık âlemi gibidir. Deizmde bir adam kızıyla evlensin hiçbir sakınca yoktur. Hangi ülkede hatırlamıyorum, bir adam köpeği ile nikâh yaptı.”

Bazı deistlerin ensest ilişkiyi kabul etmediğini dile getiren Şenlikoğlu, “Ama senin inandığın inanç buna karşı değil. Onlar anne babalarıyla evlenebilirler. Deizmde ahlak yoktur. Deizmin ateizmin kendisi bozuk” dedi. Şenlikoğlu, İrem Beyhan’ın “İnançsız bir insan iyi olamaz mı?” sorusuna da “Deizmin kendi içinde çok kibar hanımefendi, beyefendiler olabilir. Ama deizm çok ahlaksız. Her deist ya da ateist kötü olacak diye bir kaide yok. Ben sistemden bahsediyorum. Ben kişilerden bahsetmiyorum. Deist ahlaklı olabilir ama deizm ahlaksız” şeklinde yanıt verdi.[CHP Genel Başkanı Kemal Kılıçdaroğlu, Adana’da katıldığı Dünya Avşarlar Derneği  dördüncü kuruluş yıldönümü şenliğinde özetle şunları söylemiş:

“Bizim de çok kabahatimiz, kusurumuz var. Bir başörtüsü meselesini Türkiye Cumhuriyeti’nin en temel meselesi haline getirdik. Sana ne kardeşim ya, kadın ister başörtüsü takar, ister takmaz. O kız çocuğumuz üniversiteye gidiyor mu, okuyor mu, imkânını sağlıyor muyuz? Derdin o olmalı. Çocuklarımız okumalı, bilimi öğrenmeli ve hayatı sorgulamalı. ‘Neden Türkiye bu haldedir?’ demeli. Bunları yapmalıyız.”Kılıçdaroğlu yanılıyor ve başörtüsü ile türbanı karıştırıyor. Sözcükleri yanlış kullanınca işte böyle olur. İslamcılar hile yaparak bir tür üniforma olan türbana “başörtüsü” dediler. Ancak Cumhuriyetçiler bu tuzağa düşmediler, başta ben fakir olmak üzere geleneksel başörtüsüne değil, türbana karşı çıktılar. Rahmetli dostum Tunuslu şair ve filozof Abdelwahab Meddeb (1946-2014) başörtüsü ile türban farkını şöyle açıklıyordu. Açıklamaya Türkiye’yi ben kattım:

“Geleneksel başörtüsünden ideolojik başörtüsüne (türbana) geçildi. Daha önce Pakistan’daki Hindistan’daki başörtüsü sariye benziyordu. Fas’taki ise cebellaya benziyordu. İkisinin arasında bir benzerlik  yoktu. Bugün, başörtüsü -ya da hicap- Endonezya’dan Paris’e, İstanbul’a kadar aynı: (türban yani). Geleneksel başörtüsü ile hiçbir ilişkisi yok, her yerde siyasal İslamın simgesi oldu. Evrensel amaçlı bir üniforma oldu. Henüz kazanamadı ama Müslümanın aklı (mantığı) İslamcılığın etkisine girdi. Böyle bir etki son derece tehlikelidir. (i)”Kıdemli imam-hatip Ahmet Hakan, Kılıçdaroğlu’nun bu açıklamasının üzerine mal bulmuş mağribi gibi atladı: “CHP, biraz da şartların zorlamasıyla ve hayli gönülsüz olarak başörtüsünü mesele olarak görmekten vazgeçtiğine dair işaretler vermişti ama özeleştiriye asla ve kata yanaşmamıştı. / Dikkat! Dikkat! / Bu bir ilktir! / CHP, ilk kez bu konuda yan yollara sapmadan şahane bir özeleştiri yaptı./ Mırın kırın  etmeden... Hepimizin ama hepimizin...  / Bu özeleştiri nedeniyle... / Ayakta alkışlamamız gerekir Kemal Kılıçdaroğlu’nu...” (Hürriyet, 5.10.2019)“Türban”, yukarıda da işaret ettiğim gibi İslamcılığın evrensel boyut kazanmak için kullandığı en önemli silahtı. Başörtüsünün Tanrı buyruğu olduğunu ileri sürüyordu. Ama yalan söylüyordu, Kuran’da başörtüsünü zorunlu kılan özel bir ayet yoktu. Nur Suresi 31. ayetini tanık göstermeleri de mümkün değil(di).

Nur Suresi 31. ayet: “Söyle inanan kadınlara: Harama bakmaktan sakınsınlar ve cinsel organlarını (ferçlerini) saklasınlar… Örtülerini göğüsleri üzerine indirsinler.”Bu ayet dilimize mealen türlü türlü tercüme edilmiş. Ama Muhammed bin Hamza 15. yüzyılda saptırmadan şöyle çevirmiş: 

“Dakı eyit mu’mine avratlara : Örtsünler gözlerinin bir nicesin, dakı saklasınlar ferçlerini. Dakı göstermesinler bezeklerini… Dakı bıraksunlar derinceklerini göncükleri üzre…” (Kültür Bakanlığı Yayınları, 1976, s.283-284)

Günümüz Türkçesi ile şöyle: “Ve söyle inanan kadınlara : Gözlerini (harama bakmaktan) sakınsınlar, ve saklasınlar cinsel organlarını. Ve göstermesinler zinetlerini (süslerini)… Ve yakaları (göğüsleri) üzerine bıraksınlar başörtülerini…”Derincek “başörtüsü” anlamına geliyor. Ama bu başörtüsü, kadınların, erkeklerin, Putperestlerin, Yahudilerin, Hıristiyanların, Müslümanların güneşten ve çöl kumlarından korunmak için başlarına örttükleri geleneksel örtü. Bugün de var. Kuran, “başınızı örtün” demiyor, “başınızdaki örtüyü çıplak göğsünüze indirin, salın” diyor. Çünkü İslamdan önce putperest Arap kadınları göğüslerini örtmüyorlardı.

Bu konuda birçok yazı yayımladım, ayetin Almanca, İngilizce, Fransızca ve İtalyanca çevirilerinden örnekler verdim. Bana sadece küfrettiler, ölümle tehdit ettiler.Türban geleneksel başörtüsü değildir. İslamcı cihadın simgesidir! “Türban”a “başörtüsü” demek Selefi İslamcı AKP’nin tuzağına düşmek olur!] """,
        "Özdemir İnce",
    ),
    (
        """Piyanist, öğretmen, müzik araştırmacısı Leylâ Pamir, bu kitapta sözcükle müziğin anlam ve anlatımlarının birbirleriyle ne denli yakın bir ilişkide olabildiklerini örnekliyor.

Müziğin, bir opera librettosundaki sözcüğün anlamını nasıl zenginleştirebildiğini, konuşma dilinin sözcükleriyle müziğin sözcüklerinin nasıl özdeşleşebildiğini; müzikle ilişkinin bir romanın sözcüklerindeki anlamlarla hangi müziksel düşünceler, besteciler, üsluplar, etkinlikler, tarihsel olgular, hatta kuramların imlenebildiğini; ya da bir müzik yapıtının anlatımıyla bir yazarın imgeleri arasında ne gibi benzerlikler bulunduğunu inceliyor.”

Sunuş’u okuduğunuzda bestecilerin eserlerinin tahlillerini öğreniyoruz:

“18. yüzyıl sonunda ustaca, zekice yazılmış Da Ponte’nin Don Giovanni librettosuna baktığımızda ve müziği ile birlikte dinlediğimizde, bir efsaneye dayanan, trajik bir ögeyi de içeren bu libretto mizahın ağır bastığı bir opera buffa’dır sadece.

Mahler, bu dünyaya ve sanatına ilişkin her şeye karşı duyduğu kırıklıkları mektuplarında sözcüklerle yansıtıyor.

1900’lerin Rus düşünürleri, yazarları, müzisyenleri ressamları ve dilcileri çok renkli bir kültür dünyası oluşturuyor. Bir yanda gerçekçi Romantik Rus yazarları, Rus Beşleri ve Çaykovski’nin müziği öte yanda özgürce seçilmiş dogmasız bir inancın içinde ‘eylem’, ‘bilgelik’ ve ‘sonsuzluk kavramı’nın bileşimiyle bir yaratıcılık felsefesini oluşturan Rus Simgecilerinin dünyası. Felsefeci Soloviyev ve Dostoyevski’den kaynaklanan bu düşünce çizgisinde, Batı’nın ‘sanat sanat içindir’ görüşünün, Nietzsche’nin eylem ve yaratıcılık idealinin ne büyük bir önem kazandığını görüyoruz.”

Müzikle edebiyat arasında kurduğu bağ, çeşitli edebiyatçılar ve edebiyat eseri ile bağlantıyı sağlamlaştırıyor.

Okumakla dinlemek arasındaki bağı kimler pekiştiriyor.
""",
        "Doğan Hızlan",
    ),
    (
        """Ülkemizde 7 Haziran günü Türk İşaret Dili Bayramı olarak kutlanıyor. İlk kez Türk Dil Kurumu’nun 2007 yılında düzenlediği Türk İşaret Dili Çalıştayının açılış günü olan 7 Haziran, toplantıya katılan işitme engellilerin sivil toplum kuruluşlarının önerisi üzerine oy birliği ile Türk İşaret Dili Bayramı olarak kabul edilmişti.

Bu yıl da Türk İşaret Dili Bayramı Türk Dil Kurumu’nda sergi, kitap tanıtımı, kutlama konuşmaları ve açık oturum etkinlikleriyle kutlanacak. Etkinlik Türk İşaret Dili Tarihinden Bir Kesit adlı serginin açılışıyla başlayacak. Sergide Osmanlı sarayındaki işitme engelli görevlilerin kendilerine özgü giysileriyle yer aldığı minyatür ve gravürler, işitme engelli okullarında kullanılmış olan işaret dili parmak Elifbaları ile çeşitli belgeler ve görseller sergilenecek. Açış konuşmalarının ardından Prof. Dr. Şükrü Halûk Akalın’ın yazdığı ve TDK’nin yayımladığı Türk İşaret Dili Tarihi adlı kitabın tanıtımı yapılacak.Türkiye Yüzyılında Türk İşaret Dili konulu açık oturumda ise Doç. Dr. Zeynep Oral işitme engelli çocuklar için çocuk edebiyatı üzerinde duracak. İşaret diliyle çocuk edebiyatının örneklerinin yer aldığı çalışmasına değinecek. Doç. Dr. Bahtiyar Makaroğlu Türk İşaret Dilinin Söz Varlığı, Banu Şahin ve Oya Tanyeri de Türk İşaret Dili Tercümanlığı ile ilgili sunumlar yapacaklar.

Herkese açık olan toplantı 7 Haziran Cuma günü TDK’nin Atatürk Bulvarı No. 217 adresindeki binasında saat 14.00’te başlayacak.

Türk İşaret Dili Bayramı’mız kutlu olsun!

MİLLİ MÜCADELE

MİLLİ MÜCADELE üzerine birçok kitap yayımlandı. Bugün yazacağım kitap bir başka açıdan özgün bir kimlik taşıyor:

“Zor Günlerden Zafere

Mehmet Can İlkin Koleksiyonu’ndan Kartpostallarla

Milli Mücadele

Murat Uğurluel”

Takdim:

“Milli Mücadele kartpostalları, halkı bilgilendirme, umut aşılama ve bir idealin altında bir araya getirme konularında önemli bir görevi yerine getirmiştir. Diğer taraftan işgaller sırasında Anadolu insanının yaşadığı zorluklar bu kartpostallarda oldukça başarılı bir şekilde yansıtılmıştır. Söz konusu kartpostalların beni en derinden etkileyen tarafı ise Türk kadınının Milli Mücadele’deki benzersiz rolünü net bir biçimde ortaya koymalarıdır. """,
        "Doğan Hızlan",
    ),
    (
        """Türkü dinlemeyi, yöreler arasındaki farkı öğrenmeyi çok severim.

Geçen hafta adeta türkülerle yattım türkülerle kalktım, onları derinlemesine müzik bilgimin içine yerleştirdim.

Eğer türküyü seviyorsanız, bu sevginizi pekiştirmek istiyorsanız Size Ahmet Emre Dağtaşoğlu’nun kitabını tavsiye edeceğim:

“Anadolu Türkülerinde

Semboller, Örüntüler

Ve Kültürel Bağlamlar”

Kuramsal saptamalarla uygulamayı bir arada mütalaa edince, bildiğim türkülere bir katman daha kazandırdım.

2004 yılından bu yana Açık Radyo’da “Dilden Dile Titreşimler” isimli halk müziği programını hazırlayıp sunan Ahmet Emre Dağtaşoğlu tarafından yazılan kitapta yaygın olarak bilinenlerin yanı sıra pek bilinmeyen türkü sözleri de ele alınıp inceleniyor. Kültürel bağlamlar göz önünde bulundurularak bu sözlerin içerdiği semboller ve örüntüler teorik bir çerçevede yorumlanıyor. Birçok soru Anadolu’nun sosyokültürel yapısının bazı özellikleri de dikkate alınarak cevaplanıyor.Dağtaşoğlu’nun yazdığı kitap ayrıca içinde yer alan türküler okunduğunda bir türkü antolojisi niteliği taşımakta.

Giriş’te açıklama notları yer alıyor:

“Malum olduğu üzere ‘türkü’ aslında bir edebiyat terimidir fakat günümüzde maya bozlak, gurbet havası, yol havası, tatyan, halay karşılama, semah, mersiye gibi isimlerle anılan tüm türleri kapsayan şemsiye bir terim olarak kullanılmaya başlamıştır.

Bu kitaptaki türkü metinleri de yazarların niyetlerinden ziyade kültürel dünyanın derinlerinde bulunan ilkeler dikkate alınarak daha geniş bir perspektiften yorumlanmıştır.

Kitabın ilk bölümü ile üçüncü bölümünde ele alınan türkülerin içerik açısından birbirlerinden ne kadar farklı oldukları okurların dikkatinden kaçmayacaktır. Bunun temel sebebinin sembollerin sıklıkla cinsel meseleleri üstü örtülü olarak anlatmak olduğu iddia edilebilir.

Kitapta yer alan türkülerinin sözlerinin çoğunluğu TRT’nin yayımlamış olduğu Türk Halk Müziği Sözlü Eserler Antolojisi ismini taşıyan iki ciltlik repertuvar kitabından alınmıştır.”

Dağtaşoğlu’nun kitabını okurken esere sadece toplu türküler kitabı olarak bakmayın bir sanat eseri bağlamında ortaya koyduğu gerekçelere de dikkat edin. Dinlediğiniz türkünün toplumdaki yankısını da bilginize katın. """,
        "Doğan Hızlan",
    ),
    (
        """Birçok yazarın mirasçısı bulunamadığı için kitabı basılamıyor, yasadaki boşluklar yüzünden yazar veya çevirmen unutuluyor.

Hükümetin bu belirsizliğe çare bulmasını hepimiz bekliyoruz.

Kimi mirasçılar da bilmeden bu unutulmaya alet oluyorlar.

Hiç kuşkusuz bu sadece edebiyat dünyasının bir derdi değil, müzik dünyasında da aynı belirsizlik hâkim.

Bazı hukukçular bu kitapların basılmasını uygun görüyorlar, önerdikleri çözüm şu: Belirlenecek telifin bir hesaba yatırılması. Yasal mirasçı ortaya çıkınca bu hesapta biriken paranın kendisine ödenmesi. Bu bana da akla uygun bir çözüm gibi geliyor.

Bazı hukukçuların da mirasçılara yüksek telif almaları yönünde tavsiyeleri oluyor.

Herkes bestselleri hayal ediyor. Müzik dünyasının mirasçıları ise daha da gerçeklikten uzakta.

Somut örnekler vereceğim.İyi çevirmen, birçok önemli kitabı dilimize kazandıran yazar Ahmet Cemal’in kitapları basılamıyor. Yayıncının söylediğine göre mirasçı sayısı onu geçiyor ve anlaşma sağlanamıyor.

Burada Ahmet Cemal’e yapılan edebi zulmün giderilmesi için yetkililer de harekete geçmeli.

Okurlardan bu kitapları uzaklaştırmanın inandırıcı, savunabilir bir gerekçesi yoktur.

İki iyi şairin de kitapları basılamıyor.

Ercüment Behzat Lav ve Celal Sılay’ın.

Antolojiler de yapılamıyor, oysa edebiyata ilk adımı atanlar için antolojiler çok önemidir.

Yavaş yavaş iyi edebiyatçıların eserlerini okuyamayacak, iyi müzisyenlerin bestelerini dinleyemeyeceğiz.

ÇOCUKLAR RESSAMLARIMIZI TANIYOR

HAYALPEREST YAYINLARI çocukların ressamlarımızı tanımaları için bir dizi hazırladı. Türk ressamlarının hayatlarını kendi ağızlarından öğreniyoruz. 

Dizinin önemini sanat tarihçisi, akademisyen Seda Yavuz kaleme getiriyor:

“Çocuk oyun oynayarak öğrenir. Oyun ise gerçek yaşamın yansımasıdır. Sanatın bir yansıma olduğunu düşünerek, çocukların sanata ilişkin bilgi edinmeleri ve hatta sanatın içinde olmaları özellikle zihinleri için gereklidir.

Feraye Turan Pir, çocukların bu sanatsal oyunda olmalarına katkı sağlamak amacıyla Türk Ressamları Serisi’ni kaleme aldı, ressam Helin Kurt resimleriyle seriyi zenginleştirdi. """,
        "Doğan Hızlan",
    ),
    (
        """Önce Aydın Gün’ü anmalıyız, İstanbul’da Devlet Operası’nı kurdu. İKSV’ye de uzun yıllar emek verdi. Müzisyen Azra Gün, oğulları ressam Mehmet Gün...Bir sanatçı ailenin güzelliğini yansıtırlardı.

Ankara’da operanın kuruluş yıllarını ince bir ironiyle anlatırdı. “Sahnedeki sanatçılar seyirciden daha çoktu” demişti.

Panayot Abacı hem müzisyendi hem de Filarmoni Derneği’nin yöneticisiydi; Opus dergisini çıkarırdı. Zaman zaman Taksim’deki bürosuna uğrardım, ayrıca Nuri İyem Ödülleri’nde Evin Galerisi’nde buluşurduk.

Önemli Yunan yazarlarını da dilimize kazandırdı.

Filarmoni Derneği’nin kurucuları arasında Nadir Nadi de vardı. O da keman çalardı, son zamanlarında ayakta karınlı mandolinde sevdiği bestecilerin eserlerini icra ederdi. Belleğim yanıltmıyorsa Filarmoni Derneği’nin konserleri de Saray Sineması’nda verilirdi.irkaç kez onunla birlikte Aya İrini’deki konserlere gitmiştik. Konser aralarında Cemal Reşit Rey’le konuşmaları, buluşmaların ayrı bir zevkiydi. Nadir Bey de küçük konser tanıtmaları yazardı.

Müzik yazarları arasında iki sürekli yazar vardı.

- Fikri Çiçekoğlu

- Faruk Yener

- Faruk Güvenç -Suna Kan’ın eşi

Yener, radyo konuşmaları da yapardı.

Andante’in bu sayısında neler var?

- Unutulan Kadın Bestecimiz Mihter Çelebi

Serhan Yedig

- Fagot Sanatçılığından Orkestra Şefliği’ne

Eray İnal

Menekşe Tokyay

Bir soru ve yanıtı : “Bir yandan ciddi bir bestecilik geçmişiniz var. Mesnevi Kanun Konçertosu’nu bestelemenizin ardındaki ilham kaynağı nedir ve bu eseri yazarken karşılaştığınız zorluklar oldu mu?”

Kanun sanatçısı Ahmet Baran’la birçok proje yaptık. Eserde birinci bölüm barok müziği andıran bir havayla ama makamsal olarak sultaniyegâh makamında başlıyor.

İkinci bölüm hicaz makamında hüzünlü, üçüncü bölümde Türk motifleri var. Bu konçerto Hasan Ferit Alnar’ın 1951’de bestelemiş olduğu Kanun Konçertosu’ndan sonra yazılan ikinci Türk kanun konçertosu. """,
        "Doğan Hızlan",
    ),
    (
        """Yapılanları anlatırken hiçbir övgü payından söz etmiyordu.

Daha sonra birçok açılışta, toplantıda konuştuk, kültürün birkaç büyük şehre değil, bir ülkeye yayılması gerektiği görüşünü öğrendim.

Bakırköy’deki kütüphanemin açılışına da geldi. Kütüphane politikasını da konuştum.

Kültür Yolu Festivali’nin zenginliğini, kapsama alanının genişliğini gazete haberlerinden okudunuz.

Yıllardır yazdığım, önerdiğim bir genişlemeyi Kültür Yolu Festivali’nde gördüğüm için bu yazımı yazdım.

Büyük kentlerde festivaller yapılır, konserler düzenlenirdi. Peki bu konserlere gelemeyen, sergileri göremeyen, festivale erişemeyen yurttaşlar ne yapacaktı. Koca yaz günü ya da mevsimin başka günlerinde çevreleri içinde kalacaklardı. Hiçbir faaliyetten yararlanamayacaklardı. Bencil bir kültür anlayışı idi.Yalnız dinleyiciler, sanatçılar için bunun önemini vurgulamıyorum, sanatçılar da her kentte sanata susamış insanlarla buluşacaklar.

Artık günümüzde ulaşım kolaylığına rağmen bir kentten bir kente gitmek gerek ekonomi gerek vakit açısından imkansızlaşıyor.

Öte yandan imrendiğim bir durum gerçekleşti. Yazlığa tatile giden biri de artık sanatın, kültürün nimetlerinden faydalanacak.

Ersoy, tanıtım toplantısında bakın ne demiş?

“Türkiye’ye adeta bir festival iklimi yaratacağız.”

Festivallerin bir başka işlevi olduğu kanısındayım, oradaki gençler de kendi yeteneklerini keşfedecekler.

Festivallerde ressamların olması benim açımdan önemli.

Genç bir öğrenci ustalara baktığında belki o da bu yolda yürüyecek. Dünyayı nasıl algılayacaklarını bu sanatçılardan öğrenecekler. Cumhuriyetin kültür girişimlerini bu festivallerle sürdüreceğiz. Ressamlarımız yurtdışına gönderildiler, döndüklerinde bilgilerini, ustalıklarını Anadolu’yu tuvale getirerek gösterdiler.

Bence en önemli insan eşitliği kültür eşitliğidir, bu festivaller bu eksikliği giderecektir. Yerelden evrensele yolculuğumuzda genç kuşaklara birer rehberdir.

Yaşadıkları yerlere kadar sanatı, sanatçıyı götürdüğümüz oranda çağdaş bir genç kuşak yetiştireceğiz. """,
        "Doğan Hızlan",
    ),
    (
        """Fatma Tülin’in ‘Modüs Vivendi’ adlı sergisi Piyalepaşa’daki Merkür Galeri’de açıldı.

Sergi mekânına girer girmez ilk aklıma düşen söz, sanatla zanaatı sanatçının usta bir biçimde kaynaştırması.

Çekirdek benim için birçok çağrışımı barındırır, insanın yaradılışına dair sırlar çekirdeğin içindedir. O çekirdeğin içinde, resimlerin de gizinde yalnız resim sanatının değil edebiyata dair de alıntılar vardır. Kimi zaman bunu sıradan bir ziyaretçi anlamadan geçer ama dikkatli bir sanatsever sanatın insanoğluna vermek istediği birçok düşünceyi içerdiğini fark eder.Sanatçının birey olarak varoluşunun öyküsünü ben onun resimlerinde bulurum.

Resimlerin yanı sıra heykellere dikkat etmenizi istiyorum. Mekân olarak da büyük çalışmaların yer aldığı heykeller bir başka açıdan bana neyi hatırlattı biliyor musunuz?

Çekirdek kavramı, imgesi resimde nasıl sanata getirilir, resimdeki yaratıya baktığımda biçimsel boyut anlayışı, yorumu türde de bir başka yorumu getiriyor.

Resimle heykellerin bir aradalığı FatmaTülin’in yaratma yoğunluğundaki türe sığmayış anlayışını açıklıyor.

Çalışmanın her zaman her yerde yapılması bir sanatçının gerçekten de her an yaratma sürecini yaşamasından kaynaklanır. Bir sözü çok hoşuma gitti konuşmasında: “Benim için her mekân çalışma alanı olabilir.” Müziğin yaratışının her evresinde olduğunu açıklaması, benim, eserlerine bir de bu türden bakmamı sağlıyor. Gerçekten de benim için müzik bir sanatçının çalışma ortamını zenginleştirir. Soyut ögeler kullanarak soyuta ulaşmak...

İyi bir sanat eseri karşısında bildiklerimizin ötesinde bir zevk alanına taşınamıyorsak, bilineni/bildiğimizi tekrar etmekle yetiniriz. Fatma Tülin bizi düşünmeye çağırıyor, bir kavramın, objenin her halini bize sergiliyor. Bir ziyaretçiye yeni tatlar aşılıyor, belki şaşırtıyor da.


Sergideki heykeller önce bronz döküldü, sonra da alüminyum kullandı.

Her zaman öneririm; bu heykeller bir kurumun, bir holdingin girişinde kullanılmalı. Biraz şaşırmak, biraz düşünmek, bilgimizi yenilemek... """,
        "Doğan Hızlan",
    ),
    (
        """“Eşim, insan dostum Yücel Dağlı’nın anısına.”

Uzun bir teşekkür listesi var.

Giriş, Prof. Dr. Süheyl Ünver’in sözüyle başlıyor:

“Mutfak deyip geçmemeli.”

Tıp doktoru Prof. Dr. Süheyl Ünver (1898-1986) Türkiye’nin ilk tıp tarihi enstitüsü ve ilk tıp tarihi ve deontoloji kürsüsünü kuran kişi olarak tanınmaktan başka, Türk kültürü ve sanatının her yönüyle ilgili eserler yayımlayan, önemini anlayan ve bu alanda araştırma yapan ilk kişi olarak hatırlanıyor.

Osmanlı mutfak kültürü ile ilgilenenlerin çoğu gibi ilk bilgilerime Prof. Dr. Süheyl Ünver’in ‘Tarihte 50 Türk Yemeği’ ve ‘Fatih Devri Yemekleri’ atlı kitabı öncülük etti.

- Kullanılan Kaynaklar

- Kapsam

- Etimoloji

- Sonuç """,
        "Doğan Hızlan",
    ),
    (
        """Klasik müzik konusunda merak ettiklerinizin yanıtını bu kitapta bulabileceksiniz.

Kitabın düzeni şöyle:

Ünlü bestecilerin biyografileri, müzik konusunda bestecilerin, orkestra şeflerinin sözleri, enstrümanlarla ilgili bilgi.

Seçtiğim bazı ana başlıklar:

- Orkestranın enstrümanları

- Opera

- Üç Tenor nerede, hangi tarihte konser verdiler? Domingo - Pavarotti – Carreras.

- En Ünlü Keman Konçertoları

- Bruch

- Mendelssohn

- J.S.Bach

- Beethoven

- Sibelius

- Glass

- Çaykovski

- Barber

- Elgar

- Brahms

Mozart’ın Hayatı:

Tam adı: Johannes Chrysostomus Wolfgang Theofilus Mozart

Sihirli Flüt üzerine ayrıntılı bilgi.

- Büyük Sonatlar

- Beethoven

- Mozart

- Chopin

- Rachmaninov

- Mendelssohn

- Bir Orkestra Şefi Nasıl Olunur?

Richard Strauss’un Genç Yönetmenin Altın Kurlları kitabından okuyabilirsiniz.- Otuzlu Yıllarda Ölen Besteciler:

- Schubert

- Bellini

- Mozart

- Bizet

- Purcell

- Gershwin

- Mendelssohn

- Chopin

- Weber

- Çocuklar İçin 10 Klasik

- Bartok – For Children

- Bizet - Jeux d’enfans

- Brahms – Lullaby

- Britten – The Young Persons Guide to the Orchestra

- Debussy – Children’s Corner

- Faure – Dolly Suite

- Mozart, L.Toy Symphony

- Faure – Dolly Suite

- Poulenc – The Story of Babar, the Little Elephant

- Prokofiev – Peter ant the Wolf

- Saint – Saens – Carnival of the Animals

- Filmlerde Kullanılan Klasik Müzik Parçaları

Uzun bir liste filmleri anımsıyoruz da acaba müziği anımsıyor muyuz?

(Classic Ephemera

A Musica Miscellany

Foreword by

Howart Goodal

Darren Henley and Tim Lihoreau)

DİSKOTEĞİMDEN

- TSCHAIKOWSKY KLAVIERKONZERT NO.1

Piano Concerto. Concerto pour piano

London Symphony Orchestra

Ivo Pogorelich - Claudio Abbado

Deutsche Grammophon

- SHOSTAKOVICH AND PROKOFİEV

Sonatas For Cello and Piano

Luba Edlina (piano) – Yuli Turovsky (çello) """,
        "Doğan Hızlan",
    ),
    (
        """“29 Ekim 1923’te ilan edilen ve yüzüncü yılını kutladığımız Cumhuriyet’in en yoğun yılı kuşkusuz ilk yılıdır. Yıllarca süren savaşlardan, salgın hastalıklardan, geçim sıkıntısından yorgun düşmüş bir halk...Siyasi tartışmalar, ekonomik sorunlar... Nüfus mübadelesi, idari yapıdan eğitime, yargı sisteminden belediyelere kadar her alanda yapılan yenilikler... Şehirlerin imarı, demiryollarının inşası… Yabancı şirketlerin millileştirilmesi gibi pek çok atılımla ülke yeniden inşa ediliyor.

Anayasa’nın kabulü...”

Bu liste bir çok alanı içeriyor.

Son satırlarla yazıyı noktalamalıyım:“Cumhuriyet’in ilk yılında, bütün muhaliflerin yanı sıra Meclis’te mebusların, Gazi Paşa’yı karşılayan halkın, grevlerde işçilerin, genel afla salınan mahkûmların, valiliğe şikâyete giden kadın ve çocukların, kısacası her kesimden halkın ortak sloganı şuydu: Yaşasın Cumhuriyet!”

Bu kitaptan her ayın kişilerini, olaylarını öğrenmek mümkün. Kitap, bir Cumhuriyet güncesi olarak nitelendirilebilir.

Cumhuriyet ilanından sonra üç yazarın görüşleri, bugün de araştırma yapacaklar için kaynak sayılmalıdır:

* Hüseyin Cahit’in, Tanin’deki yazısı:

‘Yaşasın Cumhuriyet’

* Velid Ebüzziya’nın Tevhid-iEfkâr’daki yazısı:

‘Efendiler, Devletin Adını Taktınız İşleri de Düzeltebilecek misiniz?’

* Necmettin Sadık’ın Akşam’daki yazısı:

‘Türkiye Cumhuriyeti’

Kitaptaki bölüm başlıkları:

* Hilafet Meselesi

* Karikatürler

* Şark Demiryolları Grevi

* Nüfus Mübadelesine Başlandı (Kasım 1923)

* İstanbul -Ankara Basını Kavgası

* İstanbul’un İktisadi İstikbali

* Makam-ı Hilâfet ve Âlem-i İslam

* Leblebici Horhor

* İsmet Paşa’nın ve siyasetçilerin yabancı basına verdiği mülakatlar

* Milli Musiki Başka, Türk Musikisi Başkadır

* Memleketimizde Sinema Hayatı

* Cumhuriyet’in İlk Yılında Kadınlar ve Siyaset

* Resimli Ay Yayına Başladı

* Yargı Sisteminin Laikleşmesi

* Darülbedayi Sanatkârları

* Rahmi Bey* Riyaseti Cumhur Musiki Heyeti

* Sahne Sanatçısı Hanımlarla Görüşmeler

* Yüzellilikler

* Anadolu Ajansı – Halk Fırkası

* İstanbul’un Beş Asırdan Beri Değişmeyen Bir Derdi: Hamal ve Gedikleri

* İstanbul Bir Günde Ne Yiyor?

* 30 Ağustos Kutlamaları

* Ben Deli miyim Yargılanıyor (Hüseyin Rahmi Gürpınar’ın kitabı)

* İstanbul’un Kurtuluş Bayramı Kutlamaları

* Ziya Gökalp

* Çankaya

Bazı kitaplar var ki zaman zaman ona başvurmak gereğini duyarsınız. İşte bu çalışma buna bir örnek.

Okuduklarınızın nasıl geliştiğini, sonra nerelere vardığını merak ediyorsanız bu kitabı kitaplığınızda bulundurmalısınız. """,
        "Doğan Hızlan",
    ),
    (
        """Sahaflar hakkındaki bu kitap birçok dostumu da bana hatırlatıyor.

Okumaya başladığımda, birkaç duyguyu bir arada yaşadım.

Kitaba meraklı olanların, tutku derecesinde kitapseverlerin kitapta yer bulmasının yanı sıra edebiyat tarihine düşecek notları da içeriyor.

Kitap yazarlarının biyografilerini okurum önce. Türkmenoğlu’nun biyografisini okuyun.

İlk sayfada ithaf var:

“Rahmetli Muhterem Büyükbabam Türkmenzade Mustafa Mehmet’e.”

‘Kitap Hakkında’dan:

“Babamdan tevarüs ettiğim bir alışkanlıktı. Sahaflar Çarşısı ve kitabiyata dair elime geçen fotoğrafları ve belge niteliğindeki dokümanları kabaca biriktirdim. ‘Söz uçar, yazı kalır’ düşüncesiyle ‘görüp işittiklerimden’ kendimce önemli bulduklarım hakkında notlar aldım.

Zaman zaman fotoğraflar çektim. Bir süre sonra ortaya çıkan arşivden haberdar olan kitap muhibbi dostlar, bunları derleyip yayımlamam için teşvikten öte baskı yapmaya başladılar.Bir gün Arslan Kaynardağ’ın ‘Sende Sahaflar Çarşısı ile ilgili dokümanlar varmış, onlara bir göz atayım. Bende olmayanların fotokopisini alayım’ demesi üzerine ‘Ben de sizdekileri görürsem olur’ cevabını verdim. Kabul etmedi. Ancak talebi bana cesaret vermiş oldu. Bunun sonucu olarak bir şeyleri kâğıda dökmeye başladım.

Nerede ise 15 yıl süren bu ‘demlenme’ süreci üç yıl önce tamamlandı.

Yazdıklarımı ham haliyle ilk defa Süleyman Şenel gördü, ayrıca salgın döneminde kitabın hazırlanması için yardımlarını esirgemedi.”

Diğer adlar da şöyle sıralanıyor: Şaban Özdemir, Ötüken Neşriyat Editörleri Göktürk Ömer Çakır, Oğuzhan Murat Öztürk, Ötüken Neşriyat Genel Müdürü Ertuğrul Alpay, Ahmet Emin Saraç, Güler Doğan Averbek, fotoğrafçı Adil Sarmusak.

‘Önsöz’den:

“İşte, 60 yıllık anılardan bir demet sunan bu kitap, kronolojik bir tarih kitabı veya anı defteri değil. Birazcık Yusuf Ziya Ortaç’ın ‘Portreler – Bizim Yokuş’u, biraz da Mina Urgan’ın ‘Bir Dinozorun Anıları’ kıvamında.”

Kitap 10 bölümden oluşuyor.

Kitap ve yayın dünyasından kişilerin tanıklıkları bu kitabın belirgin özelliği.

(Ötüken Yayınları)DEPREMLER hayatımızdan birçok önemli kitabı da aldı götürdü. Kitapçılar, özellikle sahaflar bir kültür depremini de yaşadılar, yaşattılar.

O kentlerde eğitim görenler, birçok kitabı sahaflardan edinirlerdi. Şimdi böyle bir kaynaktan yoksunlar. Özellikle üç ilimizdeki sahaflar bütün varlıklarını kaybettiler.

Dükkânları yıkılan, kitapları yok olan bu sahaflar için meslektaşları bir yardım ve destek kampanyası başlattılar.

İzmir’de bulunan Hermes Sahaf’ın sahibi Ümit Nar’ın koordinasyonunu yaptığı bu kampanyanın odak noktası, “KİTANTİK” isimli kitap satış sitesi oldu.

Başta İstanbul, Ankara ve İzmir gibi büyük illerimizdeki sahafların büyük desteğiyle hem para hem de zarar gören sahafların dükkânlarına koyabilecekleri kitapları toplama konusunda esnaf başarılı bir dayanışma örneği gösterdi. """,
        "Doğan Hızlan",
    ),
    (
        """Ercüment Ekrem Talu ile Çetin Altan’ın Tokatlıyan’daki yemek sohbeti bende yer etmiş.

Rahmetli arkadaşımız Tuğrul Şavkay bir gün evine yemeğe davet etmişti. Herkes “Ne zaman sofraya oturacağız?” diye sorarken Şavkay, bütün nezaketiyle “Sosu bekliyorum” cevabını verirdi.

Şekeri olan rahmetli dostumuz Oktay Kurtböke, “Yahu bana fenalık geliyor, peynir ekmek ver yeter bari” demişti bir keresinde.

Tuğrul Şavkay üstlendiği bütün görevlerde sofra adabını uygulardı.

Şimdi yerel mutfaklar da büyük kentlere geliyor.

Sofradaki yemek düzeni üzerine ne yapılmış?

UNESCO 2010 yılında Fransızların gastronomik yemeğine somut olmayan kültürel miras unvanını verdi.

Usul çok ağır olacak ki Edith Piaff’ı çıldırtmış.

İlk ünlendiği yıllarda sosyete mensuplarının verdiği yemekte aykırı davranmış.1950 Kuşağı’nın iyi şairi Ülkü Tamer bize patlıcanlı Antep kebabı yapardı.

Yemek kültürümüz üzerine çalışmalarıyla önemli ürünler ortaya koyan Günay Kut ile rahmetli Turgut Kut’u belirtmeliyim.

Tiyatrocu dostlarımın yemek davetleri de meslekleri gereği geç saatlerde gerçekleşir. Sevgili Haldun Dormen bizi gece saat 02.00’de sofrasında ağırlamıştı.

Sabri Koz’un ‘Geçmişten Günümüze Milli Yemek Kültürümüz’ (Türk Dünyası Vakfı, 2014) kitabı ilgilenenlerin kitaplığında mutlaka bulunmalı.

*

İ. Can Şiram’ın Oğlak Yayınları’ndan çıkan kitabı ‘Vegan Yarim’in kapağındaki takdimi içerik hakkında bilgi veriyor:

10 Vegan Kahvaltı

13 Vegan Meze

6 Vegan Sos

9 Vegan Çorba

31 Vegan Ana Yemek

6 Vegan Börek / Poğaça

10 Vegan Tatlı

7 Vegan Ek Tarif

Vegan Burgerler

Vegan Sucuklar

Vegan Pizzalar

Toplam 108 Vegan tarif

Şiram, ‘Başlarken’ yazısında veganlığa başlayışını ve gerekenleri açıklıyor:

“Bir akşamüstü, eşim, vejetaryen olmaya karar verdiğini bildirdi bana.

İlk başta onun bu tuhaf kararını anlamaya bile çalışmadım, geçici bir kafa karışıklığı olduğunu düşünüyordum, dolayısıyla durumu algılamam ve abuk sabuk şakalar yapmayı bırakıp buna saygı duymayı öğrenmem oldukça uzun sürdü.

Yaklaşık bir ay süren bu kabullenme süreci sonunda artık mangalda benim miktarını biraz azalttığım etlerin yanında eşimin sebzeleri, mantarları, peynirleri için yer açılmıştı.”‘Neden kitap yazdık?’ bölümünde öğrencilikte başlayan günlerden bugüne getiriyor.

Vegan mutfakta neler lazım?

Olmazsa olmaz ekipmanlar...

Yazar vegan olmanın felsefesine değiniyor:

“Veganlık, bir diyet çeşidi değil, hayatın her alanında ve kişisel bir eylemde pratiği olan bir felsefe ve yaşam biçimidir.

Ancak bu kitap en büyük (ve kolay) değişim adımı olduğuna inandığım ve bizzat uyguladığımız vegan beslenme alanına odaklanmaktadır.

Kitapta geçen veganlık, vegan yaşam tarzı ve benzeri deyişler de bu bağlamda yazılmıştır ve öyle anlaşılmalıdır.” """,
        "Doğan Hızlan",
    ),
    (
        """Yeteneğin genlerle geçip geçmediğini düşünürüm.

Edebiyatçı babaların edebiyatçı çocukları örnekleri olsa da çok azdır. Türkiye’de aklıma ilk Bener Ailesi gelir. Behçet Necatigil’in kızı Ayşe Sarısayın da tanıdıklarımdan. Ahmet Ağaoğlu’nun oğlu Samet Ağaoğlu ve Samih Rifat’ın oğlu Oktay Rifat ve torun Samih Rifat’ı da saymak gerekir.

‘Beş Romancı Tartışıyor’ kitabını hatırladım. Fakir Baykurt, Kemal Tahir, Mahmut Makal, Orhan Kemal ve Talip Apaydın baba kavramını tartışıyorlardı. Merkezlerine Balzac’ın ‘Goriot Baba’sını almışlardı.

Türk edebiyatında belleğimde canlananlar şöyle:

- Reşat Nuri Güntekin / Yaprak Dökümü.

- Yusuf Atılgan / Aylak Adam.

- Orhan Kemal / Eskici ve Oğulları.

- Oğuz Atay / Tutunamayanlar.

- İnci Aral / Kendi Gecesinde.

Türkiye’de âşıklar geleneğini sürdüren babalar ve oğulları da listeye eklemeliyim.Sabahleyin kitabevlerini dolaştım.

Bir rafta çocuklarla ilgili müzik kitaplarını gördüm. Eskiden bu tür kitaplar Yüksek Kaldırım’da Jorj Papajorj’da bulunurdu.

Okullarda müzik toplulukları kuruluyor.

Müzik kitabı raflarında eğitim öğretim için de kitaplara rastladım.

Tatil başladı, çocuklara televizyonlarda izahlı müzik saatleri yapılmasını öneriyorum. Bir zamanlar Amerika’da Danny Kay, Türkiye’de de Hikmet Şimşek yapardı. Enstrümanları tanıtırlardı, onların yer aldığı bestelerden parçalar seslendirirlerdi.

Bu bütün müzik türleri için yapılmalı. """,
        "Doğan Hızlan",
    ),
    (
        """Vehbi Koç’un kızı Semahat Arsel’in ailenin tarihini yazıya geçiren ‘Kuşaktan Kuşağa’ kitabı, belgesel bir anlayışı yansıtıyor. Hazırlayan, Ayşe N. Sümer.
Ankara’dan başlayarak İstanbul’da devam eden kitabın içeriğinde, Koç ailesinin yanı sıra yakın akrabaları ve dostlarını da öğreniyoruz. Ticaretten sanayiye, Türkiye’nin gelişme sürecinin her aşaması bu kitapta yer alıyor.
Önsözden bir bölüm:
“Aile büyüklerimiz hayattayken, geçmişle ve eskilerle, ailemizin kökleriyle, atalarımızla ilgili aklımıza gelen soruları onlara sorar, bilgi alırdık.
Önce annem, sonra babam ve etrafımızdaki yaşlı akraba ve tanıdıklar teker teker vefat edip aramızdan ayrılınca, geçmişle ilişkilerimizin bazı yerlerde tamamen koptuğunu, bazı yerlerde de unutulduğunu daha fazla hissetmeye başladım. Artık soracak insan pek azaldı. Yok gibi... Sadece bir iki kişi kaldı. Hayretle fark ettim ki ailenin en büyüğü ve geçmişle bağlantısını kuran ben oluvermişim. Akraba ve tanıdıklar, kardeşlerim, yeğenlerim eskiyle ilgili bilgileri bana sormaya başladılar. Kitabımın amacı, dinlediklerimi, yaşadıklarımı ve gördüklerimi anlatırken, zamanla ‘yaşam tarzının’ örf ve âdetlerin nasıl değiştiğini, ülkenin, sosyal, ekonomik ve politik açılardan nelere tanıklık ettiğini bizden sonraki kuşaklara aktarmaktır.” """,
        "Doğan Hızlan",
    ),
    (
        """İstanbul’da yaşayan herkes bir kez olsun Mısır Çarşısı’ndan geçmiştir. Zaman içinde her semt gibi orası da değişti.

Mısır Çarşısı’nın yanındaki Tahmis Sokağı’na mutlaka uğrardım. Daha caddenin başında sizi kahve kokusu karşılar. Kurukahveci Mehmet Efendi’ye uğrar, zevkime göre kavrulmuş taze kahveyle Hollanda kakaosu alırdım. Baharatçılar, kuruyemişçiler oranın süsüydü.

Ö. Sıla Durhan ve Yekta Özgüven’in ‘Mısır Çarşısı’nı Düşünmek: Mekânsal Pratikler, Özneler, Gündelik Yaşam’ kitabını anılarım eşliğinde okudum.

‘İçindekiler’ bölümü bu konuda iyi hazırlanmış bir kitap olduğunu yeterince kanıtlamakta. İşlenen konulardan bazıları şöyle:

* Osmanlı Döneminde Çarşı Yapıları ve Düzeni

* Mısır Çarşısı’nın Yapım Tarihi ve Mimarları ile İlgili Tespitler

* Mısır Çarşısı ile İlgili Venedik Yerleşiminden Osmanlı’ya Uzanan İzler“Mısır Çarşısı İstanbul’da Tarihi Yarımada’da, Eminönü’nde Yeni Cami’nin hemen yanında yer alan bir anıtsal yapı ve halen baharat satışı ile ticari faaliyetine devam eden bir tarihsel imge. Ancak, bu kadarı çarşıyı anlatmaya yetmiyor.

Öyleyse nedir Mısır Çarşısı? Birçok şeyin hepsi birden. Bir çarşı değil, birden fazla çarşı. Yüzyıllardır Yeni Cami Külliyesi’ni gözleyen bir arasta. Tek bir yapı değil, külliyenin parçası. Bir döneme ait değil, birbiri üzerine yığılmış birçok döneme ait. Bugün hâlâ yaşayan özgün bir baharat çarşısı olmasının yanı sıra modern hayatın izlerini de taşıyan bir tüketim mekânı. Birbirine karışmış çeşitli kokuları soluyarak; çeşitli lokum, şekerleme ve kurutulmuş meyveyi tadarak geçip gittiğimiz bir sokak. Uzak Doğu’dan getirilmiş türlü hediyelik eşyayı ilk kez görür gibi incelediğimiz bir pazar yeri. Kentin tam ortasında, kozmopolit kalabalığın en yoğun olduğu yerde bir kamusal alan. Nostalji arayışındakilerin uğrak yeri. Belleklerde çoklu çağrışımlar zinciri oluşturan bir keşif mekânı. İlk bakışta tuhaflık, uyumsuzluk ve karmaşıklık olarak algılanabilecek bir manzara veya her şeyin birbiriyle kaynaşıp özgün bir bütünde toplandığı bir sistem. Kısacası görme, işitme, dokunma, tatma ve koku alma yoluyla deneyimlediğimiz bir mekân Mısır Çarşısı...Mısır Çarşısı, gerek İstanbul’un yüzyıllar boyu ekonomik, toplumsal ve kültürel merkezi olan Tarihi Yarımada’nın gerek Bizans döneminden beri İstanbul’un değişmez ticaret mekânı olan Eminönü’nün simgelerinden biridir. İstanbul’un tarihi, mimari ve kültürel mirasının bir parçası olan Mısır Çarşısı, kentlilerin yaşamlarında aidiyet kurduğu ve bağlandığı, toplumun her kesimini kapsayan kamusal bir mekândır. İnşa edildiği
17. yüzyıldan günümüze uzanan bir aralıkta baharat ya da aktarlar çarşısı olma işlevini kısmen korumuş, fiziksel olarak da ayakta kalmıştır.”

İstanbul’da tanınması gereken mekânlardan biri... """,
        "Doğan Hızlan",
    ),
    (
        """Yeni açılacak kütüphaneye benim adımı vereceklerini söyledi yetkililer.

Böyle onursal bir çağrıya Yemen’de olsam kuşun kanadına biner gelirdim, yürekten ‘Evet’ dedim.

Bazı günler insanın biyografisinde büyük harflerle yazılmalı, ben de böyle bir gün geçirdim.Açılışa Milli Eğitim Bakanı Mahmut Özer’in geleceğini belirttiler, ben eski kuşaktan bir İstanbullu olarak, devletin ilgisi her zaman beni mutlu eder.

Okula girer girmez beni Milli Eğitim Bakanı Mahmut Özer ile İstanbul Valisi Ali Yerlikaya karşıladı. Elbette İstanbul Milli Eğitim Müdürü, okulun müdürü onlarla birlikteydi.

Bu tür ödüllendirmelerin yalnız alana değil, bu töreni seyredenlere de bir mesaj olduğu kanısındayım. Bir mesleğe, bir sevgiye, hele kitaba kendinizi adarsanız mutlaka sizi takdir eden, ödüllendiren bir kurum, o kurumu temsil edenler çıkar.Hazırlıkları, düzenlemeyi gördükçe, yaşadıkça tören protokol açısından gerçekten çok başarılıydı diyebilirim. Böyle toplantılarda içerik kadar biçim de önemlidir, hatta zaman zaman öne geçer.

Konuşmak için salona girince öğrencileri, davetlileri görünce doğrusu heyecanlandım. Böyle toplantılarda heyecan benim peşimi bırakmaz.

Sunucumuz zarif hanımın verdiği kısa bir bilgiden sonra genç müzisyenler sahnede yerlerini aldılar.

Eski açılışları, sunuşları anımsarken, şimdi müziğin katkısının önemini, toplantıyı ne kadar güzelleştirdiğini fark ediyorum. Benim gibi müziksiz yaşayamayan biri için hoş karşılamaydı. Doğrusu oturduğum yerden onlara eşlik ettim. Sonra kitap üzerine kısa bir konuşma yaptım.

Birçok yazarın kitapla yakın arkadaşlığı bir film karesi gibi gözümün önünden geçti, benim bu kürsüye çıkmamı sağlayan ustalarıma içimden selamlarımı, saygılarımı ilettim.

Asaf Halet Çelebi’nin koltuğunun altında mutlaka birkaç kitap olurdu.

Haldun Taner, Kadıköy vapurunda ayakta kitabını okumayı ihmal etmezdi.

Benden sonra Milli Eğitim Bakanı Mahmut Özer bir konuşma yaptı, kütüphane ağının nasıl genişlediğini, kitapsız bir okul kalmasın diye gösterdikleri çabadan bizi haberdar etti.Kütüphane programının da tamamlandığı müjdesini verdi.

Vali Ali Yerlikaya, İstanbul’un kültür olaylarını takip ediyor ve katılıyor.

Gerçekten de ben bazı belediyelerin kütüphanelerini gezdim, akşam saat beşte kapanırdı kütüphaneler, kimse de yararlanamazdı. Evim Fatih’te olduğu için Fatih Belediyesi’nin kütüphanesini gezdim, öğrencilerin ruhu da midesi de düşünülmüştü.

Zeytinburnu’nda bir sanat kütüphanesi açıldı.

Trafiğin yoğun olduğu İstanbul’da semt kütüphanelerinin önemini vurgulamaya gerek yok.

Konuşmalardan sonra kütüphaneye girdik. """,
        "Doğan Hızlan",
    ),
    (
        """Kütüphanemizi zenginleştiren kitapları aldığımız kitapçıları anımsarız. Ben, Babıâli yokuşundan inerken uğradığım kitabevlerini, onların sahiplerini unutmam. Elbette kütüphanemizde bazı kitaplarda sahafların da emeği vardır. Bir okuryazarın, hele inceleme, araştırma yapacakların
en yakın dostu sahaflardır. Şimdi kitap fuarlarında sahaf stantları büyük ilgi görüyor; yalnız İstanbul’da değil, ülkenin her kentinde bu ilgiye tanık oldum.

İsmail E. Erünsal’ın ‘Osmanlılarda Kitap Ticareti-Sahaflar ve Kitapçılar’ı, kitabın bizdeki tarihi üzerine okunması gereken önemli bir çalışma. Kitap dünyasını, geçirdiği evreleri belgeler ve bilgilerle yazmış. Böyle kitapları ben zaman zaman okuma ihtiyacı duyarım. Çalışma kitaba olan sevgiyi, saygıyı arttırıyor; A’dan Z’ye kitapla uğraşan herkesi sevdiriyor. Erünsal, kitabı kimlere ithaf etmiş: “Sahaf esnafının son temsilcilerinden İbrahim Manav Bey’e ve merhum İsmail Özdoğan ağabeyimin aziz hatırasına...”Kitabın başındaki ‘İthaf ve Teşekkür’ yazısından bir bölüm: “Bu kitabı kendilerine ithaf ettiğim sahaf esnafının son temsilcilerinden İbrahim Manav ve İsmail
Özdoğan ağabeylerimin sahaflarla tanışmamda önemli payları vardır. Üniversitedeki öğrencilik yıllarımızda sınırlı cep harçlığımızla bir müddet merhum Necati Alpas’ın dükkânında kitap yığınları arasında eşelendikten sonra, kapı komşusu olan İbrahim Manav Bey’in dükkânına terfi. Burada Necati Bey’in dükkânının aksine her şey yerli yerinde, düzen ve intizam içindeydi. Her ne kadar kitap yığınları arasında önemli bir kitap keşfetmek zevkinden mahrum kalsak da burada mesleğimiz için önemli ve olmazsa olmaz kitapları, şuara tezkirelerini, vakanüvis tarihlerini, divanları, sözlükleri bulurduk. İbrahim Bey’in dükkânından sadece kitap almazdık. On beş dakikada bir uğrayan çaycının dağıttığı çaylardan da nasiplenirdik ve daha da önemlisi her birinin ayrı özellikleri olan o dönemin enteresan şahsiyetlerinin sohbetlerini dinlerdik.” """,
        "Doğan Hızlan",
    ),
    (
        """Kent Oyuncuları’nı  birçok kişi seyretmiştir. Yıldız-Müşfik Kenter kardeşlerin yanında Kent Tiyatrosu’nun önemli bir ismi de şair ve tiyatro dünyasının unutulmazı Kâmran Yüce’dir. Üçünü de seyreden biri olarak tiyatro tarihimizdeki vazgeçilmez yerlerini unutmamak gerektiğini düşünüyorum. Hiç kuşkusuz bu isimlere Şükran Güngör’ü de katmak gerekir.

Çevirmen, yayın yönetmeni Kâmran Yüce’nin kızı Deniz Yüce Başarır’ın büyük emekle hazırladığı ‘Perde Kapanmasa Görecektiniz’ kitabı, yalnız Kenterlere değil, tiyatro tarihimize de ışık tutuyor.

Kitabın kapağı şöyle:

“Perde Kapanmasa Görecektiniz

Kâmran Yüce’nin arşivinden Kent Oyuncuları’nın Kuruluş Hikâyesi (1959-1986).”

Kitabın adı Yüce’nin bir dizesinden alınmış.

Kitabın ilk yazısı İstanbul Büyükşehir Belediye Başkanı Ekrem İmamoğlu’na ait…Deniz Yüce Başarır ‘Başlarken’ yazısında kitabın oluşumunu kaleme alıyor. Çalışmaların temelini gerçekleştiren Başar Başarır’ı şu cümleyle tanıtıyor: “Değerli eşim Başar Başarır’dan rica ettim: Babamın arşivini benim için tarar mısın, diye. Çünkü tiyatronun belleği, 1986’daki ölümüne kadar oyunculukla birlikte, dergi, afiş, ilan, basınla ilişkiler gibi birçok işi de yürüten babam Kâmran Yüce idi.

Umarım okuyanlar arşivlere, belgelere, anılara dayanan ama bir tiyatrocu kızı olmanın getirdiği, belki sahne tozu iddialı olur ama kulis tozunu okuruna da bulaştırmaya niyetlenen, bu geçmiş zaman hikâyesinden keyif alırlar. Sırf
o siyah beyaz fotoğraflara bakarak bile bir zaman tüneline gireceğinizin garantisini verebilirim en azından. Ben çok güzel vakit geçirdim.”

Ben de nice oyunu o tiyatroda seyrettim, nice Türk oyun yazarını onlar sayesinde fark ettim. Siyah beyaz fotoğraflar benim de anılarımı tazeledi.

Kitabı okuduğunuzda, eserde Başar Başarır’ın nasıl bir görev yüklendiğini fark ediyorsunuz.

‘İçindekiler’e baktığınızda, Kenterler ekseninde Türkiye’de tiyatro dünyasının röntgenini görürsünüz.

Anabaşlıklardan bazıları şöyle:

- Babam, ben ve Kent Oyuncuları

- Site Tiyatrosu

- Ve onlar artık Kent Oyuncuları- Harbiye’de yeni dönem Hamlet ile başlar

- Ayrılık rüzgârları

- 80’ler… Tiyatro canlanıyor

Bazı özellikler çalışmayı beğenmemi, önemli bulmamı sağladı: Kent Oyuncuları’nın hemen hemen bütün oyunlarını seyrettim. Türkiye’de bir tiyatro kurmanın, onu yaşatmanın zorluğunu öğrendiğimde, yaptıkları iş gözümde daha da büyüdü.

Oyunları seyrediyorsunuz, alkışlıyorsunuz; kulisi, o oyun sahneye gelinceye kadar çekilen hem sanatsal hem ekonomik sıkıntıları düşünmüyorsunuz.

Hem belgelere dayanan hem de birinci elden anıları, tespitleri bize aktaran bu kitap belgesel yanıyla da övülmeye değer. Türkiye’deki bir anlayışın da altını çiziyor. Bir kuruluşun geçmişini tanıyoruz. """,
        "Doğan Hızlan",
    ),
    (
        """Otomobille, karavanla Türkiye gezisine çıkarsanız, durduğunuz her yerde “Burada ne yenir, en ünlü yemeği hangisidir, bunu en iyi hangi lokanta yapar” diye sorarsınız.

Ömür Akkor hazırladığı ‘Türkiye Gastronomi Atlası’nda işte bu soruları yanıtlıyor. Sizi yemek arama zahmetinden kurtarıyor.

Çok karayolu seyahati yapan biri değilim ama gittiğim kentlerde de böyle bir danışma kitabına ihtiyaç duyarım.

Ömür Akkor, tutkulu bir gezgin, biraz maceraperest, çok şeyin tadını çıkarmış biri.

Hayatından birkaç not yazdığımda gerisini okumak istersiniz...

- Kilis’te doğdu.

- 25.000 kitaplık özel kütüphanesi var.

- Yılda 90.000 km civarında seyahat etti.

- Yayımlanmış 28 kitabı var.

- Kazılara katılmış.

- Görme engelliler için yemek kitabı hazırladı.

- 25 Aralık 2020’de dünyanın en yüksek üs noktasında (Hakkâri, İkiyaka-3500 metre) askerlerimiz için yemek yaptı. - Kitapları yurtiçinde ve yurtdışında ödüller aldı. """,
        "Doğan Hızlan",
    ),
    (
        """Her bayram tatilinde bunu düşünürüm, bizden kaç kişi bu tür kitapları alıp okuyarak gezer?

Canlı rehberler var ama bilgiler ancak kitap sayesinde kalıcılık kazanır.

Yazlık yörelerde, vatandaşlarımıza, yabancı turistlere kendimizi nasıl tanıtacağız? Sadece doğadan, yenilen yemeklerden mi söz edeceğiz? Beni hiç ilgilendirmiyor.

Bütün dünyada yazlık yörelerde müzeler açılıyor, gelenler ülkenin sanatını, edebiyatını tanıyorlar, yoksa yenilen lahmacun ile yeni açılan beach’lerin hiçbir faydası yok.

Bir ülkenin yabancılar tarafından tanınmasını nasıl sağlayabiliriz?

Geçici resim sergilerinin Bodrum’da açıldığını okuyorum, bakıyorum yabancı bir yazar, sanat tarihçisi yok, biz bize fotoğraflar bunu gösteriyor, rahmetli Çetin Altan’ın dediği gibi bizi bize övüyoruz. Kış geliyor, onlar da bitiyor.Yabancı ülkelerde, oranın mimarisine uygun müzeler yapılıyor, hatıra kartları basılıyor. Müze katalogları satılıyor.

Kitap Fuarı için Gaziantep’e gittim, Zeugma Müzesi’ni gezdim.

Tatiller sadece yazlık yerlere gitmek midir? Kendi doğduğunuz, sonra başka kentlerde yaşamınızı sürdürdüğünüz göz önüne alındığında, oraya gitmelisiniz, değişimi görürsünüz, orayı tanıtan kitaplar edinebilirsiniz.

Program yaptığım, tanıştığım bazı yazarlar doğdukları şehri terk etmemişler, onu Türk ve dünya edebiyatında tanınır düzeye çıkarmışlar.

Ben bir başka hususu da anımsatmak isterim.

Kültür Bakanlığı’nın TEDA projesinde çalıştım, kurulun başkanı Prof. Mustafa İsen’di.

Yurtdışında bir kitabın basılmasını sağlamak için, tanıtımı, basılması, çevrilmesi için belli bir katkı verilirdi.

Onların ne olduğunu bakanlık izlemeli, onlardan birer adet getirtmeli. Özellikle yazlık yörelerde kurulacak kütüphanelerde, yabancı turistler için o kitaplar raflarda yer almalı. Gelenler Türk edebiyatını onlardan öğrenmeli.

Elbette Kültür ve Turizm Bakanlığı’nın yayın kurulu için de bir öneride bulunacağım. Türkiye’nin özellikle yabancıların yoğun olduğu yörelerindeki kütüphanelerde az sayfalı edebiyat tarihimiz, resim/heykel tarihimiz, müzik tarihimiz, mimari tarihimiz üzerine en az dört dilde kitaplar hazırlatmalı.Türkiye’nin Onur Konuğu olduğu yıl Frankfurt Kitap Fuarı için ben ‘Türk Edebiyatının 100 Köşe Taşı’ adlı bir kitap hazırlamıştım, iki dilde yayımlanmıştı.

Rahmetli Füsun Akatlı da ‘Türk Denemeci ve Eleştirmenler’ kitabını hazırlamıştı.

Bu girişimlerin bugün de sürdürülmesinin gereğine inanıyorum. """,
        "Doğan Hızlan",
    ),
    (
        """Filistin Kültür Bakanlığı'nın şubat ayında yayımladığı bir rapora göre, savaşın ilk dört ayında Gazze'deki 32 kültür kurumu ya hasar gördü ya da yıkıldı. Bu yıkım, Gazze'deki sanatçıların eserlerine yansımaya başladı. Sanat, bu zor zamanlarda adeta direnişin ve umudun bir ifadesi haline geldi. Sanatçılar, yaşadıkları trajedileri ve mücadeleleri eserleri aracılığıyla dünyaya duyurmaya çalışıyorlar. Bu sanatçılardan biri, Gazze'nin El Rimal bölgesinde yaşayan Sohail Salem. 47 yaşındaki beş çocuk babası Salem, savaşın acı gerçekleriyle başa çıkabilmek için teselliyi resimde buluyor. Onun için resim, bir tür günlük tutma yöntemi.Eserlerini sosyal medyada paylaşarak hem kendi duygularını ifade ediyor hem de dünyanın dikkatini Gazze'de yaşananlara çekmeye çalışıyor. Salem'in mahallesi, geçen yıl çatışmalar başladığında İsrail ordusu tarafından ilk hedef alınan bölgelerden biriydi. Ordunun... """,
        "Funda KARAYEL",
    ),
    (
        """Bir yer düşünün ki, sonsuzluğu temsil eden bir vaha gibi 20 milyon metrekarelik bir araziye yayılsın. Bu alan, 2 bin 600 futbol sahası büyüklüğünde ve Avrupa'nın en büyük güneş enerjisi santrali olarak parlıyor. 3 milyon 256 bin 38 adet güneş paneliyle 2 milyon hanenin elektriğini sağlayan ve yılda 3 milyar kilovatsaat elektrik üreten Kalyon Karapınar GES'ten bahsediyorum.Bu santral, her yıl 1,7 milyon ton karbon emisyonunu önleyerek çevreye nefes aldırıyor. Kalyon PV'de üretilen güneş panelleriyle yapılan bu devasa tesis modern çağın mimarlık harikası olan SCADA binasına da sahip. Gündüzü başka, gecesi ayrı güzel olan bu ikonik binanın, mimar Caner Bilgin'in Kalyon Holding'in düzenlediği yarışmada birinci seçilen tasarımı ile hayat bulduğunu öğrendim.
5 Haziran Dünya Çevre Günü'nü, bu sene Kalyon PV ev sahipliğinde, Birleşmiş Milletler İnsan Yerleşimleri Programı Gençlik Danışma Kurulu (UN Habitat) Heyeti'nin katılımı ile Kalyon Karapınar'da... """,
        "Funda KARAYEL",
    ),
    (
        """Bir eser düşünün ki, insanlığın kendi kırgınlıklarıyla barışmasına odaklanırken kırılganlığı bir zayıflık ve zaaf olarak değil tam tersine bir güç olarak ele alıyor. Bu yıl beşincisi düzenlenen ve Türkiye'nin ilk uluslararası çok-disiplinli (multi-disipliner) destinasyon festivali olan Cappadox'daki Hale Tenger'in eserinden başkası değil bahsettiğim.Değişen Gökler sergisi kapsamında Güvercinlik Vadisinde yine doğayla çevrili ortamda sergilenen çalışmada sanki kayalar, ağaçlar ve hayvanlar hep bir ağızdan şu soruyu soruyorlar: Yapmadan olabilir misin? Tenger'in ses yerleştirmesi, kayaların sizinle konuştuğu hissini uyandırıyor. Varoluşsal sorular üzerine derin düşüncelere dalma alanı sağlıyor. Festivalin en güzel yanlarından biri eserleri harita üzerinde işaretleyerek keşfe çıkmaktı. Tenger'in eserinden sonraki durağım Nermin Er'in Dinle serisi eserinin bulunduğu tepe oldu. Eser sesin evrende kaybolmadığı bilgisinden yola çıkarak, sessizliğin sesini... """,
        "Funda KARAYEL"
    ),
    (
        """Bir sabah telefonum çaldı. Arayan arkadaşım, heyecan dolu bir sesle, "Gözlerimi seninle aynı renk yaptırdım!" dedi. İlk düşüncem, mavi lens takmış olabileceği yönündeydi. Fakat konuşmanın devamında öğrendim ki, arkadaşım göz rengini değiştirmek için ciddi bir ameliyat geçirmiş. Evet yanlış duymadınız göz rengini değiştirme ameliyatı. Bu yeni trend beni şaşkına çevirdi. Burun estetiği, botoks, dudak dolgusu derken şimdi de göz rengini değiştirmek mi moda oldu diye düşünmeye başladım. Araştırmalarım sonucu öğrendim ki, bu operasyon körlüğe kadar varan ciddi komplikasyonlara yol açabiliyormuş çünkü kullanılan boya, gözün içine enjekte ediliyormuş.Daha neler göreceğiz, daha neler duyacağız? İnsanların uğruna sağlıklarını riske attıkları bu güzellik trendleri gerçekten de çılgınlık boyutuna ulaştı. Arkadaşımın bu cesaretine mi şaşırmalıyım yoksa göz sağlığını bu kadar kolay tehlikeye atmasına mı bilemiyorum. Günümüzde sosyal medya, mükemmeliyet algısını körüklüyor.... """,
        "Funda KARAYEL"
    ),
    ("""Bu hafta şahane bir sergi deneyimi yaşadım, üstelik hamamda. On Majlisism: spatial studies and prototypes by Pierre Paulin sergisi Selcan Atılgan küratörlüğünde Tarihi Küçük Mustafa Paşa Hamamı'nda Türkiye'nin önde gelen sanat koleksiyonerlerinin katılımıyla açıldı. Benjamin Paulin serginin açılışına kendi bizzat katıldı ve gelenleri deneyim yaşamaları için 'Lütfen esere dokunun dinleyin hissedin' diyerek yönlendirdi.YEREL DOKUNUŞLAR 1960'lar ve 1970'lerin modernist hareketleriyle ilişkilendirilen Paulin, organik formlar, yumuşak hatlar ve yenilikçi yapılarla tanınan kendi tarzını yaratmış. Sergide yer alan Pierre Paulin'in Tapis- Siège parçaları, Orta Doğu kültüründe ortak toplantılar ve geleneksel oturma düzenleriyle ilişkilendirilir. Bu parçalar, halının konforunu ve rahatlığını koltukların işlevselliğiyle birleştirerek, Orta Doğu'daki sıcak ve... """,
        "Funda KARAYEL"
    ),
    (
        """Eurovision şarkı yarışması bu yıl gündemi iki yüzlü bir tutumla belirledi. Avrupa Yayın Birliği (EBU), Rusya'nın Ukrayna'ya savaş açmasının ardından hızlıca düğmeye basmıştı ve Rus sanatçıları Eurovision Şarkı Yarışması'ndan men etti. Peki İsrail ve Gazze savaşı için yarışmadan beklenen reaksiyon geldi mi?Tabii ki geldi! İsrail, protestoların gölgesinde Eurovision büyük finaline katılmaya hak kazandı. Avrupa Yayın Birliği EB, Eurovision Şarkı Yarışması'nın, dünyanın dört bir yanındaki izleyicileri müzik yoluyla birleştiren apolitik bir etkinlik olmaya devam ettiğini söylüyor. Ancak bu yıl, "apolitik bir etkinlik" olma iddiası altında gizlenen bazı soru işaretleri var. Tüm bu olanlar gerçekten de politikasız bir karar mıydı, yoksa EBU da biraz siyasetçi mi dersiniz? İşte burada İsrail devreye giriyor. EBU, Rusya'ya kırmızı kart gösterirken, İsrail'e 'Buyurun, efendimiz!' dedi bu sene. Eğer Eurovision gerçekten apolitikse, neden İsrail'e kapıları sonuna kadar açıyoruz da Rusya'yı... """,
        "Funda KARAYEL"
    ),
    (
        """Mayıs ayı New York, sanat dünyası takvimindeki en büyük zirvelerden biri; ancak bu yılki fuar ve müzayede dalgası, ABD'de ekonomik tahminlerin kötüleşmesi ve İsrail'in Gazze'de devam eden savaşının Manhattan'daki üniversite kampüslerinde protestoları alevlendirmeye devam etmesiyle birlikte başladı. Sanatçı Ricci Albenda'nın Frieze New York'un bu yılki sayısında yer alan tablosunda "Para önemli değildir" ifadesi yer alıyor. Eser fuarın en konuşulan eserlerinden aynı zamanda çelişki oluşturan işlerinden biri oldu. Geçen yıl Frieze New York'ta 2,5 milyon dolarlık bir Jack Whitten tablosunun satışı görülürken, bu sefer hiçbir galeri bu kadar büyük bir işlem bildirmedi.Bir kez daha, birden fazla operasyonda galerileri bulunan iki önemli operasyon olan Hauser & Wirth ve White Cube, büyük satışlar bildirenler arasındaydı. Peki fuarda hangi eserler satıldı? Yaptığım araştırmalara göre Fiyatı: 850.000$ değerinde olan 2001 tablosu, fuarın ön... """,
        "Funda KARAYEL"
    ),
    (
        """Avrupa'nın kalbinde, kültürlerin buluştuğu noktalarda, insanlar arasında bir köprü görevi üstlenen etkinliklerin değeri büyüktür... Yunus Emre Enstitüsü ve Lacivert Dergi'nin işbirliğiyle gerçekleşen 'Yurt Dışı Buluşmaları' da tam da böyle bir köprü işlevi görüyor. Bu kez Amsterdam'dan Lahey'e uzanan bir söyleşi yolculuğuna çıktık.Klinik Psikolog Beyhan Budak'ın derinlikli psikolojik sorgulamaları, düşündürücü ve öğretici bir deneyim sunarken, Lacivert Dergi Genel Yayın Yönetmeni, yazar Mustafa Akar, Yunus Emre'nin günümüzde bizim için ne ifade ettiğini, Daily Sabah Genel Yayın Yönetmeni İbrahim Altay ise hikayeler anlatmanın hayatımızdaki önemini anlattı. Başarılı oyuncu Fadik Sevin Atasoy ile Amsterdam'da gerçekleştirdiğimiz söyleşide ise kırmızı bavulun Fadik'in kariyerindeki öneminden yola çıkarak 12 yıllık Los Angeles yolculuğunu ve bavuldaki diğer... """,
        "Funda KARAYEL"
    ),(
        """New York, sokaklarında her daim bir şov vardır. Ama son zamanlarda, şov biraz daha ilginç bir hal almış gibi görünüyor. Sokak köşelerinde, billboardların arasında, hatta bazen bir sokak lambasına asılan pankartlarla karşılaşıyorsunuz. Peki, bu pankartlarda ne yazıyor dersiniz? "Lütfen yardım edin, Chanel çanta almak istiyorum!" gibi cümlelerle karşılaşmanız hiç de sıra dışı değil. Evet, yanlış duymadınız. Artık sokaklarda dolaşırken, pankartla para toplayan insanlarla karşılaşmanız mümkün. Bir zamanlar sokaklarda dilenenler vardı, şimdi ise pankartlarla dilenenlerin çağına girdik demek yanlış olmaz. 21. yüzyılın yeni yüzü olarak gösterilen bu pankartçılar, "Para ver, hayalimi gerçekleştireyim!" dercesine sokak köşelerinde poz veriyorlar. Kimi çocukluk hayallerini, kimi seyahat tutkusunu, kimi ise - şaşırtıcı bir şekilde çanta arzusunu dile getiriyor.
        """,
        "Funda KARAYEL"
    ),
    (
        """Bodrum 2024 yazına hazır mı? Altyapı yetersizliği, yolların bozukluğu ve her sezon artan fiyatlar gibi sorunlarına yaz gelmeden en üst sıradan susuzluk girdi... Bu yıl vatandaşlar, kabristan ziyaretlerinde susuzlukla karşılaşarak şok geçirdi. Susuz kabristan ziyareti olur mu demeyin çünkü oluyormuş, bunu bu bayram Bodrum'da öğrenmiş olduk. Utanmak mı, üzülmek mi, hangisi ağır basıyor, inanın bilmiyorum. Tek bilmek istediğim su yok ise bu yaz nasıl geçecek?!Son yıllarda yaşanan kuraklık ve artan su kullanımından kaynaklı seviyesi geçen yıllara göre çok düşük kalan Bodrum Mumcular Barajı'ndaki su tükenmişti, yine tükenecek. Peki çözüm? Öncelikle, su tüketimimizi bilinçli bir şekilde yönetmeliyiz. Duş sürelerini kısaltmak, muslukları sıkı sıkı kapatmak ve suluğa biraz su doldurmak gibi küçük önlemlerle bile büyük farklar yaratabiliriz. Su tasarruflu teknolojilere ve yenilenebilir su kaynaklarına yatırım yapmalıyız. """,
        "Funda KARAYEL"
    ),
    (
        """Ramazan Bayramı tatilinin dokuz güne uzamasıyla birlikte, herkes tatil planları için harekete geçti. Ancak seçim yapmak için olasılıklar o kadar çok ki; geleneksel tatil rotaları mı tercih edersiniz yoksa, sıra dışı ve yenilikçi deneyimler mi? Örneğin, bu hafta uzayda Michelin yıldızlı şef ile gastronomi deneyime özel 495 bin dolar öder misiniz sorusu gündeme geldi. Bu soru insanları düşündürmeye ve sıra dışı seyahat deneyimlerini keşfetmeye yönlendirdi. 2024 seyahat trendleri arasında uzay deneyiminden başka denizde yüzen karavanlar ve sessiz turlar gibi ilginç ve yenilikçi seçenekler de yer alıyor.
Teknolojinin gelişmesiyle seyahat trendlerinde son yıllarda gözle görülür bir değişim yaşanıyor. Artık sıradan tatil deneyimleri yerine insanlar, farklı ve ilgi çekici seçenekler arayışına girdiler. Ramazan bayramının 9 güne uzatılmasıyla 2024'ün yeni seyahat trendleri gündeme geldi. Bu bağlamda uzayda Michelin yıldızlı şef ile birlikte yemek deneyimi gibi teknolojinin etkisi sadece uzaya değil, karadan denize kadar uzanıyor. """,
        "Funda KARAYEL"
    ),(
        """Bir şirketin başarısı sadece ürünleriyle ölçülmez. Topluma sağladığı katkıyla özellikle de kadınları güçlendiren işbirlikleriyle ve eşitlik mücadelesine verdiği destekle de ölçülmeli. Geçtiğimiz günlerde Vodafone Sultanlar Ligi'nde haftanın dev derbisinde oynanan Fenerbahçe Opet ile VakıfBank maçı sonrası Vodafone Türkiye İcra Kurulu Başkan Yardımcısı Meltem Bakiler Şahin ile keyifli bir sohbet gerçekleştirdik. İşte, işini tutkuyla yapan bir kadın yöneticinin anlattıkları...
- Kadın sporunu uzun yıllardır destekliyorsunuz. Bu destek nasıl başladı, bugüne kadar neler yaptınız kısaca anlatır mısınız?
- Vodafone olarak, kadınların güçlenmesi konusunda güçlü bir vizyona sahibiz. Tüm dünyada kadınlar için en iyi işveren olmayı hedefliyoruz. Bir ülkenin sürdürülebilir kalkınmasının dijitalleşmeyle olduğu kadar toplumun yüzde 50'sini oluşturan kadınların ekonomik ve sosyal hayatta tam katılım göstermesiyle mümkün olduğuna inanıyoruz. Her alanda kadınların yanında duran bir şirketiz. Özellikle Vodafone Vakfı çatısı altında kadınların güçlenmesine yönelik çalışmalarımızı 13 yıldır aralıksız sürdürüyoruz. Sadece vakıf çalışmalarıyla değil sporda da kadınların... """,
        "Funda KARAYEL"
    ),(
        """Uzak diyarlarda yaşarken, kendi insanımızın samimi hikayelerini dinlemek gibisi var mıdır? Lacivert Dergi ve Yunus Emre Enstitüsü işbirliğiyle yurt dışı buluşmaları kapsamında düzenlenen Fransa buluşması, zorluklar karşısında pes etmenin bir seçenek olmadığını, çalışmanın ve inanmanın her zaman başarı getireceğini bir kez daha hatırlattı. 2023'te Macaristan, Kosova ve Almanya'da gerçekleşen etkinliğin 2024'te İtalya ve Malta'dan sonraki durağı Fransa oldu.Vakıfbank'ın sponsorluğunda Paris'te gerçekleşen söyleşilere komedyen, oyuncu Doğu Demirkol, Klinik Psikolog Beyhan Budak, Yazar Tarık Tufan ve Şef Çiğdem Seferoğlu katıldı. Yunus Emre Enstitüsü Başkanı Prof. Dr. Şeref Ateş'in de katıldığı program, yoğun ilgi gördü.
15 Şubat'ta Nimes'de gerçekleşen ikinci söyleşide ise yine Klinik Psikolog Beyhan Budak, Yazar Tarık Tufan, Daily Sabah Genel Yayın Yönetmeni İbrahim Altay ve Lacivert Dergi Genel Yayın... """,
        "Funda KARAYEL"
    ),(
        """Medya, günümüz dünyasında sadece bilgi aktarımı ve iletişim aracı olmanın ötesine geçti, uluslararası ilişkilerde de artık önemli bir rol oynuyor. Haberlerin sunuluş şekli, kullanılan dil ve görseller, ülkelerin imajlarını ve halkların birbirine bakış açılarını doğrudan etkiliyor. Medyanın dostluk köprüsü oluşturma potansiyeli oldukça yüksek. Örneğin, ortak bir insani kriz durumunda, medya yardımlaşma ve dayanışma duygularını teşvik ederek ülkeler arası yakınlaşmaya katkıda bulunabilir. Tüm bunları neden söylüyorum çünkü yanlış bilgi ve propaganda yayarak, ülkeler ve halklar arasında düşmanlık ve gerginlik de yaratılabilir.



Medyanın gücünün ne denli büyük olduğunu Tiran'da Türkiye-Arnavutluk Medya Buluşmaları'nda bir kez daha tartıştık. """,
        "Funda KARAYEL"
    ),(
        """Milli İstihbarat Teşkilatı'nın zengin tarihini keşfetmek için tasarlanan bir sergi düşünün, sadece bilgilerle değil, izleyenleri derinlere çeken bir deneyim vadediyor. Serginin ilk bölümü, teşkilatın kuruluş yıllarına bir bakış sunarak izleyicilere zamanın tozlu sayfalarında bir yolculuğa çıkarıyor. Duygusal bir bağ kurmak için, o dönemin önemli figürleri ve olaylarına odaklanarak, ziyaretçilere tarihle bütünleşme fırsatı sunuluyor. Ziyaretçiler, ajanların gözünden dünya olaylarına tanıklık ederken, karar almanın karmaşıklığını ve stratejilerin nasıl oluşturulduğunu anlama fırsatı buluyorlar.Serginin en etkileyici bölümlerinden biri, soğuk savaş dönemine odaklanan kısım. İstihbarat teşkilatının bu dönemdeki rolleri, casusluk operasyonları ve kriz anlarını detaylı bir şekilde açıklıyor. Ziyaretçiler, bu tarihi dönemde yaşanan gerilimli anları adeta hissederek deneyimliyor. Ancak sergi sadece geçmişi değil, aynı zamanda günümüz... """,
        "Funda KARAYEL"
    ),
    (
        """Günümüzde, psikopati terimi sıklıkla suç konseptli televizyon dizileri veya gerilim filmleriyle ilişkilendiriliyor olsa da, etrafımızda gizli psikopatlar da olabilir. Psikopatlar genellikle soğukkanlı, duygusuz ve empati eksikliği ile karakterize edilen kişilik özelliklerine sahiptirler. Ancak, bu kişilerle karşılaşmak ve onları tanımak o kadar da kolay olmayabilir. Belki iş arkadaşınız, belki de sevgiliniz... Belki de kendiniz... Peki soğukkanlı bir psikopat mısınız? Cevap tam olarak sizin elinizde olabilir.80 KİŞİ İNCELENDİ
Kevin Dutton'ın 'Olağan Psikopatlar' kitabını okuduğumdan beri etrafımızda gezen psikopatları araştırıyorum. Sokakta, kafede, sinemada her yerde yanımızdalar. Kanada'daki araştırmacılar, psikopatinin 'biyolojik kökenli' olup olmadığını belirlemek için klinik olarak psikiyatrik sorunları teşhis edilen gönüllülerin parmak uzunluklarını analiz etmiş. Yapılan bilimsel çalışma için araştırmacılar 80... """,
        "Funda KARAYEL"
    ),(
        """Sanatın evimizde ve hayatımızda önemli bir rol oynadığı konusunda hemfikiriz. Ancak, son zamanlarda sanat dünyasında ortaya çıkan ilginç bir trend, feng shui ile bağlantılı bir krizi gün yüzüne çıkardı.
Özellikle ev dekorasyonunda, tabloların seçimi ve konumu, insanların enerjilerini etkiliyor. Üzgün suratlı tabloların eve üzüntü ve sıkıntı getirdiği inancı, birçok insanın evinde sanat eserlerini gözden geçirmesine neden oldu. Yanlış duymadınız, Miami'de çok ünlü bir galerinin sahibiyle konuştuğumda bir feng shui krizi çıktı ki herkes evindeki eserleri değiştirmeye başladı diyor.Koleksiyonerlerin her zaman birbirini takip etmesi burada da kötü etki yapmış. Eserler geri iade edilmek isteniyor ya da müzayedede açık artırmaya çıkıyor. Büyük bir değişim var. Enerji akımcıları, sonbahar temalı tabloların yaprak dökümü sembolizmiyle ayrılık getirdiğine inanıyor. Bu ilginç yaklaşım, sanatın sadece estetik değil, aynı... """,
        "Funda KARAYEL"
    ),
    (
        """İyisiyle, kötüsüyle, aldığımız derslerle bir yılı daha geride bırakıyoruz... '2023'ü nasıl bilirdiniz?' diye kime sorsam, 'Merhum iyiydi ama çok üzdü' diye cevap veriyor. Sağlıklı ve dengeli bir psikolojiye sahip olmanın zor olduğu bir yıldı. İnişleri çoktu, çıkışları 2024'e bıraktı sanki. Winston Churchill, "Ne başarı bir sondur, ne de kaybetmek ölümcül; önemli olan devam etme cesaretidir" diye boşuna dememiş. İhtiyacımız olan şey şu an tam da bu, devam etme cesareti, her şeyi geride bırakıp devam edelim.Sahi siz nasıl bilirdiniz 2023'ü? Bu yılın ardında bıraktığı izleri, acıları ve umutları gelin hep birlikte değerlendirelim...
2023 yılı dünyada savaşlara, doğal afetlere sahne oldu, bu yüzden de herkes için zordu. 2023 yılının başlarında Türkiye, binlerce kişinin hayatını kaybettiği Kahramanmaraş depremleriyle sarsıldı. Felaketler zinciri, Afrika'da askeri darbeler ve Ukrayna-Rusya savaşı ile devam etti. İsrail-Hamas çatışması ise küresel bir çalkantıya... """,
        "Funda KARAYEL"
    ),(
        """Yeni medya sanat dünyası, gün geçtikçe büyüyen ve evrilen bir alan olarak karşımıza çıkıyor. Ancak, bu büyüme bir dizi tartışmayı da beraberinde getiriyor.
Miami sanat haftasında şahit olduğum bir olay son zamanlarda, birçok yeni medya sanatçısı, eserlerinin diğer sanatçılar tarafından acımasızca taklit edildiği iddiaları. Bazı sanatçılar, eserlerinin benzerlerinin, hatta aynılarının, başka sanatçılar tarafından kullanıldığını öne sürüyor ve bu durumun yaratıcılıklarına zarar verdiğini söylüyorlar. Aynı veriler kullanılarak yapay zekada üretilen işler, bir sanat eserinin hangi ölçüde orijinal sayılması gerektiği konusunda yeni medya sanat dünyasında belirsizlik yaratıyor.Konuştuğum bir yeni medya sanatçısı, sektörün en bilinen isimleri, kimse bu yolda başarıyla ilerlesin istemiyor, benim verilerimi kullanıyor, 'Benim tekniğim' diyor. Zaten hepsi aynı veriyi kullanıyor diyor. """,
        "Funda KARAYEL"
    ),
    (
        """Cumhuriyetimizin 100. yılında ülkemize yakışan bir prodüksiyon... Sahneyi görünce ilk anda şaşırmamak elde değil çünkü sahne yerine, dev bir çukurla karşılaşıyorsunuz. Sonra dekor, bu dev çukurdan iki katlı bir apartman yüksekliğinde yukarı doğru çıkıyor, tüm görkemiyle... Wolfgang Amadeus Mozart'ın bestelediği 'Don Giovanni Operası'ndan başkası değil bahsettiğim. Opera literatürünün başyapıtları arasındadır bu eser, Richard Wagner ise 'tüm operaların operası' olduğunu iddia ediyor.Lorenzo Da Ponte'nin kaleme aldığı librettosu ile kara komedi olarak nitelendirilebilen opera; ün yapmış, çapkın bir İspanyol asilzade olan Don Giovanni adındaki efsanevi karakterin maceralarını anlatıyor. 'Don Giovanni' rolünde izlediğimiz İstanbul Devlet Opera ve Balesi'nin yeni sanat yönetmeni ve müdürü Caner Akgün'ün performansı gerçekten görülmeye değer.
Mozart'ın bestelediği Don Giovanni'nin prömiyeri 29 Ekim 1787'de Prag'da Ulusal Tiyatro'da oynandı. 236 yıl sonra... """,
        "Funda KARAYEL"
    ),
    (
        """Gündem öyle yoğun ki..

Konu seçmekte zorlanıyoruz.. 

Bir yandan % 1 oy farkı ile Hakkari’de Belediye Başkanı seçilen DEM’li isimin, 2014’den bu yana süren davası.. Daha yeni, dün sonuçlanıp, 19.5 yıl ceza verilmesi.

Yapılan teröre destek şeklindeki somut suçlamalara cevap veremeyen DEM’lilerin, CHP’lilerin, “Milletin iradesine saldırı” yorumları yapması..

Sanki % 46 oy alan DEM’li yerine, % 45 oy alan AK Partili aday koltuğa oturtulmuş gibi algı oluşturulması.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Yüzünüze söylüyorum, soldan çarklılar, yalancısınız, sahtekarsınız..

Ulusalcılarınız da öyle, ateistleriniz de öyle..

Somutlaştırayım.

Kime söylüyorum, azınlıkta kalan dürüstlerini de kenara çekerek, açık açık anlatayım:


“Atatürkçü subaylara kumpasta yeni gelişme: Mahkemeden ‘garip karar’ ” başlığı atanlara söylüyorum.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Yüzyıllar öncesinde yaşanılan olayları yorumlayan bazı kişilerin çok kesin ifadeler kullanmasına hep şaşkınlık duymuşumdur.

Hele hele sosyal olayları, birden fazla sebebe bağlı gelişmeleri tartışırken, bazı arkadaşlar öyle kesin ifadeler kullanıyorlar ki, sanırsınız olayların birebir canlı canlı şahitleri.

Yüzyıllar öncesindeki olayları bir kenara bırakın.

Şunun şurasında 11 yıl önce yaşadığımız gezi isyanını doğru yorumlayabiliyor muyuz?

Birimiz “isyan” diyoruz, diğerimiz “sivil direniş”.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """27 Mayıs darbesinin, dün yıldönümü idi..

Sandıktan çıkan neticeyi, silah zoru ile by-pass eden darbeciler, bir de utanmadan, seçilmiş insanları astılar..

Yetinmediler.. Yaptıkları darbeyi, bu ülkenin hukuk fakültelerinin salonlarında hala ismi bulunan sözde profesörlere “meşrudur” fetvası verdirdiler..

Abartmıyorum. 

Çarpıtmıyorum. """,
        "Ali Karahasanoğlu",
    ),
    (
        """“Döviz alın”..

Maksat piyasayı etkilemek..

Maksat enflasyon rakamlarında yaptıkları hokkabazlığı, dövizde, diğer verilerde de, gücü nispetinde yapmak..

Birileri de Veysel Ulusoy’u dinlemiş.

Döviz almış.Hatta iddiaya göre, bir amcamız sotadaki altınlarını da satmış, döviz almış.. 

Ben böyle abartılı iddialara mesafeliyim ama, iddiaya göre bir de kredi çekmiş, çektiği kredi parası ile de döviz almış!

Veeee.

Seçim sonrasında dövizin patlamasını beklerken..

Kendileri patlamış.

Profesör Doktor Veysel Ulusoy patlamış.

Altınlarını da bozup döviz alan Veysel Ulusoy takipçisi amcamız, kızının iddiasına göre üzüntüden vefat etmiş.

Vefat eden Veysel Ulusoy takipçisi, muhtemelen borsanın kırdığı 11 binlik rekoru da göremeden, cenazesi kaldırıldı.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Ankara Barosu İnsan Hakları Merkezi üyesi avukat Alperen Demirdiş, uyuşturucu ticareti yapan birilerinin avukatlığını almış..

Hani Süleyman Soylu, “Bir uyuşturucu satıcısını gördükleri zaman, beni ne kadar kınarlarsa kınasınlar, ne kadar eleştirirlerse eleştirsinler, o uyuşturucu satıcısının ayağını kırmayan polis görevini yapmamış demektir. Benim ülkemin gencinin canına mal olacak, onu zehirleyecek ve aileleri huzursuz yapacak bir kişiye gereğini yerine getiren suçunu bana atsın” demişti de, laikçi takımı ayaklanmıştı ya..

Şimdi de, bir başka laikçi takım mensubu, bakmış ki, mahkeme heyetinde başörtülü başkan ve bir de üye var..

“Bu heyetten bizim alaverelerimize eyvallah eden çıkmaz.. Biz heyeti değiştirtelim” diye düşünmüş..


Ve hakimleri, başörtülü oldukları için reddetmiş.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Afyonkarahisar Belediye Başkanlığı’na seçilen CHP’li Burcu Köksal, mazbatasını alır almaz bir yaygara koparttı..

“Böcek var böcek.. Her yerde böcek.. Makam odamda.. Prizlerin altında.. Hatta asansörde..” 

Savcı önce bekledi ki, belediye başkanı suç duyurusunda bulunsun, böcekleri koyanları yargılattırsın..

Gelen giden yok..

Çünkü böcek möcek yok..

Oturduğu yerden, CHP’li başkan, tıynetleri gereği yalan uyduruyor.. İftira atıyor..CHP’li başkan suç duyurusunda bulunmayınca.

Bulduğunu öne sürdüğü böcekler ile ilgili belgeleri savcılığa sunmayınca..

Savcı, re’sen soruşturma başlatıyor ve o böcekleri CHP’li başkandan istiyor..

Böcekleri istiyor ki, böcekleri makam odasına koyanları bulsun, AK Partili falan demeden, cezalandırılmaları için davaları açsın..CHP’li Belediye Başkanı, “Böcekler yok. Ne yapacaktık, böcekleri yiyecek halimiz yok herhalde. Hem böcekler uzun süreli yaşayabilecek hayvanlar değildi” diyecek de.

O kadar şımarıklar çünkü.. O kadar kendilerini beğeniyorlar, milletle alay eden bir tıynete sahipler, çünkü..

CHP’li başkan, “Bu sefer pabuç galiba pahalı” diye düşünüp, dalga geçercesine bir cevap veremiyor....

“Böcekleri attık” diyor..

Soruyorlar, “Böcekleri kim buldu? Emniyet birimleri mi, bir şirket mi, kim buldu?”

“Savcı bey.. O gün burası hamam böceği kaynıyordu.. Her gelen böcekleri görüyordu.. Emniyet birimlerine ne gerek var. Odaya giren herkes böcekleri görüyordu..” diyecek de. CHP’li başkan, şımarıklığının pahalıya patlayacağını anlamış, diyemiyor..“Hazırda bulunan kişilerce bulundu” diyor..

Savcılık araştırma yapıyor.

Tam da o gün, Zeren Güvenlik ve Danışma Şirketi’nin iki elemanının belediye binasına geldiği tespit ediliyor..

Gelenler bulunup soruluyor: “Belediye binasına niçin gittiniz?”

Onlar da işi gırgıra vuracaklar da.

“Belediye iştiraki YÜNTAŞ’tan Kemal bey, ‘Çin’den misafirlerimiz geldi. Yanlarında canlı böcek getirmişler, pişirip yiyeceğiz, siz de gelsenize’ dediler.. Biz de gittik hep birlikte yedik. Böcek kalmadı. Kalsaydı size de yedirirdik” diye cevap verecekler de..

Karşılarında devletin resmi görevlileri..

Eski dönemin ağır aksak işleyen yargısı değil, “Böcek bulduk diyorsan, getir böceği” diye direten ve görevini hakkıyla yapan savcılık..

Ve sonuçta güvenlik şirketi gerçeği anlatıyor:

“Afyonkarahisar Belediye binasında güvenlik araması yapılması istendi. Binaya gittik. Yaklaşık 2 saat boyunca belediye başkanının makam odasında, toplantı odasında ve bir belediye başkan yardımcısının odasında elektronik ve teknik güvenlik kontrolü çalışması yaptık. (..) herhangi bir cihaz veya materyal bulamadık.”

Afyonkarahisar Belediye Başkanı tiyatrocu Burcu ne diyordu?

“Her yerde böcek var.. Prizlerin altında.. Makam odamda, koltuğumun tam karşısında..”

Bu arada, Zeren Güvenlik Danışmanlık şirketi işin ederini de anlatıyor:

Güvenlik araması karşılığında Yüntaş firmasına (Belediyenin şirketi) danışmanlık hizmeti bedeli olarak 66.000 TL’lik bir fatura kestik.”

Ooooh..

2 saatlik bir arama..

66 bin TL’lik fatura..

Ne diyordu CHP’liler?

“AK Parti’de israf var.”

Saadet Partisi’nin 80 yaşındaki genel başkanı da, “Ben kefilim” diyerek, CHP’nin yanlış işlere imza atmayacağını ilan ediyordu..

Ne yapılmış?

Yeni belediye başkanı olan Burcu hanım, “böcek” demiş..

Çağırmışlar danışmanlık şirketini..

İki saat arama yapmışlar.

Böcek bulamamışlar ama, belediyenin şirketinin 66 bin TL’sini alıp gitmişler..

Böcek olayında şimdi son durum nedir?

Böcek möcek mafiş..

Savcı istiyor, “bulduğunuz böcekleri verin” diye.

Aramayı yapan danışmanlık şirketi, “Böcek bulamadık ki..” diyor.

Belediye başkanı ise, “Böcekleri bulduğumuz gibi, çöpe attık” diyor..

Savcılık, böcekleri koyanları soruşturmak ve cezalandırmak için başlattığı soruşturmayı, “Böcek yok. Dolayısı ile böceği koyduğu ileri sürülen kişiler için de soruşturma yok. Ama, böcek bulduk diyenler için, iftira suçundan yeni bir dosya açılması için gereken işlemin yapılmasına” diyor..

Böcek dosyasını kapatıyor..

İftira dosyasını açmak üzere, gerekenler yapılmaya başlanıyor..

Bu vesile ile devlet, devletliğini hakkıyla gösteriyor..

“Böcek bulundu” iddiasının üzerine gidiyor..

Belediye başkanına soruyor, “hani böcek?”

Yok öyle, günlük siyasi tartışmalar içinde, “Ben böcek bulundu demedim. Ben böcek bulunma ihtimali var dedim. Ben AK Parti’yi suçlamadım. Makam odasında bulunmuş olabilir dedim” kıvırtmaları eşliğinde yeni iftiralara hazırlık yapmak..

Hakkıyla soruşturma yapılıyor..

Belediye başkanı kıvırtıp, “Böcek imha edildi” dese bile..

Savcılık o günlerde belediye binasında gerçekten böcek araması yapılıp yapılmadığını araştırıyor..

Bir şirketin binaya geldiğini tespit ediyor..

O şirketin yetkililerini çağırıp dinliyor..

Böcek bulunamadığı bilgisini zabta geçiriyor..

Tam ben “Savcılık işini yarım bırakmış. O şirket bu işi herhalde bedavaya yapmamıştır. Parası niye sorulmamış” diyecek iken.

Bir de bakıyorum, savcılık o konuda da şirket yetkilisinin beyanını almış: “66 bin TL fatura kestik.”  

Haydi bakalım şimdi..

Yalancılık derseniz CHP’li Başkan Burcu Köksal’da.. 

Dava açmaya kalkmasın.. ‘Böcek bulduk” dediği ve ispat edemediği için yalanı tescillendi..

Sahtekarlık derseniz, CHP’li Başkan Burcu Köksal’da.. 

Dava açmaya kalkmasın, rezil olur.. “Orda bulunanlar böcek araması yaptı” derken. 66 bin TL fatura ödemesi yapıp, özel olarak bir şirkete arama yaptırdıkları ortaya çıkıyor..

Korkaklık derseniz CHP’li Başkan Burcu Köksal’da.. 

Odasında böcek olabilir endişesi ile ilerde yapacağı üçkağıtçılıkların dışardan haberi alınmasın diye, şimdiden böcek taraması yaptırtıyor..

Müsriflik derseniz CHP’li Başkan Burcu Köksal’da.. 

Sadece böcek korkusu sebebi ile iki saatlik aramaya, 66 bin TL ödettiriyor..

Kendi cebinden de değil, belediyenin şirketinden..

Haydi bakalım, CHP’liler çıksın, bu başkanlarını savunsunlar.

Savunabiliyorlarsa.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Türkiye genelinde, birçok büyükşehir ve ilin belediyesi, artık CHP’li isimlerin elinde..

Özellikle emeklilerle ilgili, EYT sorununun çözümü sonrasında maaşların düşük kaldığı algısı, AK Partili seçmenin sandığa gitmemesini, CHP’li adayların kazanması sonucunu doğurdu.

Mazeret değil, gerçek bu!

CHP’li isimlerin önceki dönemlerdeki başkanları ne idi ki, şimdi yenileri ne olsun?

İstanbul Büyükşehir Belediye Başkanı Ekrem İmamoğlu, deste deste avroları, balya balya dolarları izah etti mi ki, şimdi yeni başkanlar da, benzerlerine imza atmasın..Ekrem İmamoğlu, aile şirketi gerçekte ciddi bir faaliyeti olmadığı halde, kendisi de son 5 yıldır zaten belediye başkanı olması hasebi ile ticaret yapmamış, özel iş kotarmamış olması gerekmesine rağmen.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Kariye Camii dün açıldı..

Milli Gazete “Hah hah ha.. Açılışına bir gün kala restorasyona alındı” diye manşet atıp, “Batılılara söz verildi, açılmayacak” imasında bulunuyordu.

Dün Kariye Camii’nde ezanlar okunmaya başlandı. Namaz kılınmaya başlandı..

Saadet Partililerin “şikayetçi değiliz” dedikleri 28 Şubat davasında, yüzbinlerce başörtülü kızın üniversitede eğitim hakkını gasbeden, meslek liselilerin üniversiteye girişte katsayı zulmü ile puanlarını çalan Kemal Gürüz’ün duruşması vardı..Dünkü duruşmaya Namık Kemal Zeybek geldi, ahlaksızca yalanlar uydurup, “Refah Partisi iktidara gelince birdenbire yer altında yaşayan örgütler ortaya çıktılar” iftirası attı. DYP’den ANAP’a, MHP’den BBP’ye ve ATA’ya girmediği parti bırakmayan bu müfteri adam, Saadet Partililerin şikayetçi olmadığı 28 Şubatçıları aklayıp, “O dönemin şartlarında eğer bir darbe düşüncesi olsaydı yapılırdı. Bunu kimse engelleyemezdi”... """,
        "Ali Karahasanoğlu",
    ),
    (
        """Anayasa Mahkemesi’nin iki ayrı kararı var.

Birisinde Taksim’de mitingin engellenmesi için açılan davada, “Gerekçe tatmin edici.. Hak ihlali yok” diyor..

Diğerinde “Gerekçe yetersiz. Hak ihlali var” diyor.

Ama iki karardan birisini alıp, milletin gözüne sokanlar, “Anayasa Mahkemesi kararına rağmen, Taksim’de mitingimiz engelleniyor” diyorlar..

DİSK öyle diyor.. Tabipler Birliği öyle diyor. TMMOB öyle diyor.  """,
        "Ali Karahasanoğlu",
    ),
    (
        """Milli Eğitim Bakanlığı’nın müfredat değişikliği konuşulurken..

“Yusuf Tekin çok birikimli bir isim, müfredat değişikliğini yaparsa o yapar” diye övgüler dizildiği bir dönemde, özellikle biyoloji alanında müfredat değişikliğinden geri adım atıldığı iddiaları tartışılıyor iken.. 

Özellikle de Marksist kafalı geri zekalılar, sanki yaratılış inancı; ilk insan Hazreti Adem’den bu yana, İslam inancının temel prensibi değilmiş de, 1950’lerde doğan ve televizyonda dansöz oynatmasıyla meşhur Adnan Oktar ile dünya gündemine girmiş gibi, “Adnancılar kaybetti” bakış açısını topluma empoze etmeye çalışırken..

Diğer taraftan ise Din Kültürü ve Ahlak Bilgisi müfredatında, savunma sanayii ile ilgili yapılan çalışmaların ayrıntılarının bulunmasını, “savunma ile dinin ne alakası var” diye yorumlayan ateist kafalar itirazlarını, “dinin cihattan ibaret olduğu” temel bilgisinden mahrum olarak yapmayı sürdürürler iken.. Aynı zamanda da biyolojiden din kültürüne, matematikten... """,
        "Ali Karahasanoğlu",
    ),
    (
        """ CHP’li Burcu Köksal, Afyonkarahisar Belediye Başkanı seçildi..Her ne kadar Ekrem İmamoğlu, seçim öncesinde, “Kendisine bir başka parti bulsun” diye tehdit etmişse de..“DEM’lileri belediye binasına...CHP’li Burcu Köksal, Afyonkarahisar Belediye Başkanı seçildi..

Her ne kadar Ekrem İmamoğlu, seçim öncesinde, “Kendisine bir başka parti bulsun” diye tehdit etmişse de..“DEM’lileri belediye binasına almayacağım” diyerek, Afyon’da Kürt düşmanlığı ile seçim kazanan CHP..

İstanbul’da ise PKK sempatizanlığı ile seçim kazanmış ise de..

İstanbul’da seçim kazanmak için, Esenyurt’ta PKK ile ilgili “terör örgütüdür” diyemeyen bir ismi aday göstermiş ise de.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Marksistler dökülüyor. Kemalistler dökülüyor.. Ama CHP kazanıyor..

Nasıl bir iş bu?

Marksistler dökülüyor dedim..

Ne yapıyorlar? Faiz onlarda da  emekçinin sömürülme aracıdır..

İslam dininde de faiz haramdır..Tayyip Erdoğan “Nas var” dediğinde.

“Bizim de Marks öyle emrediyor” diyerek..

Küçük bir destek çıktılar mı? Hayır..

Kapitalistlerle birlikte, “Saldır Co” diyerek, hoyratça algı operasyonlarına imza attılar..Kemalist ve Marksistlerin tükenmişliklerini, hangi başlıkları ile size aktarayım..

Birkaç örnek ile yetineceğim..

Dünkü Cumhuriyet gazetesinde bir ilan..

İlker Başbuğ, Tele1 televizyonunda, ayda bir programa çıkacakmış.

İlker Başbuğ kim? 

Bir cümle ile: “Terör örgütü PKK’nın 10 bin mensubunu şehid ettiği Türk Silahlı Kuvvetleri’nde en tepe noktaya kadar çıkmış bir subay!”40 bin sivil insanı da buna ekleyebiliriz.. Gün olmuş karakollarımız, PKK’lı teröristler tarafından basılmış, 10 saat, 20 saat yardım bile götürememişiz..

Gün olmuş, bir ilimizden bir ilimize sivil olarak götürülen erlerimiz, otobüsten indirilip, PKK’lı teröristler tarafından kurşuna dizilerek şehid edilmiş..   

Ara ara, bir iş kazası olduğunda, “Japonya’da olsaydı, yönetici intihar ederdi” diyorlar ya..

İlker Başbuğ’un emri altındaki insanların şehadet noktasındaki son nefesleri üzerine, belki kendisi de bin defa intihar etmesi gerekirdi..

İntihar dinimizde haram.

Biz intihar etmesini beklemeyelim..

Ama, birazcık vicdan.. 

Binlerce askerinizin şehid edilme emrini veren PKK’nın başındaki Öcalan için, “Siyasi mahkum” diyen bir hokkabazın televizyonuna da çıkmayın yani..

Evet, Merdan Yanardağ, bir yıl kadar önce, Abdullah Öcalan için, “Siyasi mahkum” nitelemesinde bulunmuştu..

“Çok uzun süre cezaevinde yattı” demişti..

“Tecrit altında” olduğunu iddia etmişti. Tele 1’in bu yöneticisi tutuklanmıştı.. 

Şimdi, bu ülkede Genelkurmay Başkanlığı yapmış, halen de bizim vergilerimizle masrafları karşılanan Fenerbahçe Orduevi’nde korunaklı olarak kalan İlker Başbuğ, terör örgütünün liderini cezaevinden çıkarmak için çırpınan Merdan Yanardağ’ın televizyonuna çıkıp, onu meşrulaştırıyor..

Kemalist İlker Başbuğ’un geldiği nokta bu. Onun için “Kemalistler iflas etti” diyorum..

Ve dönüyorum.. İlker Başbuğ’un Genelkurmay Başkanlığı koltuğuna kadar çıktığı TSK’da, vatani görevini yapan 20 yaşındaki gencecik fidanlara kurşun sıkıp öldüren “Marksistler” için yaptığım  “iflas ettiler” nitelememe..

Onlar niye iflas ettiler?

Düne kadar hedef gösterdikleri, yanındaki erleri bile şehid ettikleri bir kişiyi, televizyonlarına çıkarıyorlar..

İşe bakın.. Askerine kurşun sıkıyorlar.. Üstündeki subay ile televizyonda oturup sohbet ediyorlar..

Riyakârlığa şaşıyorum..

İki taraf açısından da şaşıyorum..

“Birbirinizin yüzlerine, nasıl bakabiliyorsunuz” diyorum.. 

Biriniz diğerinize “terörist”, diğeriniz berikine “cuntacı katil” derken..

“Şimdi nasıl kanka oldunuz, ne uğruna birliktesiniz?” diyor, hayretimi tekrarlıyorum..

Marksistler, Kemalistler sadece bu noktada mı dökülüyorlar?

Her alanda dökülüyorlar.

Metin Akpınar’ın gayrimeşru çocuğu tescillenmişti.. Şaşırdık mı?

Kemalist, laikçi birisinden ne bekleyebilirdik ki?

Ama ardından.. Bir de Metin Akpınar’ın yakın arkadaşı Uğur Dündar’ın da aynı kadınla ilişkisi ortaya atılınca..

“İşte Kemalistlerin mide bulandıran sapıklıkları” dedik..

Ama daha göreceğimiz ne sapıklıklar, ne rezillikler varmış!

Sözcü tv’de bir program.. Konuklar; emekli general başörtü karşıtı Ahmet Yavuz, Ceza hukuku profesörü Hasan Sınar, Sözcü yazarı Rahmi Turan’ın oğlu Murat Muratoğlu ve araştırmacı gazeteci(!) dindar karşıtı Uğur Dündar.. 

Moderatör kim? 

“Namaz kılıyorsunuz ama asıl ahlaklı olmanız lazım/Namaz 5 vakit, ahlak 24 saat farz” sözleri ile dindarları köşeye sıkıştırdığını zanneden Koç Lisesi/Koç Üniversitesi mezunu Ece Üner..

Kurban olduğum Yaratanımız, öyle bir şamar indirdi ki bu Ece’nin suratına, tam da “24 saat ahlaklı olma”nın farziyetini (o alay ederek söylüyordu ama), attığı kahkahaları izleyen insanların acıklı gülümsemeleri ile sabıkasına kaydedildi....

Gayri meşru çocuğu üzerinden Uğur Dündar savunma yapıyor.. İlk açıklaması, “Adli Tıp raporu geldi. Benim çocuğum değil” idi..

Sorduk, “Peki o kadın ile ilişkin oldu mu?” 

Unutulur umudu ile bir gün sessiz kaldı. Sonra Ece Üner’in belirttiğimiz programında şu rezil sözleri sarfetti:

“O dönem ben de bekar adamım, normal sağlıklı cinsel bir yaşamım var. Bunun konuşulacak başka bir tarafı yok ki!”

Yani, “önüme gelen kadınla yattım, onları istismar ettim” diyor vicdansız adam.

Tam bu sırada Ece Üner’de bir kahkaha.. Emekli generalde, ceza hukukçusunda, Sözcü yazarında, bıyık altı gülümsemeler.

Devam ediyor rezaleti itiraf eden Uğur Dündar:

“85 ve 86 yıllarında yediğimiz içtiğimiz ayrı gitmeyen çok samimi bir arkadaşım var, hala konuşuyoruz. O arkadaşıma sordum, dedim ki kardeşim hadi ben diyelim ki milyonda bir ihtimal -unutmam ya- beraberlik yaşadığım bir kadını unutmam da ‘Sen bu isimde bir kadını hatırlıyor musun?’ dedim. ‘Asla ya’ dedi. Çünkü biz çapkınlıklarımızı, her şeyimizi anlatırdık.”

Kadını bir meta aracı olarak gören Kemalistler, Marksistler.. Kadını ile erkeği ile.. Generali ile hukukçusu ile.. Gazetecisi ile..

İşte tel tel dökülüyorlar..

Bir de sporcularından örnek vereyim.

Fenerbahçe Başkanı Ali Koç, futbol takımı Riyad’a da gitmişken, “Atatürk yok ise, biz de yokuz” deyip, oynanacak maçta Atatürk silüetli forma olduğu halde, “ısınma turuna da Atatürklü forma ile çıkacağım” diye dayatmış ve “önceden verilen forma örneğinde yok” denilince, maça çıkmadan geri dönmüştü.. 

Şimdi aynı Ali Koç, Fenerbahçe’yi Yunan takımının karşısına, hem de Türkiye’de Atatürksüz çıkardı..

Ve.. “Türk hakemler bilgisiz, beceriksiz, taraf tutuyorlar” diyerek, “VAR” hakemlerini yurtdışından getirttiren Ali Koç.. Dört yabancı hakem eşliğinde, önceki akşam Yunan takımına elendi.

Saadet Parti Genel Başkanı Temel Karamollaoğlu, 2018’de başkan koltuğuna oturan Ali Koç için, “dip dalgası” diyordu..

O dalga, altıncı yılında bir şampiyonluk göremedi ama, Olimpiakos’a, yabancı hakemlerin onayı ile elendi!

Kemalistler dökülüyor.. Marksistler iflas ediyor..

Bu arada soru şu: “Kemalistler, Marksistler tel tel dökülürken, bunların oy verdikleri CHP, mahalli seçimleri nasıl kazandı?! """,
        "Ali Karahasanoğlu",
    ),
    (
        """İsrail’in altı aydır Gazze’ye yaptığı saldırıların bir küçük örneğini İran, İsrail’e yapar gibi oldu.

Kıyametler koptu.Amerika’sından başladık, İngiltere’ye kadar hepsinden kınamalar.

Rusya hariç olmak üzere bütün gâvurlar, İran’ı telin ettiler. """,
        "Ali Karahasanoğlu",
    ),
    (
        """CHP ve teşkilatı bile, gerçeği bildiği için, fazla bağırmıyorlarken..

Sevinç gösterileri eşliğinde “ezdik geçtik” tezahüratları yapmıyorken..

Yapmaya cesaret edemiyorlarken..

Bizim mahalledeki bazı arkadaşlar, seçim sonuçlarının yüzde 90 oranında sebebi de ayan beyan belli iken..


“Beni aday göstermediniz, benim adamımı aday göstermediniz onun için böyle oldu” ile başlayacaklar da.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """İlk defa Ahmet Davutoğlu'ndan duydum..
Artık klasikleşen cümle: “28 Şubat’ta bile bunu yapmaya cesaret edemezlerdi.“
Akit TV'ye konuk etmiştik kendisini.
Yönetiminde olduğu Şehir Üniversitesi’ne, başbakanlık makamını kullanarak, zamanın parasıyla iki katrilyonluk devlet arsasını bedava veren Ahmet Davutoğlu, bir yandan da Tayyip Erdoğan Erdoğan’a kara çalmaktan geri durmuyordu: “Haberi olmadan hiçbir arsa kimseye verilmez.”
Ben Tayyip Erdoğan’ın değil iki katrilyonluk, onda biri değerindeki bir devlet arsasını bile, bedavaya ne bir kişiye ne bir vakfa verdiğini duymamıştım
Ama etik siyaset iddiasıyla ortaya çıkan Ahmet Davutoğlu, hem de yönetiminde olduğu bir üniversiteye devletin iki katrilyonluk arsasını bir çırpıda tapu devri ile verivermişti.
Danıştay “böyle rezillik olmaz” diye karar verince de, Davutoğlu da o meşhur... """,
        "Ali Karahasanoğlu",
    ),
    (
        """Sözcü gazetesinden, yakın süreç içinde birkaç başlık aktarayım..

“Devlet ilkokulunda bunu da yaptılar: Okulda namaz”

“Tertemiz alnını Arap’a öptürttü”

“Dindar nesil için dindar öğretmen atadılar! Türkiye Cumhuriyeti Devleti’nde sonunda bunu da gördük” (Feraceli öğretmen manşet)

YÖK Başkanı’nın Anayasa’yı çiğneyip türbana kapıyı aralamasına CHP tepkili: Yakında ünversitelere türbansız girilmeyecek”

“İktidar, 10 yaşındaki çocukları okulda türbana soktu: Hedef 2023 Bunlar bu gidişler doğmamış çocuğa bile türban takacak!” (Anne karnındaki bebeğe görselde türban takılmış manşet) """,
        "Ali Karahasanoğlu",
    ),
    (
        """Muhteremin ilk icraatı, Beylikdüzü’ndeki çaycısını, İBB’ye getirmek olmuştu..

Ama o kadar büyük bir şovmendi ki..

Beylikdüzü’ndeki çaycısını, iki gün içinde Saraçhane’deki İBB’ye getirdiği değil, o çaycısı ile kameralar huzurunda yaptığı söyleşi ile kendisini gündeme oturtmuştu.

Sosyal medyada ve CHP’ye yakın yayın organlarında yayınlanan görüntülerde İmamoğlu.. 

Şimdilerde Beylikdüzü’nden yanında getirdiği Fatih Keleş’i İBB Spor Klübü başkanı yapan ve balya balya avroları İBB Spor Klübü çantaları da taşıtan, deste deste dolarların sayımına eşlik eden yine Beylikdüzü’nden gelme Tuncay Yılmaz..


Yine Beylikdüzü’nden, TBMM’ye gönderdiği eski ilçe başkanı Turan Taşkın Özer.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Dindar insanları sürekli suçlarlar.. Bilime gerekli önemi vermeyen, bilim insanlarına gerekli değeri vermeyen insanlar gibi damgalar, mahkum ederler..

Buyrun 31 Mart seçimlerinde, bilim insanlarına değer verme konusunda, solcusuna-sağcısına, dindarına-ateistine, komünistine-faşistine hodri meydan diyelim..

Adaylara baktığınızda, iki isim var..

Biri Murat Kurum. Diğeri Ekrem İmamoğlu..

Bilim insanlarına değer verilmesini isteyenler, bu iki isimden hangisini tercih etmeliler.. """,
        "Ali Karahasanoğlu",
    ),
    (
        """Şu CHP’lilere şaşıyorum.

Ne kadar rahat yalan söyleyebiliyorlar.

Çıkan yalanların üzerine, ne kadar pişkince yenilerini tekrarlayabiliyorlar..

Muhafazakar partilerden birisinden genel başkan demiyorum. Herhangi bir görevdeki kişi şu netlikte bir yalanlama yaşasın, mümkün değil, bir gün sonra canlı yayına çıksın, bir ay evinden çıkmaz..

Ama CHP’nin genel başkanı Özgür Özel, “Afyon Belediye Başkan adayımız beni aradı. Ne olur genel başkanım, dilim sürçtü, sen durumu düzelt dedi” şeklinde beyanı, bir saat sonra ilgili kişi tarafından yalanlandı..

Özgür Özel gayet pişkin… """,
        "Ali Karahasanoğlu",
    ),
]

# Morphology nesnesini oluştur
morphology = TurkishMorphology.create_with_defaults()

# Veriyi hazırla
texts, authors = prepare_data(corner_texts, morphology)

# Etiketleri sayısal değerlere çevir
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(authors)

# Metinleri vektörize et ve model oluştur
vectorizer = TfidfVectorizer(stop_words=get_stop_words("turkish"))
X = vectorizer.fit_transform(texts)
y = encoded_labels

# Eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Lojistik regresyon modeli ile eğitim yap
model = LogisticRegression()
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yap ve doğruluğu kontrol et
accuracy = model.score(X_test, y_test)
print(f"Model doğruluğu: {accuracy * 100:.2f}%")

# Hedef metni ön işleme tabi tut ve tahmin yap
target_text = """Bu tür çalışmaları iki açıdan değerlendiririm. Birincisi seyrettiğimiz ya da seyredeceğimiz filmler konusunda bilgi sahibi oluruz, diğer açıdan da farklı kişilerin çalışmalarını bir özel sayıda buluruz.

HECE Dergisi’nin çıkardığı 2 ciltlik Türk Sineması kitaplığınızda yerini almalıdır.

Yıllar önce çevrilen filmler televizyonlarda gösterilmektedir, onları seyrederken bu özel sayıya başvurabilirsiniz.

Sunuş özel sayı ile ilgili açıklamayı içeriyor:

“Türkiye’de sinemaya duyulan ilgi bugün neredeyse 60’lı 70’li yılları yakaladı. Ancak bu ilginin Türk sinemasından çok yabancı sinemaya, Doğu’nun ve Batı’nın eski ve yeni sinemalarına ve özümsenmeyen teorik metinlere doğru bir temayülü olduğunu biliyoruz. Türk kültürü ve medeniyetinin sanatla ve ilimle yoğrulmasını isteyen herkes gibi biz de Türk sinemasının gelişmesini, dünya çapında bir marka halini almasını, bu toprakların özgün sesinin, söyleminin sözcüsü olmasını, insanlığa miras kalacak filmlerle büyümesini arzu ederiz. Ancak bu filmler vücut bulurken ve seyircisi ile buluşurken, entelektüel çevrelere büyük bir rol düşmekte. Özellikle endüstri olmaktan öte bir sanat olarak sinema üzerine düşünen ve yazan herkesin yönünü en az yabancı sinema kadar ve mutlaka daha fazla Türk sinemasına çevirmesini isteriz. Yaklaşıp bakmak, üzerinde düşünüp yazıp tartışmak, sağlıklı ve tutarlı bir inceleme ve eleştiri ortamı oluşturmak, ‘sağa’ ‘sol’a çekiştirmeden, benimki sizinki demeden dikkatimizi Türk sinemasına vermek istedik."""



target_text_preprocessed = preprocess_text(target_text, morphology)
target_vector = vectorizer.transform([target_text_preprocessed])
predicted_label = model.predict(target_vector)[0]
predicted_author = label_encoder.inverse_transform([predicted_label])[0]

print(f"Test edilen köşe yazısı, {predicted_author} tarafından yazılmış olabilir.")


# Kelimelerin türlerini ve frekanslarını yazdırma
def write_word_frequencies(
    texts: List[str], morphology: TurkishMorphology, output_file: str
):
    all_tokens = []
    for text in texts:
        analyzed_tokens = analyze_text(text, morphology)
        all_tokens.extend(analyzed_tokens)

    token_counter = Counter(all_tokens)

    with open(output_file, "w", encoding="utf-8") as file:
        for (lemma, pos), frequency in token_counter.items():
            file.write(f"{lemma}\t{pos}\t{frequency}\n")


# Frekans dosyasını oluşturma
output_file = "kelime_frekanslari.txt"
write_word_frequencies(texts, morphology, output_file)
print(f"Kelime frekansları {output_file} dosyasına yazdırıldı.")