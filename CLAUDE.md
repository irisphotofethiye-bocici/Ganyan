# CLAUDE.md — At Yarışı Analiz Sistemi (Ganyan Fork)

Bu dosya, Claude Code'un bu repoda çalışırken HER OTURUMDA uyması gereken
kurallardır. Kurallar tartışılmaz. Bir görev bu kurallarla çelişiyorsa, görevi
yapma — dur ve kullanıcıya sor.

Bu proje **fatihbozdag/Ganyan** repo'sunun fork'udur. Sıfırdan yazılmıyor;
mevcut çalışan kod temel alınır, üzerine kurallar ve ajan katmanı eklenir.

---

## 0. EN ÖNEMLİ KURAL — SIFIR HALÜSİNASYON

Bu sistem para kararına bağlanabilecek sayılar üretir. Yanlış ama
"doğru görünen" bir sayı, hiç sayı olmamasından daha tehlikelidir.
Aşağıdaki ilkeler her şeyin üstündedir:

1. **LLM (sen ve Claude Code) HİÇBİR sayı üretmez.**
   Olasılık, oran, Kelly payı, ROI, feature değeri, istatistik — hepsi
   deterministik Python kodundan gelir. Bir sayıyı "hesaplayıp" yazma;
   o sayıyı üreten kodu yaz ve kodu çalıştır.

2. **Bilmiyorsan "bilmiyorum" de.** Bir veri alanının ne olduğundan,
   bir TJK kısaltmasının anlamından, bir formülün doğruluğundan emin
   değilsen TAHMİN ETME. Dosyadan oku, kullanıcıya sor, ya da "doğrulanmadı"
   olarak işaretle. Uydurulmuş bir alan eşleştirmesi tüm modeli bozar.

3. **Her sayısal çıktı kaynağına kadar izlenebilir olmalı.** Bir tahmin
   üretildiğinde hangi model versiyonu, hangi feature seti, hangi veri
   tarihiyle üretildiği loglanır (provenance). "Nereden geldi bu?" sorusu
   her zaman cevaplanabilir olmalı.

4. **Şüphedeyken sus.** Eksik veya şüpheli veriyle tahmin üretmektense
   o yarışı "değerlendirilmedi" olarak atlamak doğrudur. Sistem sessiz
   kalmayı bilmeli.

---

## 1. MİMARİ SINIRLAR (aşılmaz)

```
VERİ KATMANI    Python scraper (Ganyan'ın scraper/'ı) — TJK açık sayfaları
                ↓ (yapılandırılmış veri, PostgreSQL)
BEYİN KATMANI   Python — features + LightGBM + Bayesian + Harville (saf kod)
                ↓ (predictions, provenance damgalı)
AJAN KATMANI    Orkestratör + Doğrulama + Rapor (LLM, ama sayı üretmez)
                ↓
```

- Veri katmanı ile beyin katmanı arasındaki sınır aşılmaz. Scraper model
  bilmez, model scraper bilmez. İkisi sadece DB üzerinden konuşur.
- Yeni bir bağımlılık eklemeden önce sor. Mevcut Ganyan stack'i kullan:
  Python 3.12+, PostgreSQL, SQLAlchemy 2.0, Alembic, uv, LightGBM, PyMC,
  Flask+HTMX, Typer.

---

## 2. VERİ KAYNAĞI — kesinleşmiş kararlar

- **Kaynak:** TJK'nın HERKESE AÇIK web sayfaları (tjk.org), Ganyan'ın
  `scraper/` modülüyle. Tarih parametreli erişim mevcut.
- **Anahtar yok:** `vhs.tjk.org` iç API'sine ve `X-Auth` anahtarına
  GİRİLMEZ. bilaleren/tjk-api yaklaşımı (anahtar gerektiren) bu projede
  KULLANILMAZ. Meşru, anahtarsız açık-sayfa kazıma esastır.
- **Kibar kazıma:** Saniyede en fazla 1 istek. Rate-limit/ban riskine
  karşı bekleme ve retry zorunlu. Agresif paralel kazıma YASAK.
- **Statik vs güncel ayrımı:**
  - Statik (geçmiş): bir kez backfill edilir, DB'de yaşar, tekrar çekilmez.
  - Güncel (günlük): sadece o günün programı + sonucu eklenir (artımlı).
- **Yasal sınır:** Bu KİŞİSEL ANALİZ ARACIDIR. Ticari ürün/servis değildir.
  Ürünleştirme kullanıcının ayrıca hukuki danışmanlık alacağı bir karardır;
  kod bu sınırı varsayar.

---

## 3. KONUMLANDIRMA — bu bir "+EV bahis botu" DEĞİLDİR

Ganyan'ın 1.841 canlı pick'lik defteri Türkiye havuzunda **pozitif edge
olmadığını** kanıtladı (tüm stratejiler takeout tabanında, ROI negatif).
Bu yüzden:

- Sistemin hedefi "kâr maksimize etmek" DEĞİL. Hedef: **top-1 isabet
  doğruluğu** + "hangi yarış değerli/belirsiz, hangisi atlanmalı" filtresi.
- Birincil metrik **top-1 hit oranı**, ROI değil (ROI havuz dinamiklerini
  yansıtır, model kalitesini değil).
- "+EV bulduk, şuna oyna" tarzı kesin tavsiye üretme. "Model şu olasılığı
  veriyor, karar senin" çerçevesi esastır.
- Gerçek para modu KOD SEVİYESİNDE KİLİTLİDİR (bkz. §7).

---

## 4. LEAKAGE — bir numaralı teknik tuzak

TJK sonuç verisi (Ganyan `ResultsTransformer`) yarış ÖNCESİ ve SONRASI
bilgiyi aynı objede taşır. Bunları karıştırmak backtest'i şişirir,
canlıda çökertir. KURAL:

**Yalnızca yarış öncesi bilinen alanlar feature olabilir:**
- İzin verilenler: kilo, fazla kilo, kilo indirimi, jokey, antrenör,
  sahip, yetiştirici, pedigree (baba/anne/3 kuşak), KGS (koşmama gün
  sayısı), handikap, son6/son20 (ÖNCEKİ yarışlardan), ekipman (TAKI),
  start kapısı, AGF (yayınlandıysa), pist/zemin/hava (yarış öncesi tahmin).

**Şunlar SADECE hedef (target), ASLA feature değildir:**
- SONUC (varış sırası), DERECE (bitiş zamanı), GANYAN (gerçekleşen oran),
  FARK (kazanana fark), SON800 (o yarışın pace'i), foto-finiş, video.

- Her yeni feature için `shift(1)` / zamansal pencere kontrolü zorunlu:
  bir atın "son 5 form"u BUGÜNÜ içermez, sadece öncesini.
- Ganyan'ın `tests/` altındaki leakage smoke testlerini KORU ve her
  feature eklendiğinde çalıştır. Test yoksa yaz.
- Şüpheli bir alanın hangi tarafta olduğundan emin değilsen: feature
  yapma, kullanıcıya sor.

---

## 5. HESAPLAMA — sayılar nasıl üretilir

- Feature üretimi pandas/numpy ile, vektörize (satır-satır döngü değil).
- Model: Ganyan'ın LightGBM LambdaRank ranker + Hierarchical Bayesian
  Plackett-Luce (skip-gate) yapısı korunur.
- Olasılıklar yarış içinde normalize edilir (toplam = 1).
- Kombinasyon (üçlü/ikili) = Harville joint probabilities (kod), LLM değil.
- Kalibrasyon her retrain'de kontrol edilir (model %20 diyorsa gerçekten
  ~%20 mi kazanıyor). Kalibrasyonsuz olasılıkla stake hesabı yapılmaz.
- Manuel veri girişi varsa (oran vb.): iki otomatik kontrol zorunlu —
  (a) makul aralık (oran 1.0–200 dışı = red), (b) implied probability
  toplamı ≈ 1 + kesinti; bant dışıysa "giriş hatası olası" uyarısı.

---

## 6. AJAN KATMANI — roller ve yetki sınırları

Ajanlar serbest sohbet etmez. Sabit sıra (DAG): Veri → Doğrulama →
Model(kod) → Kapı(kod) → Rapor. Her ajanın yetkisi sınırlıdır.

| Ajan | Yapabilir | YAPAMAZ |
|---|---|---|
| Orkestratör (Şef) | DAG yürüt, state oku/yaz, hata yönet, eskalasyon | hesap, sayı, kapı atlama, canlı kilit açma |
| Veri Ajanı | scraper çalıştır, parse hatası teşhisi, "eksik" raporla | eksik veriyi uydur, kendini onayla |
| Doğrulama (Critic) | ham veri tutarlılık yargısı, PASS/FAIL | veriyi düzelt, "muhtemelen doğru" deyip geçir |
| Rapor Ajanı | anlatı yaz, SHAP yorumla, "doğrulanmadı" damgala | sayı üret/yuvarla/düzelt, claims dışı iddia |

- **Critic bağımsızdır:** Veri Ajanı'nın gerekçelerini GÖRMEZ, sadece ham
  çıktıyı görür. (Aynı bağlamı paylaşan ajan kendi hatasını onaylar.)
- **Rapor şablon-doldurma modunda çalışır:** anlatı serbest, ama sayı
  alanları doğrudan predictions JSON'undan enjekte edilir. LLM'in eline
  sayı yazma fırsatı yapısal olarak geçmez.

---

## 7. GÜVENLİK KİLİTLERİ (kod seviyesinde, LLM kapatamaz)

1. **Paper-only kilidi:** Sistem varsayılan olarak sahte-parayla çalışır.
   Gerçek para modu config'de kilitlidir. Açılma koşulu: ≥3 ay paper +
   tanımlı metrik eşikleri. Bu kilidi yalnızca İNSAN açar; hiçbir ajan,
   hiçbir kod otomatik açamaz.
2. **İki kapı:**
   - Kapı 1 (veri girişi): Critic + kod kontrolleri (yarış sayısı, null
     oranı, tarih tutarlılığı, leakage audit). Geçmeyen veri modele girmez.
   - Kapı 2 (rapor çıkışı): Rapordaki HER sayı, kod ile predictions
     JSON'una diff'lenir. Eşleşmeyen tek sayı → rapor reddedilir. Bu
     kontrolü LLM DEĞİL kod yapar.
3. **3-deneme kuralı:** Aynı adım 3 kez başarısız olursa DUR, "FAILED"
   yaz, kullanıcıya tek satır özet ver. Sonsuz retry YASAK. "Tahminle
   devam edeyim" seçeneği YOK.
4. **Trip-wire:** Ganyan'ın ±2σ konfidans sapma freni korunur. Model
   günlük ortalama konfidansı baseline'dan saparsa banner/halt.
5. **Bütçe tavanı:** Adım başına max tool call, run başına süre/token
   tavanı. Aşım → kill + alarm.

---

## 8. YEREL LLM — sistemden ÇIKARILDI

Yerel model (Gemma/Ollama) bu sistemden bilinçli olarak çıkarıldı.
Gerekçe: tek gerçek görevi at/jokey isim eşleştirmeydi; bu iş TJK
verisindeki benzersiz ID'lerle (AtId, KEY vb.) çözülür — LLM gerektirmez.
Kalan ufak normalizasyon (büyük/küçük harf, boşluk) basit Python string
işlemleriyle, halüsinasyon riski sıfır, yapılır. Ek bir LLM = ek bağımlılık,
ek bakım, ek halüsinasyon kaynağı; faydası bu yükten düşük.

- **Kural:** Bu sisteme yerel veya bulut, EK bir LLM çağrı katmanı
  eklenmez. İsim/etiket eşleştirme ID ve string işlemleriyle yapılır.
- Eğer ileride ID ile çözülemeyen, gerçekten serbest-metin bir normalizasyon
  ihtiyacı çıkarsa, önce kullanıcıya sorulur — varsayılan olarak eklenmez.
- "LLM sayı üretmez" kuralı (Bölüm 0) zaten Claude Code için de geçerli;
  yerel model gitmesi bu savunmayı zayıflatmaz.

---

## 9. ÇALIŞMA TARZI (vibecoding kuralları)

1. **Plan önce, kod sonra.** Geri alınamaz veya büyük bir değişiklik
   öncesi tek satır özet sun, onay bekle (kullanıcının prensibi).
2. **Cerrahi ol.** Görev kapsamı dışına çıkma. Başka sorun görürsen NOT
   et, SOR, dokunma.
3. **Her şeyi çalıştırarak doğrula.** "Bu kod çalışır" deme; çalıştır,
   çıktıyı göster. Test varsa koştur.
4. **Token bilinci:** Gereksiz dosya yükleme, gereksiz uzun çıktı üretme.
   Az ve doğru bilgi, çok ve gürültülü bilgiden iyidir.
5. **Sessiz varsayım yasak.** Belirsizliği yüzeye çıkar. Özellikle
   fiyat/tarih/metrik/oran gibi yüksek riskli alanlarda dur ve doğrula.
6. **Hafıza sıfırdan:** Her oturum bu dosyayı okuyarak başlar. "Hatırladım"
   deme; bağlam lazımsa bu dosyadan veya repodan oku.

---

## 10. KURULUM SIRASI (bu sırayla ilerle, atlama)

```
[1] Önkoşul testleri      authKey YOK (atlandı); TJK ToS/robots.txt oku
[2] Fork + bu CLAUDE.md   repo'yu klonla, Postgres/uv kur, Ganyan'ı ayağa
                          kaldır, BU dosyayı köke koy
[3] Scraper doğrula       Ganyan scraper'ı bugünün verisini çekiyor mu;
                          TJK sayfası değiştiyse tamir
[4] Statik backfill       geçmiş veriyi tarih döngüsüyle çek (saniyede 1),
                          DB'ye yaz — bir kez
[5] Feature + model       Ganyan'dan adapte, HER feature için leakage testi
[6] Kapılar               Kapı 1 (Critic+kod) ve Kapı 2 (kod-diff) kur
[7] Ajan katmanı          Orkestratör/Critic/Rapor — çekirdek sağlamken.
                          ÖNCE skill-kurulum kapılarını kontrol et (aşağı bak);
                          dördü de doğru değilse skill YAZMA, çekirdeğe dön.
[8] Paper trading         kilit kapalı, 3 ay sayaç başlar
```

(Not: Yerel LLM/Gemma entegrasyonu adımı çıkarıldı — bkz. Bölüm 8.)

Her adım bitmeden sonrakine geçme. Bir adım FAILED olursa dur, raporla,
yeni talimat bekle.

---

## 10.1 SKILL KURULUMU — ne zaman ve neler

Ajan katmanı (Adım 7) **skill** dosyaları olarak kurulur. Ama skill
yazmadan önce DÖRT KAPI da doğru olmalı. Biri bile eksikse skill YAZMA,
çekirdeğe dön — erken yazılan skill havada kalır ve halüsinasyon kapısı açar.

**Dört kapı (hepsi doğru olmalı):**
1. Çekirdek pipeline uçtan uca bir kez çalıştı (scraper → DB → feature →
   model → bir predictions JSON'u üretildi).
2. Gerçek veri yapısı görülüyor (predictions JSON, DB tabloları, feature
   çıktısı — tahmin değil, gözle görülen gerçek alan adları).
3. Kapı 2 (kod-diff) çalışıyor (rapor sayılarını JSON'a karşı denetleyen kod).
4. İlgili "ajan işi" en az bir kez elle yapıldı ve tekrar ettiği görüldü.

**Tek soruyla test:** "Skill içeriğini gerçek bir çıktıya BAKARAK mı
yazıyorum, yoksa nasıl olacağını HAYAL EDEREK mi?" Hayal ediyorsan erken.

**Yazılacak skill'ler (sadece YARGI/ROL içerir, hesap/sayı ASLA):**
| Skill | Görevi | İçinde ASLA olmaz |
|---|---|---|
| `orchestrator` | DAG yürüt, state oku/yaz, hata yönet, eskalasyon dili | hesap, sayı |
| `data-critic` | ham veri denetim tarzı, PASS/FAIL kararı, anomali yakalama | veri düzeltme yetkisi |
| `report-writer` | anlatı kurma, "doğrulanmadı" damgası, ton | sayı yazma (JSON'dan enjekte) |

**Skill OLMAYACAK olanlar (deterministik = kod):** scraper, feature
üretimi, model, Kapı 2 diff, Kelly/olasılık/istatistik. Bunlar skill
yapılırsa LLM yorumuna açılır ve her gün farklı sonuç riski doğar.

---

## ÖZET — tek cümle

Bu sistem geleceği görmez; geçmişi düzgün hatırlar, hesabı duygusuz kod
yapar, yanıldığını ölçer, emin olmadığında susar, ve gerçek paraya
yalnızca insan onayıyla dokunur. Sayı üreten her şey koddur; LLM sadece
metin işler ve asla uydurmaz.
