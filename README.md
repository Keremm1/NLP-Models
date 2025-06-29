# GPT-2 TensorFlow 2.0 

Bu depo, GPT-2 mimarisinin TensorFlow 2.0 kullanılarak sıfırdan uygulanmasını içermektedir. Model, temel bileşenleri olan attention, embedding, feed-forward ve layer normalization katmanları ayrı ayrı oluşturularak modüler biçimde geliştirilmiştir. Projede eğitim, örnek üretimi ve veri işleme pipeline’ı da dahil edilmiştir.

---

## Proje Dosya Yapısı

```plaintext
NLP_Architectures/
├── train_and_test.py # Eğitim ve test süreçlerini yöneten script
├── transformer.py # Genel transformer mimarisi

models/
└── gpt2/
    ├── layers/ # GPT-2 modelindeki temel katmanlar
    │   ├── init.py
    │   ├── attention_layer.py # Çoklu başlıklı dikkat mekanizması
    │   ├── embedding_layer.py # Gömme (embedding) katmanı
    │   ├── feed_forward.py # İleri beslemeli ağ katmanı
    │   ├── layer_norm.py # Katman normu uygulaması
    │
    ├── utils/ # Yardımcı fonksiyonlar
    │   ├── init.py
    │   ├── tf_utils.py # TensorFlow ile ilgili araçlar
    │
    ├── data_pipeline.py # Eğitim verisinin hazırlanması
    ├── gpt2_model.py # GPT-2 model sınıfı
    ├── pre_process.py # Veri ön işleme fonksiyonları
    ├── requirements.txt # Gerekli kütüphaneler listesi
    ├── sample.py # Basit metin üretimi örneği
    ├── sequence_generator.py # Gelişmiş metin üretimi
    ├── train_gpt2.py # Modeli eğitme scripti

readme/ # Belgelendirme klasörü
```

---

## Proje Amacı

Bu projenin amacı, GPT-2 modelinin bileşenlerini anlamak ve bunları TensorFlow 2.0 kullanarak sıfırdan inşa etmektir. Model, metin tamamlama ve üretimi için kullanılabilir hale getirilmiştir. Bu sayede doğal dil işleme alanında transformer tabanlı modellerin yapısı, veri akışı ve eğitim süreçleri derinlemesine öğrenilmektedir.

---

##  GPT-2 Mimarisi

<p align="center">
  <b>GPT-2 Decoder Mimarisi</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/akanyaani/gpt-2-tensorflow2.0/master/images/GPT-2_Decoder.jpg" width="700"/>
</p>

<p align="center">
  <b>GPT-2 Hesaplama Grafiği</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/akanyaani/gpt-2-tensorflow2.0/master/images/GPT-2_Graph.jpg" width="700"/>
</p>

---

## Kurulum

Gerekli ortamı kurmak için aşağıdaki adımları takip edebilirsiniz:

### 1. Python ortamı oluşturun:
```bash
python -m venv gpt2-env
source gpt2-env/bin/activate  # veya Windows için: gpt2-env\Scripts\activate
```
### Gerekli kütüphaneleri yükleyin:
```bash
pip install -r models/gpt2/requirements.txt
```
**Requirements**

*  python >= 3.6
*  setuptools==41.0.1
*  ftfy==5.6
*  tqdm==4.32.1
*  Click==7.0
*  sentencepiece==0.1.83
*  tensorflow-gpu==2.3.0
*  numpy==1.16.4


## Kullanım

### Eğitim

Modeli eğitmek için aşağıdaki komutu çalıştırın:

```bash
python models/gpt2/train_gpt2.py
```
### Metin Üretimi

Eğitilen model ile metin üretmek için:

```bash
python models/gpt2/sample.py --text "Bir zamanlar"
```

### Gelişmiş (Uzun Metin) Üretimi

Eğitilen model ile metin üretmek için:

```bash
python models/gpt2/sequence_generator.py --prompt "Yapay zeka geleceği nasıl şekillendirecek?"
```

