# Turkish Text Analysis with Zemberek and Logistic Regression

This repository contains a Python project for analyzing Turkish texts using Zemberek, a natural language processing tool for Turkish. The project includes functionalities for preprocessing texts, extracting roots and parts of speech, and performing author prediction using logistic regression.

## Project Overview

The main components of this project are:

1. **Text Analysis with Zemberek**: Using Zemberek to tokenize and analyze Turkish texts, extracting lemmas and parts of speech.
2. **Preprocessing Texts**: Lowercasing, removing stop words, and extracting lemmas from texts.
3. **Author Prediction**: Training a logistic regression model to predict the author of a given text.
4. **Word Frequency Analysis**: Generating a frequency list of words and their parts of speech in the given texts.

## Installation

To run this project, you need to install the following dependencies:

- zemberek-python
- scikit-learn
- stop-words

You can install these dependencies using pip:

```bash
pip install zemberek-python scikit-learn stop-words
```

# Usage
## Text Analysis
The analyze_text function tokenizes and analyzes a given Turkish text, returning a list of lemmas and their parts of speech.

Preprocessing Texts
The preprocess_text function processes a given Turkish text, lowercasing, removing non-alphabetic characters, removing stop words, and extracting lemmas.

## Author Prediction
The project includes a sample dataset of texts and authors. The texts are preprocessed, vectorized using TF-IDF, and used to train a logistic regression model. You can test the model with a new text to predict its author.

## Word Frequency Analysis
The write_word_frequencies function analyzes all given texts, counts the frequencies of each lemma and part of speech, and writes the results to a file.

## Example
Here is an example of how to use the project to predict the author of a given text:
```bash
target_text = "Your target text here."
target_text_preprocessed = preprocess_text(target_text, morphology)
target_vector = vectorizer.transform([target_text_preprocessed])
predicted_label = model.predict(target_vector)[0]
predicted_author = label_encoder.inverse_transform([predicted_label])[0]

print(f"The given text is likely written by {predicted_author}.")
```
## License
This project is licensed under the MIT License.


--------------------------------------------------------------------------------------------------------------------
# Zemberek ve Lojistik Regresyon ile Türkçe Metin Analizi

Bu depo, Türkçe metinleri analiz etmek için Zemberek'i kullanan bir Python projesini içerir. Proje, metinleri ön işleme, kök ve kelime türlerini çıkarma ve lojistik regresyon kullanarak yazar tahmini yapma işlevlerini içerir.

## Proje Genel Bakış

Bu projenin ana bileşenleri şunlardır:

1. **Zemberek ile Metin Analizi**: Türkçe metinleri tokenleştirmek ve analiz etmek için Zemberek'i kullanarak kök ve kelime türlerini çıkarmak.
2. **Metinleri Ön İşleme**: Metinleri küçük harflere dönüştürme, durma kelimelerini çıkarma ve kökleri çıkarma.
3. **Yazar Tahmini**: Belirli bir metnin yazarını tahmin etmek için bir lojistik regresyon modeli eğitme.
4. **Kelime Frekans Analizi**: Verilen metinlerdeki kelimelerin ve kelime türlerinin frekans listesini oluşturma.

## Kurulum

Bu projeyi çalıştırmak için aşağıdaki bağımlılıkları yüklemeniz gerekmektedir:

- zemberek-python
- scikit-learn
- stop-words

Bu bağımlılıkları pip kullanarak yükleyebilirsiniz:

```bash
pip install zemberek-python scikit-learn stop-words
```
# Kullanım
## Metin Analizi
analyze_text fonksiyonu, verilen bir Türkçe metni tokenleştirir ve analiz eder, kök ve kelime türlerinin bir listesini döndürür.

## Metinleri Ön İşleme
preprocess_text fonksiyonu, verilen bir Türkçe metni işler, küçük harflere dönüştürür, alfasayısal olmayan karakterleri çıkarır, durma kelimelerini çıkarır ve kökleri çıkarır.

## Yazar Tahmini
Proje, metinler ve yazarlar içeren bir örnek veri seti içerir. Metinler ön işlenir, TF-IDF kullanılarak vektörleştirilir ve bir lojistik regresyon modeli ile eğitilir. Modeli yeni bir metinle test ederek yazarını tahmin edebilirsiniz.

## Kelime Frekans Analizi
write_word_frequencies fonksiyonu, verilen tüm metinleri analiz eder, her bir kök ve kelime türünün frekansını sayar ve sonuçları bir dosyaya yazar.

## Örnek
İşte belirli bir metnin yazarını tahmin etmek için projenin nasıl kullanılacağına dair bir örnek:

```bash
target_text = "Hedef metninizi buraya girin."
target_text_preprocessed = preprocess_text(target_text, morphology)
target_vector = vectorizer.transform([target_text_preprocessed])
predicted_label = model.predict(target_vector)[0]
predicted_author = label_encoder.inverse_transform([predicted_label])[0]

print(f"Verilen metin muhtemelen {predicted_author} tarafından yazılmıştır.")
```
