# Media Tracking HF

Скрипты для скачивания и загрузки версий моделей и датасетов в репо HF

## Установка

```
git clone https://github.com/ddreamboy/media_tracking_hf.git
```

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
cp .env.example .env
```


## Использование

- `python upload.py` - загрузка модели и датасета на Hugging Face Hub
- `python download.py` - скачивание последней версии модели и датасета

## Структура проекта

- `model/` - папка с моделями BERTopic
- `dataset/` - папка с датасетами (raw/ и processed/)