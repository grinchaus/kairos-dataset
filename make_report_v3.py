from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/lab1-v3"))
    parser.add_argument(
        "--output", type=Path, default=Path("submission-v3/Lab1_Rostislav_Manzyuk.pdf")
    )
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text())
    if metrics.get("data_kind") != "official-release":
        raise ValueError("The report requires official release measurements")
    selection = metrics["selection"]
    selected = metrics["test"]["selected_blend"]
    previous = metrics["test"]["previous_v2"]
    baseline = metrics["test"]["official_baseline"]
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in [
        ("Report", "DejaVuSans.ttf"),
        ("Report-Bold", "DejaVuSans-Bold.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold")
    styles = getSampleStyleSheet()
    for name in ["Title", "Heading2", "BodyText"]:
        styles[name].fontName = "Report" if name == "BodyText" else "Report-Bold"
    styles["Title"].fontSize = 18
    styles["Title"].leading = 24
    styles["Heading2"].fontSize = 12
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].keepWithNext = True
    styles["BodyText"].fontSize = 9.4
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 6
    story = []

    def paragraph(value, style="BodyText"):
        value = value.replace("—", "-").replace("–", "-")
        for char in ["«", "»", "“", "”"]:
            value = value.replace(char, '"')
        item = Paragraph(value, styles[style])
        story.append(KeepTogether([item]) if style == "BodyText" else item)

    def table(rows):
        item = Table(rows, hAlign="LEFT", repeatRows=1)
        item.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Report"),
                    ("FONTNAME", (0, 0), (-1, 0), "Report-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f54")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.lightgrey),
                ]
            )
        )
        story.extend([item, Spacer(1, 0.25 * cm)])

    paragraph("ЛР 1. Прогноз результата LinkedIn-поста", "Title")
    paragraph(
        "Ростислав Манзюк, grinchaus<br/>Social Media Analytics, Agents &amp; LLMs, ИТМО"
    )
    paragraph("Задача и результат", "Heading2")
    paragraph(
        "Нужно определить, сработает ли пост хуже обычного результата своего автора, примерно так же "
        'или лучше. Классы - "below", "typical" и "above". Обычный уровень - медиана log(1 + reactions) '
        "по более ранним собственным постам автора, которых должно быть хотя бы пять. Основная метрика - macro-F1."
    )
    target_text = (
        "Порог 0.70 достигнут."
        if metrics["target_reached"]
        else f"До порога 0.70 осталось {0.70-selected['macro_f1']:.4f}."
    )
    paragraph(
        f"В этом раунде цель была приблизиться к macro-F1 0.70. Итог на официальном test - "
        f"<b>{selected['macro_f1']:.4f}</b>, предыдущая локальная версия - {previous['macro_f1']:.4f}. {target_text}"
    )
    paragraph("Данные и схема проверки", "Heading2")
    paragraph(
        f"Взял официальный релиз Kairos: {metrics['train_posts']:,} размеченных постов от "
        f"{metrics['train_authors']:,} авторов в train и {metrics['test_posts']:,} постов от "
        f"{metrics['test_authors']:,} авторов в test. Разметка полностью воспроизводится штатным kairos.label. "
        "Авторы между train и test не пересекаются."
    )
    paragraph(
        "Новые подходы выбирал на двух разбиениях train по авторам: seed 42 и 137, по 20% авторов "
        "для валидации. PCA, словари, масштабирование и графовая статистика обучаются только на fit. "
        "Веса смешивания и поправки классов общие для двух разбиений. Разрешил не больше трёх "
        "добавлений в ансамбль, каждое с приростом среднего macro-F1 хотя бы 0.0004. "
        "На каждом разбиении результат должен сохраняться не ниже фиксированной предыдущей версии."
    )
    paragraph(
        "После выбора зафиксировал конфигурацию и отдельно проверил её на отложенных авторах "
        "с seed 2026. По этой проверке настройки уже не менял. Разбиения частично пересекаются, "
        "поэтому их оценки нельзя считать независимыми. Официальный test оценивался в предыдущих "
        "версиях работы; в этом раунде его повторная оценка выполнена после фиксации всех настроек."
    )
    paragraph("Что добавил и проверил", "Heading2")
    paragraph(
        "Сохранил предыдущий ансамбль CatBoost, TF-IDF и небольшой нейросети на multilingual E5-small. "
        "Он использует содержание поста, профиль, историю реакций и сходство с более ранними постами автора. "
        "В своей предобработке также сохранил исправление смешанных ISO-форматов дат, которое восстановило "
        "8 185 дат сбора. Исходный baseline запускается без изменений."
    )
    paragraph(
        "В новых признаках исправил чтение размеров изображений: раньше выбиралось поле с вложенными "
        "списками, из-за чего размеры оставались нулевыми. Добавил форму и площадь изображений, "
        "историю результатов для такого же формата поста и повторных упоминаний, временные окна "
        "7, 30, 90 и 365 дней, квантили предыдущих реакций и историю комментариев, репостов и типов реакций."
    )
    paragraph(
        "Проверил три постановки для CatBoost: обычные три класса, две бинарные задачи для границ классов "
        "и регрессию отклонения от обычного уровня автора. Для регрессии и бинарных моделей "
        "привёл вероятности к общей шкале с учётом частот классов на fit. Затем сравнил их вклад в ансамбль."
    )
    paragraph(
        "Для истории текста дополнительно проверил ядровую регрессию по предыдущим постам того же автора. "
        "Она даёт оценку ожидаемого отклонения и неопределённости с учётом сходства текстов. "
        "Текущий пост и более поздние реакции в эту историю не входят."
    )
    paragraph(
        "Проверил граф упоминаний людей и компаний и общих работодателей. Графовая статистика строится "
        "по fit-авторам. Для обучающего поста полностью вычитается вклад его автора, для validation "
        "его целевые значения вообще не читаются. Отдельно проверил TF-IDF по кодам упоминаний."
    )
    paragraph(
        "Дополнительно проверил фиксированный encoder Qwen3-Embedding-0.6B. "
        "Использовал инструкцию задачи, ограничение 1024 токена, вектор последнего действительного "
        "токена и L2-нормализацию. По этим векторам проверил CatBoost и небольшую нейросеть с историей "
        "автора. Все тексты обрабатывались локально. Во внешние API данные не отправлялись."
    )
    paragraph(
        "Также проверил сами изображения: по одному изображению, обложке или кадру видео на пост. "
        "Для GIF использовал первый кадр. Локальный CLIP ViT-B/32 дал векторы размерности 512. "
        "Добавил их PCA-представление, сходство с подписью и примерами типов контента, а также "
        "результаты похожих более ранних изображений автора. Визуальные варианты сравнил отдельно "
        "и вместе с Qwen. Все выбранные файлы проверены по SHA-256, изображения во внешние API не отправлялись."
    )
    paragraph("Результаты выбора на train", "Heading2")
    labels = {
        "v2": "Предыдущий ансамбль",
        "rich-multiclass": "Расширенные признаки",
        "rich-ordinal": "Две границы классов",
        "rich-regression": "Регрессия отклонения",
        "kernel-multiclass": "Ядровая история",
        "graph-text": "TF-IDF + упоминания",
        "network": "Графовая статистика",
        "vision": "Изображения + CatBoost",
        "qwen-vision": "Qwen + изображения",
        "qwen-depth6": "Qwen + CatBoost, depth 6",
        "qwen-depth7": "Qwen + CatBoost, depth 7",
        "qwen-dense": "Qwen + нейросеть",
        "qwen-dense-1000": "Qwen + нейросеть, seed 2",
        "qwen-dense-ensemble": "Qwen + две нейросети",
        "multimodal-dense-ensemble": "Нейросеть: Qwen + CLIP",
    }
    rows = [["Подход с общей калибровкой", "Val 42 F1", "Val 137 F1"]]
    for name, candidate in selection["candidates"].items():
        rows.append(
            [
                labels.get(name, name),
                *[f"{value:.4f}" for value in candidate["per_split"]],
            ]
        )
    rows.append(
        [
            "Выбранный ансамбль",
            *[
                f"{selection['validation'][seed]['macro_f1']:.4f}"
                for seed in ["42", "137"]
            ],
        ]
    )
    table(rows)
    paragraph(
        "Эти оценки использовались для выбора модели и могут быть оптимистичными. "
        "Таблица сравнивает подходы целиком; вклад каждого отдельного признака здесь не выделял."
    )
    audit = metrics["audit"]
    table(
        [
            ["Дополнительная проверка, seed 2026", "Macro-F1", "QWK", "Accuracy"],
            [
                "Предыдущая версия",
                *[
                    f"{audit['previous_v2'][key]:.4f}"
                    for key in ["macro_f1", "qwk", "accuracy"]
                ],
            ],
            [
                "Фиксированный новый ансамбль",
                *[
                    f"{audit['selected_blend'][key]:.4f}"
                    for key in ["macro_f1", "qwk", "accuracy"]
                ],
            ],
        ]
    )
    paragraph(
        f"В дополнительной проверке {audit['authors']:,} авторов и {audit['posts']:,} постов. "
        "Настройки до её запуска уже были зафиксированы."
    )
    paragraph("Итог на официальном test", "Heading2")
    table(
        [["Модель", "Macro-F1", "QWK", "Accuracy"]]
        + [
            [name, *[f"{value[key]:.4f}" for key in ["macro_f1", "qwk", "accuracy"]]]
            for name, value in [
                ("Исходный baseline", baseline),
                ("Предыдущая локальная версия", previous),
                ("Выбранный новый ансамбль", selected),
            ]
        ]
    )
    low, high = metrics["confidence_intervals"]["macro_f1_95ci"]["selected_blend"]
    difference = metrics["previous_comparison"]
    delta_low, delta_high = difference["paired_delta_95ci"]
    paragraph(
        f"Прирост к предыдущей версии - {difference['delta_macro_f1']:+.4f} macro-F1. "
        f"95% парный интервал разницы по авторам - [{delta_low:+.4f}; {delta_high:+.4f}]. "
        f"95% интервал итогового macro-F1 - [{low:.4f}; {high:.4f}]. "
        f"Использовал bootstrap по авторам, {metrics['confidence_intervals']['samples']} повторов: посты одного автора остаются вместе."
    )
    paragraph(
        "В README исходного проекта указан baseline 0.5410. Здесь использую результат "
        f"его отдельного запуска в этом окружении - {baseline['macro_f1']:.4f}."
    )
    table(
        [["Компонент итогового ансамбля", "Вес"]]
        + [
            [labels.get(name, name), f"{weight:.3f}"]
            for name, weight in selection["components"].items()
        ]
    )
    paragraph(
        "Множители вероятностей классов: "
        + escape(str([round(value, 3) for value in selection["class_weights"]]))
        + ". "
        "Число итераций и эпох для полного обучения зафиксировано по медиане выбранных значений на двух валидациях. "
        "Все параметры, версии encoder и отпечатки исходников сохранены в selection.json."
    )
    report = selected["classification_report"]
    table(
        [["Класс", "Precision", "Recall", "F1", "Постов"]]
        + [
            [
                name,
                f"{report[name]['precision']:.4f}",
                f"{report[name]['recall']:.4f}",
                f"{report[name]['f1-score']:.4f}",
                str(int(report[name]["support"])),
            ]
            for name in ["below", "typical", "above"]
        ]
    )
    fig, axis = plt.subplots(figsize=(5.5, 3.8))
    matrix = np.asarray(selected["confusion_matrix"])
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=["below", "typical", "above"],
        yticklabels=["below", "typical", "above"],
        xlabel="Предсказанный класс",
        ylabel="Настоящий класс",
    )
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(
            column,
            row,
            f"{value:,}",
            ha="center",
            va="center",
            color="white" if value > matrix.max() / 2 else "black",
        )
    fig.tight_layout()
    image = args.results / "confusion_matrix.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)
    story.append(Image(str(image), width=11 * cm, height=7.6 * cm))
    story.append(PageBreak())
    paragraph("Ограничения и воспроизведение", "Heading2")
    paragraph(
        "Реакции текущего поста, его label и residual не входят в признаки для прогноза. "
        "В собственной истории используются только более ранние посты. Реакции прошлого поста "
        "при этом известны на момент сбора данных, как в официальном baseline. Изменение текущих "
        "и будущих реакций не меняет признаки текущего поста - это проверяется тестами."
    )
    paragraph(
        "Число подписчиков, профиль и возраст поста относятся к снимку на момент сбора. "
        "Поэтому это оценка для офлайн-задания. Она не обещает такой же результат сразу "
        "при публикации нового поста. В графе используются обезличенные коды общих сущностей. "
        "В визуальном эксперименте использованы изображения и отдельные кадры; движение и звук видео не обрабатывались."
    )
    paragraph(
        f"Python {escape(metrics['python'])}. Кодирование текстов выполняется отдельно от обучения. "
        "Для чистого запуска нужны официальный распакованный набор и зависимости из requirements-embeddings.txt. "
        "Зафиксированную конфигурацию воспроизводит команда:"
    )
    paragraph(
        "Тяжёлые этапы выполняются последовательно с низким приоритетом CPU и контролем памяти. "
        "Матрицы нейросети хранятся на диске, на GPU передаются отдельные батчи. "
        "При полном переобучении на другом оборудовании возможны небольшие численные отличия; "
        "повторный прогноз из сохранённой модели проверяется на точное совпадение CSV."
    )
    paragraph("<font face='Courier'>python run_lab1_v3.py --data data</font>")
    paragraph(
        "Команда готовит недостающие локальные эмбеддинги, обучает модели с фиксированными "
        "настройками, выполняет дополнительную проверку и оценивает полный test. После сохранения "
        "модели отдельный запуск predict_lab1_v3.py обязан дать тот же CSV. В комплекте лежат "
        "выполненный notebook, отчёт, предсказания и метрики. Данные, пароли и веса моделей в Git не добавлены."
    )
    paragraph(
        'Источники: <a href="https://github.com/Danil482/kairos-dataset">официальный baseline</a>, '
        '<a href="https://kairos-labs.ai/dataset/">Kairos</a>, '
        '<a href="https://huggingface.co/intfloat/multilingual-e5-small">multilingual E5-small</a>, '
        '<a href="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B">Qwen3-Embedding-0.6B</a>, '
        '<a href="https://huggingface.co/openai/clip-vit-base-patch32">CLIP ViT-B/32</a>, уточнения из чата курса.'
    )

    def footer(canvas, document):
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.7 * cm, 1 * cm, "Ростислав Манзюк | ЛР 1")
        canvas.drawRightString(19.3 * cm, 1 * cm, str(document.page))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(args.output),
        pagesize=(21 * cm, 29.7 * cm),
        title="ЛР 1 - Дополнительные эксперименты",
        author="Ростислав Манзюк",
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
