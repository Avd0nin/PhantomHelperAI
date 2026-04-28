import os
from openai import OpenAI
from flask import Flask, request, render_template, jsonify
import json
import re
import html as html_utils

from env_loader import load_env


load_env()


class AICore:
    def __init__(self):
        api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("AI_API_KEY is not set. Add it to .env.")

        self.client = OpenAI(
            base_url=os.getenv("AI_BASE_URL", "https://neuroapi.host/v1"),
            api_key=api_key
        )
        self.model = os.getenv("AI_MODEL", "gpt-4.1-mini")

    def _chat(self, messages, temperature=0.7, max_tokens=2000, response_format=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            payload["response_format"] = response_format

        response = self.client.chat.completions.create(**payload)
        return response.choices[0].message.content.strip()

    def normalize_summary_html(self, html):
        if not html:
            return html

        cleaned = html.replace('```html', '').replace('```', '').strip()
        cleaned = re.sub(r'(?im)^\s*-{2,}\s*$', '', cleaned)
        cleaned = re.sub(r'(?is)<p>\s*-{2,}\s*</p>', '', cleaned)
        cleaned = re.sub(r'(?is)<div>\s*-{2,}\s*</div>', '', cleaned)
        cleaned = re.sub(
            r'(<h1[^>]*>\s*)Краткий\s+конспект\s+по\s+теме\s*',
            r'\1',
            cleaned,
            flags=re.IGNORECASE
        )
        return cleaned.strip()

    def normalize_html(self, html):
        if not html:
            return ""

        cleaned = html.replace('```html', '').replace('```', '').strip()
        cleaned = re.sub(
            r'^\s*<html[^>]*>\s*<body[^>]*>\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*</body>\s*</html>\s*$', '',
                         cleaned, flags=re.IGNORECASE)
        return cleaned

    def _normalize_option_text(self, raw_text):
        text = html_utils.unescape(str(raw_text or ""))
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def _extract_choice_option_groups(self, html):
        if not html:
            return []

        groups = []
        ul_pattern = re.compile(
            r'<ul[^>]*class=["\'][^"\']*\boptions-list\b[^"\']*["\'][^>]*>([\s\S]*?)</ul>',
            flags=re.IGNORECASE
        )
        li_pattern = re.compile(
            r'<li[^>]*class=["\'][^"\']*\boption-item\b[^"\']*["\'][^>]*>([\s\S]*?)</li>',
            flags=re.IGNORECASE
        )
        label_pattern = re.compile(r'<label[^>]*>([\s\S]*?)</label>', flags=re.IGNORECASE)
        value_pattern = re.compile(r'\bvalue\s*=\s*["\']([^"\']+)["\']', flags=re.IGNORECASE)

        for ul_match in ul_pattern.finditer(html):
            ul_content = ul_match.group(1)
            options = []
            for li_match in li_pattern.finditer(ul_content):
                li_html = li_match.group(1)
                value_match = value_pattern.search(li_html)
                option_value = (value_match.group(1).strip().lower() if value_match else "")

                label_match = label_pattern.search(li_html)
                option_text_raw = label_match.group(1) if label_match else li_html
                option_text = self._normalize_option_text(option_text_raw)

                options.append({
                    "value": option_value,
                    "text": option_text
                })
            if options:
                groups.append(options)
        return groups

    def _test_html_has_invalid_choices(self, html):
        option_groups = self._extract_choice_option_groups(html)
        if not option_groups:
            return True

        for options in option_groups:
            if len(options) != 4:
                return True

            correct_count = sum(1 for option in options if option["value"] == "correct")
            if correct_count != 1:
                return True

            option_texts = [option["text"] for option in options]
            if any(not text for text in option_texts):
                return True
            if len(set(option_texts)) != 4:
                return True

        return False

    def _repair_test_html(self, html, subject, klass, theme):
        candidate = self.normalize_html(html)
        if not candidate:
            return candidate

        if not self._test_html_has_invalid_choices(candidate):
            return candidate

        for _ in range(2):
            repair_prompt = f"""
Исправь HTML теста по предмету "{subject}" для {klass} класса по теме "{theme}".

Строгие правила:
1) Верни только HTML, без markdown и комментариев.
2) Оставь общий формат question-block/options-list/check-btn/feedback.
3) В каждом вопросе с выбором:
- ровно 4 варианта ответа;
- ровно один вариант имеет value="correct";
- три варианта имеют value="wrong";
- все 4 варианта должны быть текстово разными (никаких дублей).
4) Сохрани нумерацию вопросов и адекватную сложность.
5) Тест должен содержать ровно 10 вопросов.

Исходный HTML:
{candidate}
"""
            repaired = self._chat(
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            candidate = self.normalize_html(repaired)
            if not self._test_html_has_invalid_choices(candidate):
                return candidate

        return candidate

    def _safe_json_parse(self, raw_content):
        if not raw_content:
            return None

        cleaned = raw_content.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _normalize_contest_payload(self, payload, fallback_tasks_count=3):
        def sanitize_io_text(value):
            raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            raw = raw.replace("```", "").replace("`", "")
            raw = re.sub(r'(?im)^\s*(input|output|expected|actual)\s*:\s*', '', raw)
            lines = []
            for line in raw.split("\n"):
                cleaned = line.rstrip()
                # Preserve negative numbers that sometimes arrive as "- 998".
                cleaned = re.sub(r'^\s*-\s+(\d+(?:[.,]\d+)?)\s*$', r'-\1', cleaned)
                cleaned = re.sub(r'^\s*[-*]\s+', '', cleaned)
                cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned)
                lines.append(cleaned)
            while lines and lines[0] == "":
                lines.pop(0)
            while lines and lines[-1] == "":
                lines.pop()
            return "\n".join(lines)

        if not isinstance(payload, dict):
            payload = {}

        contest_title = str(payload.get("contest_title") or "Новый контест").strip()
        intro = str(payload.get("intro") or "Контест по олимпиадному программированию").strip()

        raw_tasks = payload.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raw_tasks = []

        normalized_tasks = []
        requested_count = max(1, min(10, int(fallback_tasks_count or 3)))
        letter_base = ord('A')

        for idx, raw_task in enumerate(raw_tasks[:requested_count]):
            if not isinstance(raw_task, dict):
                continue

            task_id = str(raw_task.get("id") or chr(letter_base + idx)).strip()[:8]
            title = str(raw_task.get("title") or f"Задача {task_id}").strip()
            difficulty = str(raw_task.get("difficulty") or "medium").strip().lower()
            try:
                difficulty_score = int(raw_task.get("difficulty_score", 0))
            except (TypeError, ValueError):
                difficulty_score = 0
            if difficulty_score < 1 or difficulty_score > 10:
                if difficulty == "easy":
                    difficulty_score = 3
                elif difficulty == "hard":
                    difficulty_score = 8
                elif difficulty == "olymp":
                    difficulty_score = 10
                else:
                    difficulty_score = 5
            statement_html = self.normalize_html(str(raw_task.get("statement_html") or raw_task.get("statement") or "").strip())
            if not statement_html:
                statement_html = "<p>Описание задачи не сгенерировано.</p>"

            input_format = str(raw_task.get("input_format") or "См. условие.").strip()
            output_format = str(raw_task.get("output_format") or "См. условие.").strip()
            constraints = str(raw_task.get("constraints") or "Ограничения не указаны.").strip()

            raw_examples = raw_task.get("examples", [])
            if not isinstance(raw_examples, list):
                raw_examples = []
            examples = []
            for example in raw_examples[:5]:
                if not isinstance(example, dict):
                    continue
                examples.append({
                    "input": sanitize_io_text(example.get("input", "")),
                    "output": sanitize_io_text(example.get("output", "")),
                    "explanation": str(example.get("explanation", "")).strip()
                })

            raw_tests = raw_task.get("tests", [])
            if not isinstance(raw_tests, list):
                raw_tests = []
            tests = []
            for test in raw_tests[:25]:
                if not isinstance(test, dict):
                    continue
                input_text = sanitize_io_text(test.get("input", ""))
                output_text = sanitize_io_text(test.get("output", ""))
                if not input_text and not output_text:
                    continue
                tests.append({
                    "input": input_text,
                    "output": output_text,
                    "note": str(test.get("note", "")).strip()
                })

            if not tests and examples:
                tests = [
                    {"input": ex["input"], "output": ex["output"], "note": "Тест из примера"}
                    for ex in examples
                    if ex["input"] or ex["output"]
                ]

            if not tests:
                tests = [{"input": "", "output": "", "note": "Тесты не сгенерированы"}]

            normalized_tasks.append({
                "id": task_id,
                "title": title,
                "difficulty": difficulty,
                "difficulty_score": difficulty_score,
                "statement_html": statement_html,
                "input_format": input_format,
                "output_format": output_format,
                "constraints": constraints,
                "examples": examples,
                "tests": tests
            })

        if not normalized_tasks:
            for idx in range(requested_count):
                letter = chr(letter_base + idx)
                normalized_tasks.append({
                    "id": letter,
                    "title": f"Задача {letter}",
                    "difficulty": "medium",
                    "difficulty_score": 5,
                    "statement_html": "<p>Не удалось сгенерировать условие задачи. Повторите попытку.</p>",
                    "input_format": "См. условие.",
                    "output_format": "См. условие.",
                    "constraints": "Ограничения не указаны.",
                    "examples": [],
                    "tests": [{"input": "", "output": "", "note": "Тесты не сгенерированы"}]
                })

        return {
            "contest_title": contest_title,
            "intro": intro,
            "tasks": normalized_tasks
        }

    def create_contest_round(self, description, difficulty, tasks_count, topics):
        safe_description = str(description or "").strip()
        raw_difficulty = str(difficulty or "5").strip().lower()
        safe_tasks_count = max(1, min(10, int(tasks_count or 3)))
        safe_topics = topics if isinstance(topics, list) else []
        safe_topics = [str(topic).strip() for topic in safe_topics if str(topic).strip()]

        difficulty_alias_map = {
            "easy": 2,
            "medium": 5,
            "hard": 7,
            "olymp": 10
        }
        try:
            difficulty_level = int(raw_difficulty)
        except (TypeError, ValueError):
            difficulty_level = difficulty_alias_map.get(raw_difficulty, 5)

        difficulty_level = max(1, min(10, difficulty_level))

        if difficulty_level <= 4:
            style_rules = (
                "Формулировки делай простыми и учебными, как в методичке: короткое и прямое условие, "
                "без лишнего сюжета. На базовом уровне допустимы задачи формата "
                "\"даны A и B, сделайте ...\"."
            )
        elif difficulty_level <= 7:
            style_rules = (
                "Формулировки делай понятными и практичными: можно добавить небольшой контекст, "
                "но главная цель — ясность и тренировка базовых/средних приёмов."
            )
        else:
            style_rules = (
                "Формулировки делай интересными и небанальными: допускается короткий сюжет/контекст "
                "(игры, роботы, логистика, анализ данных, соревнования), но без воды."
            )

        difficulty_label_map = {
            1: "очень базовый",
            2: "базовый",
            3: "ниже среднего",
            4: "средний-",
            5: "средний",
            6: "средний+",
            7: "повышенный",
            8: "сложный",
            9: "очень сложный",
            10: "олимпиадный"
        }
        difficulty_label = difficulty_label_map.get(difficulty_level, "средний")
        topics_text = ", ".join(safe_topics) if safe_topics else "без фиксированного списка"
        tests_per_task = 12

        prompt = f"""
Ты тренер по олимпиадному программированию и автор контестов высокого качества.
Сгенерируй JSON-объект контеста на {safe_tasks_count} задач.

Параметры:
- Сложность по шкале 1-10: {difficulty_level}
- Уровень сложности: {difficulty_label}
- Темы: {topics_text}
- Доп. пожелания: {safe_description if safe_description else "не указаны"}

Требования:
1) Задачи должны быть самостоятельными, разного характера и строго соответствовать уровню сложности.
1.1) У каждой задачи сложность обязана быть привязана к шкале 1-10:
     easy: 1-3, medium: 4-6, hard: 7-8, olymp: 9-10.
1.2) По набору задач сложность должна различаться: не делай все задачи одинаковыми.
     Дай естественный разброс от проще к сложнее в рамках выбранного уровня.
2) Стиль формулировок по уровню сложности: {style_rules}
3) Избегай полного копирования одной и той же структуры условий во всех задачах.
4) Для каждой задачи верни понятное условие в HTML (без html/head/body).
5) Для каждой задачи верни:
   - id (A, B, C...)
   - title
   - difficulty
   - difficulty_score (целое число 1-10, согласованное с difficulty)
   - statement_html
   - input_format
   - output_format
   - constraints
   - examples: ровно 2 примера с input/output/explanation
   - tests: не меньше {tests_per_task} тестов, где у каждого input/output/note
6) Тесты должны реально проверять решение и ловить неверные алгоритмы.
6.1) Для каждой задачи в tests обязательно смешай типы кейсов:
     - обычные рабочие случаи
     - граничные случаи
     - сложные/стрессовые случаи
     - случаи против типичных ошибок
6.2) Не ограничивайся только minimum/maximum. Добавляй "живые" и разнообразные данные:
     разные размеры входа, разные распределения значений, нетривиальные комбинации.
6.3) В examples тоже не делай только тривиальные крайние точки:
     примеры должны быть понятными, но содержательными и не однотипными.
6.1) Поля input/output возвращай как чистый текст данных без markdown, без префиксов
     "Input:", "Output:", без маркеров списка ("- ", "* ", "1. ").
7) Вывод строго в формате JSON, без markdown и без комментариев.
8) Ничего не сокращай и не обрывай JSON.

Формат:
{{
  "contest_title": "Название контеста",
  "intro": "Краткое описание контеста",
  "tasks": [
    {{
      "id": "A",
      "title": "Название",
      "difficulty": "easy|medium|hard|olymp",
      "difficulty_score": 1,
      "statement_html": "<h2>...</h2><p>...</p>",
      "input_format": "Описание ввода",
      "output_format": "Описание вывода",
      "constraints": "Ограничения",
      "examples": [
        {{"input": "...", "output": "...", "explanation": "..."}}
      ],
      "tests": [
        {{"input": "...", "output": "...", "note": "..."}}
      ]
    }}
  ]
}}
"""

        try:
            raw_content = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.45,
                max_tokens=9000,
                response_format={"type": "json_object"}
            )
        except Exception:
            raw_content = self._chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.45,
                max_tokens=9000
            )

        parsed = self._safe_json_parse(raw_content)

        if not parsed:
            repair_prompt = f"""
Преобразуй текст в валидный JSON строго по указанной схеме.
Верни только JSON без пояснений.
Текст:
{raw_content[:18000]}
"""
            repaired_content = self._chat(
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.1,
                max_tokens=9000
            )
            parsed = self._safe_json_parse(repaired_content)

        if not parsed:
            compact_prompt = f"""
Сгенерируй контест в JSON на {safe_tasks_count} задач.
Сложность: {difficulty_level}/10.
Для каждой задачи дай 1 пример и 3 теста.
Стиль условий: {style_rules}
Верни только JSON.
"""
            compact_content = self._chat(
                messages=[{"role": "user", "content": compact_prompt}],
                temperature=0.35,
                max_tokens=7000
            )
            parsed = self._safe_json_parse(compact_content)

        return self._normalize_contest_payload(parsed, fallback_tasks_count=safe_tasks_count)

    def generaty_summary(self, subject, klass, theme):
        messages = [
            {"role": "user", "content": None}
        ]

        messages[0]["content"] = (
            f'отвечай с форматированием html (оставь только body) и только по теме вопроса '
            f'(абсолютно без постороннего контента, без побочных надписей). '
            f'все математические формулы пиши строго в latex-нотации в разделителях '
            f'\\( ... \\) для строковых формул и \\[ ... \\] для блочных '
            f'(без двойных экранирований). представь, что ты учитель школьного предмета '
            f'"{subject}" в {klass} классе. твой ученик хочет подготовиться к контрольной '
            f'работе по теме {theme}. пиши максимально понятно, подробно и структурированно. '
            f'сделай конспект длинным и полноценным, чтобы по нему реально можно было готовиться: '
            f'раскрой тему последовательно, не сжимай объяснения до пары фраз, '
            f'чаще используй короткие списки, подзаголовки и блок «самое важное» '
            f'(когда это уместно), добавляй несколько коротких примеров с пояснением, '
            f'обязательно объясняй основные термины, правила, типичные ошибки и важные нюансы по теме. '
            f'если тема большая, разбей её на несколько логических разделов. '
            f'делай больше визуальных выносок отдельными блоками (не называй их "самое важное"): '
            f'для определений, правил, типичных ошибок и советов. '
            f'для таких блоков используй div с классом summary-callout и дополнительным классом: '
            f'summary-callout-definition / summary-callout-rule / summary-callout-warning / summary-callout-tip. '
            f'формулы в эти выноски не помещай — формулы оставляй отдельно обычным текстом/блоками. '
            f'заголовок h1 начинай сразу с темы и класса, без слов «краткий конспект по теме».'
        )

        raw_content = self._chat(
            messages=messages,
            temperature=1,
            max_tokens=4000
        )

        print(raw_content)
        return self.normalize_summary_html(raw_content)

    def answer_question(self, subject, klass, theme, question, history):
        safe_subject = (subject or "").strip()
        safe_theme = (theme or "").strip()
        safe_question = (question or "").strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты дружелюбный, умный и понятный AI-помощник. "
                    "Отвечай естественно, по-человечески и по делу. "
                    "Ты умеешь быть хорошим наставником в учёбе, но не обязан превращать каждый диалог в школьный урок. "
                    "Если пользователь просто здоровается, задаёт обычный вопрос или говорит на свободную тему — отвечай нормально как полезный собеседник. "
                    "Если вопрос учебный, объясняй как сильный учитель: понятно, структурированно, без воды, с примерами. "
                    "Если уместно, возвращай HTML-фрагмент без тегов html/head/body: можно использовать h2, h3, p, ul, ol, li, strong, em, blockquote, pre, code. "
                    "Не используй script, style, iframe. "
                    "Если есть формулы, пиши их в LaTeX: \\( ... \\) или \\[ ... \\]. "
                    "Если нужен код, используй блоки <pre><code class=\"language-...\">...</code></pre>. "
                    "Для class language-... используй базовые языки: python, javascript, java, cpp, csharp, go, sql, bash, html, css, json. "
                    "Не придумывай лишнюю тему, если пользователь её не задавал."
                )
            }
        ]

        if history and isinstance(history, list):
            for item in history:
                if isinstance(item, dict) and item.get("role") and item.get("content"):
                    messages.append({
                        "role": item["role"],
                        "content": item["content"]
                    })

        context_parts = []
        if safe_subject:
            context_parts.append(f"Предмет: {safe_subject}.")
        if klass:
            context_parts.append(f"Класс: {klass}.")
        if safe_theme:
            context_parts.append(f"Текущая тема: {safe_theme}.")

        context_text = " ".join(context_parts)

        user_prompt = (
            f"{context_text}\n"
            f"Сообщение пользователя: {safe_question}\n\n"
            f"Правила ответа:\n"
            f"- если это обычное приветствие или обычный разговорный вопрос, ответь естественно и кратко;\n"
            f"- если это учебный вопрос, ответь понятно, структурированно и полезно;\n"
            f"- не навязывай школьную тему, если пользователь о ней не просил;\n"
            f"- если нужен код, используй pre/code и class language-...;\n"
            f"- если форматирование помогает, используй HTML-фрагмент. Чаще используй HTML и емодзи"
        )

        content = self._chat(
            messages=messages + [{"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=1800
        )

        return self.normalize_html(content)

    def create_test(self, subject, klass, theme):
        test_format_html = render_template('test_format_snippet.html')

        prompt = f"""
Сгенерируй тест по предмету "{subject}" для {klass} класса по теме "{theme}".

Нужно:
- ровно 10 вопросов;
- уровень сложности должен соответствовать {klass} классу;
- вопросы не должны быть слишком примитивными;
- задания должны реально проверять понимание темы, а не только очевидные факты;
- всегда делай 4 варианта ответа в вопросах с выбором;
- у каждого вопроса с выбором должно быть ровно 4 варианта;
- только один правильный вариант;
- в рамках одного вопроса все 4 варианта должны быть текстово разными (дубли запрещены);
- неправильные варианты должны быть правдоподобными, а не абсурдными;
- чередуй типы вопросов: в основном тестовые с 4 вариантами, но можно добавить 2-3 открытых вопроса;
- не делай тест слишком лёгким;
- не добавляй комментарии вне HTML.

Формат HTML должен быть максимально близок к этому шаблону:
{test_format_html}

Дополнительные правила:
1. После каждого вопроса должна быть кнопка "Проверить ответ"
2. Должен быть div для результата проверки
3. Для вопросов с выбором правильный вариант обязан иметь value="correct"
4. Для неправильных вариантов используй value="wrong"
5. Для открытых вопросов добавляй текстовое поле
6. Верни только HTML без markdown-обёртки
"""

        content = self._chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000
        )

        normalized = self.normalize_html(content)
        return self._repair_test_html(normalized, subject, klass, theme)

    def check_answer_with_ai(self, subject, question, user_answer, klass=None, theme=None):
        safe_subject = (subject or "предмет").strip()
        safe_theme = (theme or "").strip()

        prompt = f"""
Ты внимательный и справедливый преподаватель по предмету "{safe_subject}".

Проверь ответ ученика на вопрос.

Контекст:
- Класс: {klass if klass else "не указан"}
- Тема: {safe_theme if safe_theme else "не указана"}

Вопрос:
{question}

Ответ ученика:
{user_answer}

Твои правила проверки:
1. Оцени ответ по смыслу, но без излишней жёсткости.
2. Засчитывай ответ как верный, если по сути мысль правильная:
   допускаются мелкие неточности формулировки, краткость, синонимы и небольшие опечатки.
3. Не требуй дословного совпадения с учебником.
4. Если ключевая идея неверная или ответ уходит от вопроса — is_correct = false.
5. Верни только два варианта итоговой оценки: true или false.
6. Если is_correct = false, НЕ ПИШИ правильный ответ.
7. Если is_correct = false, объясни кратко и понятно, что именно не так.
8. Если is_correct = true, кратко объясни, почему ответ засчитан.
9. Не пиши "примерно правильно".
10. Верни строго JSON и ничего кроме JSON.

Формат ответа:
{{
  "is_correct": true,
  "feedback": "краткое объяснение, почему да или почему нет",
  "correct_answer": ""
}}
"""

        content = self._chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)

        if "is_correct" not in parsed:
            parsed["is_correct"] = False
        if "feedback" not in parsed:
            parsed["feedback"] = "Не удалось корректно проверить ответ."
        if "correct_answer" not in parsed:
            parsed["correct_answer"] = ""

        # Жёстко убираем правильный ответ, если ответ неверный
        if parsed["is_correct"] is False:
            parsed["correct_answer"] = ""

        return parsed
