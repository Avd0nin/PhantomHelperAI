import os
import re
import sqlite3
import json
import html as html_utils
from datetime import datetime, timezone, timedelta
from functools import wraps
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import Flask, request, render_template, jsonify, redirect, session, url_for, g, make_response
from werkzeug.security import generate_password_hash, check_password_hash

from env_loader import load_env


load_env()

from ai_core import AICore
from contest_routes import create_contest_blueprint

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def resolve_app_path(value, default=None):
    raw = (value or "").strip()
    if not raw:
        return default
    if os.path.isabs(raw):
        return raw
    return os.path.join(APP_DIR, raw)


model = AICore()
app = Flask(__name__)
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    raise RuntimeError("SECRET_KEY is not set. Add it to .env.")
app.config["SECRET_KEY"] = secret_key
app.config["DATABASE"] = resolve_app_path(
    os.getenv("DATABASE_PATH"),
    os.path.join(APP_DIR, "app.db")
)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
app.config["SESSION_COOKIE_SECURE"] = env_bool("SESSION_COOKIE_SECURE", False)

USERNAME_REGEX = re.compile(r"^[A-Za-z0-9_]{3,32}$")
DEFAULT_CHAT_THREAD_TITLE = "Новый чат"
try:
    MOSCOW_TZ = ZoneInfo("Europe/Moscow")
except ZoneInfoNotFoundError:
    # Fallback for environments without tzdata installed.
    MOSCOW_TZ = timezone(timedelta(hours=3))


def collapse_spaces(value):
    return re.sub(r"\s+", " ", (value or "")).strip()


def to_moscow_time(value):
    raw = collapse_spaces(str(value or ""))
    if not raw:
        return raw

    parsed = None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        return raw

    parsed_utc = parsed.replace(tzinfo=timezone.utc)
    return parsed_utc.astimezone(MOSCOW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def row_to_dict_with_moscow(row, fields=("created_at", "updated_at", "last_attempt_at", "best_attempt_at")):
    if row is None:
        return None
    data = dict(row)
    for field in fields:
        if field in data and data[field]:
            data[field] = to_moscow_time(data[field])
    return data


def rows_to_dicts_with_moscow(rows, fields=("created_at", "updated_at", "last_attempt_at", "best_attempt_at")):
    return [row_to_dict_with_moscow(row, fields=fields) for row in (rows or [])]


def plural_ru(value, one, few, many):
    number = abs(int(value or 0))
    mod10 = number % 10
    mod100 = number % 100
    if mod10 == 1 and mod100 != 11:
        return one
    if 2 <= mod10 <= 4 and not (12 <= mod100 <= 14):
        return few
    return many


def normalize_contest_difficulty_label(value):
    raw = collapse_spaces(str(value or "")).lower()
    if not raw:
        return "средний"

    numeric_map = {
        "1": "очень легкий",
        "2": "легкий",
        "3": "ниже среднего",
        "4": "средний",
        "5": "средний+",
        "6": "выше среднего",
        "7": "сложный",
        "8": "очень сложный",
        "9": "предолимпиадный",
        "10": "олимпиадный",
    }
    if raw in numeric_map:
        return numeric_map[raw]

    alias_map = {
        "easy": "легкий",
        "medium": "средний",
        "hard": "сложный",
        "olymp": "олимпиадный",
        "очень лёгкий": "очень легкий",
        "очень лёгкий+": "очень легкий",
    }
    if raw in alias_map:
        return alias_map[raw]

    return collapse_spaces(str(value or "средний"))


def is_generic_contest_title(title):
    raw = collapse_spaces(title).lower()
    generic = {
        "",
        "контест",
        "новый контест",
        "contest",
        "new contest",
    }
    if raw in generic:
        return True
    return raw.startswith("контест на ")


def extract_contest_theme(description):
    text = collapse_spaces(description)
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_+\- ]+", " ", text)
    parts = [p for p in cleaned.split() if len(p) >= 3]
    if not parts:
        return ""
    return " ".join(parts[:4])


def build_contest_title(contest_payload, description, difficulty_label, tasks_count):
    payload_title = collapse_spaces(
        (contest_payload or {}).get("contest_title") or
        (contest_payload or {}).get("title") or
        ""
    )
    if payload_title and not is_generic_contest_title(payload_title):
        return payload_title[:160]

    theme = extract_contest_theme(description)
    count = int(tasks_count or 0)
    if count <= 0:
        tasks = (contest_payload or {}).get("tasks") if isinstance((contest_payload or {}).get("tasks"), list) else []
        count = len(tasks)
    count = max(1, count)
    tasks_part = f"{count} {plural_ru(count, 'задача', 'задачи', 'задач')}"
    if theme:
        return f"{theme}: {difficulty_label}, {tasks_part}"[:160]
    return f"{difficulty_label.capitalize()} контест: {tasks_part}"[:160]


def strip_html_tags(raw_html):
    without_tags = re.sub(r"<[^>]+>", " ", raw_html or "")
    return collapse_spaces(without_tags)


PDF_FONT_REGULAR = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"
PDF_FONT_READY = False


def first_existing_path(paths):
    return next((path for path in paths if path and os.path.exists(path)), None)


def ensure_pdf_fonts():
    global PDF_FONT_READY, PDF_FONT_REGULAR, PDF_FONT_BOLD
    if PDF_FONT_READY or not REPORTLAB_AVAILABLE:
        return

    regular_candidates = [
        resolve_app_path(os.getenv("PDF_FONT_REGULAR_PATH")),
        os.path.join(APP_DIR, "static", "fonts", "DejaVuSans.ttf"),
        os.path.join(APP_DIR, "static", "fonts", "NotoSans-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    bold_candidates = [
        resolve_app_path(os.getenv("PDF_FONT_BOLD_PATH")),
        os.path.join(APP_DIR, "static", "fonts", "DejaVuSans-Bold.ttf"),
        os.path.join(APP_DIR, "static", "fonts", "NotoSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/local/share/fonts/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ]

    regular_path = first_existing_path(regular_candidates)
    bold_path = first_existing_path(bold_candidates)

    try:
        if regular_path:
            if "AppSans" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("AppSans", regular_path))
            PDF_FONT_REGULAR = "AppSans"
        if bold_path:
            if "AppSansBold" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("AppSansBold", bold_path))
            PDF_FONT_BOLD = "AppSansBold"
        elif regular_path:
            PDF_FONT_BOLD = PDF_FONT_REGULAR
        if PDF_FONT_REGULAR != "Helvetica":
            pdfmetrics.registerFontFamily(
                PDF_FONT_REGULAR,
                normal=PDF_FONT_REGULAR,
                bold=PDF_FONT_BOLD,
                italic=PDF_FONT_REGULAR,
                boldItalic=PDF_FONT_BOLD,
            )
    except Exception:
        PDF_FONT_REGULAR = "Helvetica"
        PDF_FONT_BOLD = "Helvetica-Bold"

    PDF_FONT_READY = True


LATEX_REPLACEMENTS = {
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\leq": "≤",
    r"\le": "≤",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\neq": "≠",
    r"\ne": "≠",
    r"\approx": "≈",
    r"\pm": "±",
    r"\infty": "∞",
    r"\quad": " ",
    r"\qquad": " ",
    r"\,": " ",
    r"\;": " ",
    r"\:": " ",
    r"\left": "",
    r"\right": "",
}


def is_simple_formula_part(value):
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9.,+\-]+", collapse_spaces(value)))


def format_latex_for_pdf(value):
    text = str(value or "")
    text = text.replace("\xa0", " ")
    text = text.replace(r"\[", "\n").replace(r"\]", "\n")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\\", "\n")

    def replace_wrapped_command(command, replacement):
        nonlocal text
        pattern = re.compile(rf"\\{command}\{{([^{{}}]+)\}}")
        for _ in range(8):
            updated = pattern.sub(lambda match: replacement(format_latex_for_pdf(match.group(1))), text)
            if updated == text:
                break
            text = updated

    def replace_fraction(match):
        numerator = format_latex_for_pdf(match.group(1))
        denominator = format_latex_for_pdf(match.group(2))
        if is_simple_formula_part(numerator) and is_simple_formula_part(denominator):
            return f"{numerator}/{denominator}"
        return f"({numerator})/({denominator})"

    fraction_pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
    for _ in range(12):
        updated = fraction_pattern.sub(replace_fraction, text)
        if updated == text:
            break
        text = updated

    replace_wrapped_command("sqrt", lambda content: f"√({content})")
    replace_wrapped_command("text", lambda content: content)
    replace_wrapped_command("operatorname", lambda content: content)

    for source, target in LATEX_REPLACEMENTS.items():
        text = text.replace(source, target)

    text = re.sub(r"\\begin\{[^{}]+\}", "", text)
    text = re.sub(r"\\end\{[^{}]+\}", "", text)
    text = re.sub(r"\\([A-Za-zА-Яа-яЁё]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    lines = [collapse_spaces(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def pdf_escape_text(value):
    text = html_utils.escape(format_latex_for_pdf(value), quote=False)
    return text.replace("\n", "<br/>")


def clean_pdf_markup(value):
    text = re.sub(r"[ \t\r\f\v]+", " ", str(value or ""))
    text = re.sub(r"\s*<br\s*/>\s*", "<br/>", text)
    text = re.sub(r"(<br/>){3,}", "<br/><br/>", text)
    text = text.strip()
    text = re.sub(r"^(<br/>)+", "", text)
    text = re.sub(r"(<br/>)+$", "", text)
    return text.strip()


def pdf_markup_to_plain(value):
    text = re.sub(r"(?i)<br\s*/?>", "\n", str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return collapse_spaces(html_utils.unescape(text))


PDF_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote"}
PDF_INLINE_TAGS = {
    "strong": ("<b>", "</b>"),
    "b": ("<b>", "</b>"),
    "em": ("<i>", "</i>"),
    "i": ("<i>", "</i>"),
}


class SummaryPdfHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.current = None
        self.context_stack = []
        self.list_stack = []
        self.skip_depth = 0

    def get_attr(self, attrs, name):
        return next((value or "" for key, value in attrs if key.lower() == name), "")

    def in_callout(self):
        return "callout" in self.context_stack

    def ensure_current(self, tag="p"):
        if self.current is None:
            self.start_block(tag)

    def start_block(self, tag):
        self.flush_current()
        block_type = tag
        if tag in {"h4", "h5", "h6"}:
            block_type = "h3"
        if tag == "blockquote" or (self.in_callout() and tag not in {"h1", "h2", "h3"}):
            block_type = "callout"
        if tag == "li":
            block_type = "li"

        level = max(0, len(self.list_stack) - 1)
        bullet = None
        if tag == "li":
            list_context = self.list_stack[-1] if self.list_stack else {"tag": "ul", "index": 0}
            if list_context["tag"] == "ol":
                list_context["index"] += 1
                bullet = f'{list_context["index"]}.'
            else:
                bullet = "•"

        self.current = {
            "tag": tag,
            "type": block_type,
            "parts": [],
            "bullet": bullet,
            "level": level,
        }

    def flush_current(self):
        if self.current is None:
            return

        text = clean_pdf_markup("".join(self.current["parts"]))
        if text:
            self.blocks.append({
                "type": self.current["type"],
                "text": text,
                "bullet": self.current["bullet"],
                "level": self.current["level"],
            })
        self.current = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return

        if tag == "br":
            self.ensure_current()
            self.current["parts"].append("<br/>")
            return
        if tag in {"ul", "ol"}:
            self.flush_current()
            self.list_stack.append({"tag": tag, "index": 0})
            return
        if tag == "div":
            class_name = self.get_attr(attrs, "class")
            self.flush_current()
            self.context_stack.append("callout" if "summary-callout" in class_name else "div")
            return
        if tag in PDF_BLOCK_TAGS:
            if tag == "p" and self.current and self.current["tag"] == "li":
                return
            self.start_block(tag)
            return
        if tag in PDF_INLINE_TAGS:
            self.ensure_current()
            self.current["parts"].append(PDF_INLINE_TAGS[tag][0])

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return

        if tag in PDF_INLINE_TAGS:
            self.ensure_current()
            self.current["parts"].append(PDF_INLINE_TAGS[tag][1])
            return
        if tag in PDF_BLOCK_TAGS:
            if tag == "p" and self.current and self.current["tag"] == "li":
                return
            self.flush_current()
            return
        if tag in {"ul", "ol"}:
            self.flush_current()
            if self.list_stack:
                self.list_stack.pop()
            return
        if tag == "div":
            self.flush_current()
            if self.context_stack:
                self.context_stack.pop()

    def handle_data(self, data):
        if self.skip_depth or not data:
            return
        if not data.strip() and self.current is None:
            return
        self.ensure_current("p")
        formatted = pdf_escape_text(data)
        if not formatted:
            if data.isspace() and self.current["parts"]:
                self.current["parts"].append(" ")
            return
        if data[0].isspace():
            formatted = f" {formatted}"
        if data[-1].isspace():
            formatted = f"{formatted} "
        self.current["parts"].append(formatted)


def html_to_pdf_blocks(raw_html):
    parser = SummaryPdfHTMLParser()
    try:
        parser.feed(str(raw_html or ""))
        parser.close()
        parser.flush_current()
    except Exception:
        plain_text = strip_html_tags(str(raw_html or ""))
        return [{"type": "p", "text": pdf_escape_text(plain_text), "bullet": None, "level": 0}]

    if parser.blocks:
        return parser.blocks

    plain_text = strip_html_tags(str(raw_html or ""))
    if not plain_text:
        return []
    return [{"type": "p", "text": pdf_escape_text(plain_text), "bullet": None, "level": 0}]


def build_pdf_styles():
    body = ParagraphStyle(
        "SummaryBody",
        fontName=PDF_FONT_REGULAR,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#202124"),
        spaceAfter=7,
    )
    return {
        "title": ParagraphStyle(
            "SummaryTitle",
            parent=body,
            fontName=PDF_FONT_BOLD,
            fontSize=18,
            leading=23,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "SummaryMeta",
            parent=body,
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#5f6368"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "SummaryH1",
            parent=body,
            fontName=PDF_FONT_BOLD,
            fontSize=16,
            leading=21,
            spaceBefore=8,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "SummaryH2",
            parent=body,
            fontName=PDF_FONT_BOLD,
            fontSize=13.5,
            leading=18,
            spaceBefore=9,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "SummaryH3",
            parent=body,
            fontName=PDF_FONT_BOLD,
            fontSize=12,
            leading=16,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": body,
        "li": ParagraphStyle(
            "SummaryListItem",
            parent=body,
            leftIndent=18,
            bulletIndent=4,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "SummaryCallout",
            parent=body,
            backColor=colors.HexColor("#F6F8FB"),
            borderColor=colors.HexColor("#D7DEE8"),
            borderWidth=0.7,
            borderPadding=7,
            leftIndent=4,
            rightIndent=4,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "pre": ParagraphStyle(
            "SummaryPre",
            parent=body,
            fontName=PDF_FONT_REGULAR,
            fontSize=9.5,
            leading=12,
            backColor=colors.HexColor("#F6F8FB"),
            borderColor=colors.HexColor("#D7DEE8"),
            borderWidth=0.7,
            borderPadding=6,
            spaceBefore=5,
            spaceAfter=8,
        ),
    }


def append_pdf_paragraph(story, text, style, bullet_text=None):
    try:
        story.append(Paragraph(text, style, bulletText=bullet_text))
    except Exception:
        story.append(Paragraph(pdf_escape_text(pdf_markup_to_plain(text)), style, bulletText=bullet_text))


def build_summary_pdf(summary):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF generator is unavailable")

    ensure_pdf_fonts()

    title = collapse_spaces(summary.get("title") or "Конспект")
    meta = f'{summary.get("subject", "")} • {summary.get("klass", "")} класс • {summary.get("theme", "")}'

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=44,
        bottomMargin=42,
        title=title,
        author="PhantomHelperAI",
    )
    styles = build_pdf_styles()
    story = [
        Paragraph(pdf_escape_text(title), styles["title"]),
        Paragraph(pdf_escape_text(meta), styles["meta"]),
        Spacer(1, 4),
    ]

    skipped_duplicate_h1 = False
    list_styles = {}
    for block in html_to_pdf_blocks(summary.get("content_html") or ""):
        block_type = block.get("type") or "body"
        text = clean_pdf_markup(block.get("text") or "")
        if not text:
            continue

        if not skipped_duplicate_h1 and block_type == "h1":
            skipped_duplicate_h1 = True
            if collapse_spaces(pdf_markup_to_plain(text)).lower() == title.lower():
                continue

        if block_type == "li":
            level = int(block.get("level") or 0)
            if level not in list_styles:
                list_styles[level] = ParagraphStyle(
                    f"SummaryListItem{level}",
                    parent=styles["li"],
                    leftIndent=18 + level * 14,
                    bulletIndent=4 + level * 14,
                )
            append_pdf_paragraph(story, text, list_styles[level], block.get("bullet") or "•")
            continue

        if block_type == "pre":
            story.append(Preformatted(pdf_markup_to_plain(text), styles["pre"]))
            continue

        append_pdf_paragraph(story, text, styles.get(block_type, styles["body"]))

    document.build(story)
    return buffer.getvalue()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db_path = app.config["DATABASE"]
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            theme TEXT NOT NULL,
            klass INTEGER NOT NULL,
            title TEXT NOT NULL,
            content_html TEXT NOT NULL,
            content_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_favorite INTEGER NOT NULL DEFAULT 0,
            is_archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            summary_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (summary_id) REFERENCES summaries(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            theme TEXT NOT NULL,
            klass INTEGER NOT NULL,
            generated_html TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS test_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            correct_count INTEGER NOT NULL,
            duration_sec INTEGER,
            is_final INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS contests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            difficulty TEXT NOT NULL DEFAULT 'medium',
            tasks_count INTEGER NOT NULL DEFAULT 0,
            duration_minutes INTEGER NOT NULL DEFAULT 60,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS contest_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contest_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            score INTEGER NOT NULL,
            solved_count INTEGER NOT NULL,
            total_tasks INTEGER NOT NULL,
            partial_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            attempts_count INTEGER NOT NULL DEFAULT 0,
            time_used_sec INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (contest_id) REFERENCES contests(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_summaries_user_created
            ON summaries(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated
            ON chat_threads(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created
            ON chat_messages(thread_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_tests_user_created
            ON tests(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_test_attempts_test_created
            ON test_attempts(test_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_contests_user_created
            ON contests(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_contest_attempts_contest_created
            ON contest_attempts(contest_id, created_at DESC);
        """
    )
    columns = db.execute("PRAGMA table_info(test_attempts)").fetchall()
    column_names = {str(row["name"]).lower() for row in columns}
    if "is_final" not in column_names:
        db.execute("ALTER TABLE test_attempts ADD COLUMN is_final INTEGER NOT NULL DEFAULT 0")

    chat_columns = db.execute("PRAGMA table_info(chat_threads)").fetchall()
    chat_column_names = {str(row["name"]).lower() for row in chat_columns}
    if "summary_id" not in chat_column_names:
        db.execute("ALTER TABLE chat_threads ADD COLUMN summary_id INTEGER")

    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_threads_summary ON chat_threads(summary_id, user_id)"
    )
    db.commit()


def get_user_by_username(username):
    return get_db().execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,)
    ).fetchone()


def get_user_by_id(user_id):
    row = get_db().execute(
        "SELECT id, username, created_at FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    return row_to_dict_with_moscow(row, fields=("created_at",))


def current_user_id():
    user = g.get("current_user")
    return int(user["id"]) if user else None


def is_safe_next_url(next_url):
    if not next_url:
        return False
    parsed = urlparse(next_url)
    return parsed.scheme == "" and parsed.netloc == "" and next_url.startswith("/")


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.get("current_user") is not None:
            return view(*args, **kwargs)

        if request.path.startswith("/api/"):
            return jsonify({"error": "Требуется авторизация"}), 401

        next_url = request.path
        if request.query_string:
            next_url = f"{request.path}?{request.query_string.decode('utf-8')}"
        return redirect(url_for("login", next=next_url))

    return wrapped_view


@app.before_request
def load_current_user():
    g.current_user = None
    user_id = session.get("user_id")
    if not user_id:
        return

    user = get_user_by_id(user_id)
    if user is None:
        session.clear()
        return
    g.current_user = user


@app.context_processor
def inject_current_user():
    return {"current_user": g.get("current_user")}


def validate_username(username):
    if not username:
        return "Логин обязателен."
    if not USERNAME_REGEX.fullmatch(username):
        return "Логин: 3-32 символа, только латинские буквы, цифры и _."
    return None


def validate_password(password):
    if not password:
        return "Пароль обязателен."
    if len(password) < 8:
        return "Пароль должен содержать минимум 8 символов."
    return None


def build_summary_title(subject, theme, klass, content_html):
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", content_html or "", flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        h1_text = strip_html_tags(h1_match.group(1))
        if h1_text:
            return h1_text[:120]

    safe_subject = collapse_spaces(subject) or "Предмет"
    safe_theme = collapse_spaces(theme) or "Тема"
    return f"{safe_subject}: {safe_theme} ({klass} класс)"[:120]


def build_summary_chat_title(summary):
    summary_title = collapse_spaces(summary["title"]) if summary else "Конспект"
    return f"Чат к конспекту: {summary_title}"[:160]


def build_summary_chat_welcome(summary):
    if not summary:
        return "Задавайте любые вопросы по этому конспекту."
    summary_title = collapse_spaces(summary["title"])
    return (
        f"Это чат по конспекту «{summary_title}». "
        f"Задавайте любые вопросы по теме — разберём шаг за шагом."
    )


def build_summary_system_context(summary):
    if not summary:
        return ""
    summary_text = strip_html_tags(summary.get("content_html") or "")
    summary_text = summary_text[:12000]
    summary_title = collapse_spaces(summary.get("title") or "Конспект")
    return (
        "Ты отвечаешь в режиме чата по конкретному конспекту. "
        "Опирайся в первую очередь на этот конспект, объясняй понятно и по теме. "
        "Если вопрос вне конспекта, мягко скажи об этом и предложи связанный разбор.\n\n"
        f"Название конспекта: {summary_title}\n"
        f"Текст конспекта:\n{summary_text}"
    )


def save_summary_for_user(user_id, subject, theme, klass, content_html):
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO summaries (user_id, subject, theme, klass, title, content_html, content_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            collapse_spaces(subject),
            collapse_spaces(theme),
            int(klass),
            build_summary_title(subject, theme, klass, content_html),
            content_html,
            strip_html_tags(content_html)
        )
    )
    db.commit()
    return int(cursor.lastrowid)


def build_chat_thread_title(first_message):
    message = collapse_spaces(first_message)
    if not message:
        return DEFAULT_CHAT_THREAD_TITLE
    return f"{message[:87]}..." if len(message) > 90 else message


def create_chat_thread_for_user(user_id, first_message="", summary_id=None, title_override=None):
    title = collapse_spaces(title_override) if title_override else ""
    if not title:
        title = build_chat_thread_title(first_message)

    db = get_db()
    cursor = db.execute(
        "INSERT INTO chat_threads (user_id, title, summary_id) VALUES (?, ?, ?)",
        (user_id, title, summary_id)
    )
    db.commit()
    return int(cursor.lastrowid)


def get_chat_thread_for_user(user_id, thread_id):
    return get_db().execute(
        "SELECT id, user_id, title, summary_id, created_at, updated_at FROM chat_threads WHERE id = ? AND user_id = ? AND is_archived = 0",
        (thread_id, user_id)
    ).fetchone()


def get_summary_chat_thread_for_user(user_id, summary_id):
    return get_db().execute(
        """
        SELECT id, user_id, title, summary_id, created_at, updated_at
        FROM chat_threads
        WHERE user_id = ? AND summary_id = ? AND is_archived = 0
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """,
        (user_id, summary_id)
    ).fetchone()


def rename_chat_thread_if_default(thread_id, first_message):
    candidate_title = build_chat_thread_title(first_message)
    if candidate_title == DEFAULT_CHAT_THREAD_TITLE:
        return

    db = get_db()
    db.execute(
        """
        UPDATE chat_threads
        SET title = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND title = ?
        """,
        (candidate_title, thread_id, DEFAULT_CHAT_THREAD_TITLE)
    )
    db.commit()


def append_chat_message(thread_id, role, content):
    db = get_db()
    db.execute(
        "INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content)
    )
    db.execute(
        "UPDATE chat_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (thread_id,)
    )
    db.commit()


def get_chat_messages(thread_id, limit=200):
    return get_db().execute(
        """
        SELECT role, content, created_at
        FROM chat_messages
        WHERE thread_id = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (thread_id, limit)
    ).fetchall()


def save_test_for_user(user_id, subject, theme, klass, generated_html):
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO tests (user_id, subject, theme, klass, generated_html)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            collapse_spaces(subject),
            collapse_spaces(theme),
            int(klass),
            generated_html
        )
    )
    db.commit()
    return int(cursor.lastrowid)


def get_test_for_user(test_id, user_id):
    row = get_db().execute(
        """
        SELECT id, user_id, subject, theme, klass, generated_html, created_at
        FROM tests
        WHERE id = ? AND user_id = ? AND is_archived = 0
        """,
        (test_id, user_id)
    ).fetchone()
    return row_to_dict_with_moscow(row, fields=("created_at",))


def save_test_attempt(test_id, user_id, score, total_questions, correct_count, duration_sec=None, is_final=0):
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO test_attempts (test_id, user_id, score, total_questions, correct_count, duration_sec, is_final)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (test_id, user_id, score, total_questions, correct_count, duration_sec, int(1 if is_final else 0))
    )
    db.commit()
    return int(cursor.lastrowid)


def get_profile_summaries(user_id, limit=25):
    rows = get_db().execute(
        """
        SELECT id, title, subject, theme, klass, created_at
        FROM summaries
        WHERE user_id = ? AND is_archived = 0
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    ).fetchall()
    return rows_to_dicts_with_moscow(rows, fields=("created_at",))


def get_profile_chat_threads(user_id, limit=25):
    rows = get_db().execute(
        """
        SELECT
            ct.id,
            ct.title,
            ct.created_at,
            ct.updated_at,
            COUNT(cm.id) AS message_count
        FROM chat_threads ct
        LEFT JOIN chat_messages cm ON cm.thread_id = ct.id
        WHERE ct.user_id = ? AND ct.is_archived = 0
        GROUP BY ct.id
        ORDER BY ct.updated_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    ).fetchall()
    return rows_to_dicts_with_moscow(rows, fields=("created_at", "updated_at"))


def get_profile_tests(user_id, limit=25):
    rows = get_db().execute(
        """
        SELECT
            t.id,
            t.subject,
            t.theme,
            t.klass,
            t.created_at,
            (SELECT COUNT(*) FROM test_attempts a WHERE a.test_id = t.id AND a.is_final = 1) AS attempts_count,
            (SELECT a.score FROM test_attempts a WHERE a.test_id = t.id AND a.is_final = 1 ORDER BY a.created_at DESC, a.id DESC LIMIT 1) AS last_score,
            (SELECT a.correct_count FROM test_attempts a WHERE a.test_id = t.id AND a.is_final = 1 ORDER BY a.created_at DESC, a.id DESC LIMIT 1) AS last_correct_count,
            (SELECT a.total_questions FROM test_attempts a WHERE a.test_id = t.id AND a.is_final = 1 ORDER BY a.created_at DESC, a.id DESC LIMIT 1) AS last_total_questions,
            (SELECT a.created_at FROM test_attempts a WHERE a.test_id = t.id AND a.is_final = 1 ORDER BY a.created_at DESC, a.id DESC LIMIT 1) AS last_attempt_at
        FROM tests t
        WHERE t.user_id = ? AND t.is_archived = 0
        ORDER BY t.created_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    ).fetchall()
    return rows_to_dicts_with_moscow(rows, fields=("created_at", "last_attempt_at"))


def get_summary_for_user(summary_id, user_id):
    row = get_db().execute(
        """
        SELECT id, title, subject, theme, klass, content_html, created_at
        FROM summaries
        WHERE id = ? AND user_id = ? AND is_archived = 0
        """,
        (summary_id, user_id)
    ).fetchone()
    return row_to_dict_with_moscow(row, fields=("created_at",))


def get_test_attempts_for_user(test_id, user_id, limit=30):
    rows = get_db().execute(
        """
        SELECT id, score, total_questions, correct_count, duration_sec, created_at
        FROM test_attempts
        WHERE test_id = ? AND user_id = ? AND is_final = 1
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (test_id, user_id, limit)
    ).fetchall()
    return rows_to_dicts_with_moscow(rows, fields=("created_at",))


def save_contest_for_user(user_id, payload, description="", difficulty="medium", tasks_count=0, duration_minutes=60):
    contest_payload = payload if isinstance(payload, dict) else {}
    tasks = contest_payload.get("tasks") if isinstance(contest_payload.get("tasks"), list) else []
    difficulty_label = normalize_contest_difficulty_label(difficulty)
    title = build_contest_title(
        contest_payload=contest_payload,
        description=description,
        difficulty_label=difficulty_label,
        tasks_count=(int(tasks_count) if tasks_count else len(tasks)),
    )
    contest_payload["contest_title"] = title
    contest_payload["difficulty_label"] = difficulty_label

    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO contests (user_id, title, description, difficulty, tasks_count, duration_minutes, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            title[:160],
            collapse_spaces(description or ""),
            difficulty_label,
            int(tasks_count) if tasks_count else len(tasks),
            max(15, int(duration_minutes or 60)),
            json.dumps(contest_payload, ensure_ascii=False),
        ),
    )
    db.commit()
    return int(cursor.lastrowid)


def save_contest_attempt(
        contest_id,
        user_id,
        score,
        solved_count,
        total_tasks,
        partial_count=0,
        failed_count=0,
        attempts_count=0,
        time_used_sec=0,
):
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO contest_attempts (
            contest_id, user_id, score, solved_count, total_tasks,
            partial_count, failed_count, attempts_count, time_used_sec
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contest_id,
            user_id,
            int(score),
            int(solved_count),
            int(total_tasks),
            int(partial_count),
            int(failed_count),
            int(attempts_count),
            int(time_used_sec),
        ),
    )
    db.commit()
    return int(cursor.lastrowid)


def get_contest_for_user(contest_id, user_id):
    row = get_db().execute(
        """
        SELECT id, user_id, title, description, difficulty, tasks_count, duration_minutes, payload_json, created_at
        FROM contests
        WHERE id = ? AND user_id = ? AND is_archived = 0
        """,
        (contest_id, user_id),
    ).fetchone()
    return row_to_dict_with_moscow(row, fields=("created_at",))


def get_contest_attempts_for_user(contest_id, user_id, limit=30):
    rows = get_db().execute(
        """
        SELECT
            id, score, solved_count, total_tasks, partial_count, failed_count,
            attempts_count, time_used_sec, created_at
        FROM contest_attempts
        WHERE contest_id = ? AND user_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (contest_id, user_id, limit),
    ).fetchall()
    return rows_to_dicts_with_moscow(rows, fields=("created_at",))


def get_profile_contests(user_id, limit=25):
    rows = get_db().execute(
        """
        SELECT
            c.id,
            c.title,
            c.description,
            c.difficulty,
            c.tasks_count,
            c.duration_minutes,
            c.created_at,
            (SELECT COUNT(*) FROM contest_attempts a WHERE a.contest_id = c.id) AS attempts_count,
            (
                SELECT a.score
                FROM contest_attempts a
                WHERE a.contest_id = c.id
                ORDER BY a.score DESC, a.solved_count DESC, a.time_used_sec ASC, a.created_at DESC, a.id DESC
                LIMIT 1
            ) AS best_score,
            (
                SELECT a.solved_count
                FROM contest_attempts a
                WHERE a.contest_id = c.id
                ORDER BY a.score DESC, a.solved_count DESC, a.time_used_sec ASC, a.created_at DESC, a.id DESC
                LIMIT 1
            ) AS best_solved_count,
            (
                SELECT a.total_tasks
                FROM contest_attempts a
                WHERE a.contest_id = c.id
                ORDER BY a.score DESC, a.solved_count DESC, a.time_used_sec ASC, a.created_at DESC, a.id DESC
                LIMIT 1
            ) AS best_total_tasks,
            (
                SELECT a.created_at
                FROM contest_attempts a
                WHERE a.contest_id = c.id
                ORDER BY a.score DESC, a.solved_count DESC, a.time_used_sec ASC, a.created_at DESC, a.id DESC
                LIMIT 1
            ) AS best_attempt_at
        FROM contests c
        WHERE c.user_id = ? AND c.is_archived = 0
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return rows_to_dicts_with_moscow(rows, fields=("created_at", "best_attempt_at"))


def serialize_contest_row(row):
    if row is None:
        return None
    data = dict(row)
    data["difficulty_label"] = normalize_contest_difficulty_label(data.get("difficulty"))
    return data


def get_profile_stats(user_id):
    db = get_db()

    def row_int(row, key):
        if row is None:
            return 0
        try:
            return int(row[key] or 0)
        except (KeyError, TypeError, ValueError):
            return 0

    summaries_count = db.execute(
        "SELECT COUNT(*) AS value FROM summaries WHERE user_id = ? AND is_archived = 0",
        (user_id,),
    ).fetchone()["value"]

    chats_row = db.execute(
        """
        SELECT
            COUNT(DISTINCT ct.id) AS threads_count,
            COUNT(cm.id) AS messages_count
        FROM chat_threads ct
        LEFT JOIN chat_messages cm ON cm.thread_id = ct.id
        WHERE ct.user_id = ? AND ct.is_archived = 0
        """,
        (user_id,),
    ).fetchone()

    tests_row = db.execute(
        """
        SELECT
            COUNT(DISTINCT t.id) AS tests_count,
            COUNT(CASE WHEN a.is_final = 1 THEN a.id END) AS attempts_count,
            COALESCE(MAX(CASE WHEN a.is_final = 1 THEN a.score END), 0) AS best_score,
            COALESCE(ROUND(AVG(CASE WHEN a.is_final = 1 THEN a.score END)), 0) AS avg_score
        FROM tests t
        LEFT JOIN test_attempts a ON a.test_id = t.id
        WHERE t.user_id = ? AND t.is_archived = 0
        """,
        (user_id,),
    ).fetchone()

    contests_row = db.execute(
        """
        SELECT
            COUNT(DISTINCT c.id) AS contests_count,
            COUNT(a.id) AS attempts_count,
            COALESCE(MAX(a.score), 0) AS best_score,
            COALESCE(SUM(a.solved_count), 0) AS solved_total
        FROM contests c
        LEFT JOIN contest_attempts a ON a.contest_id = c.id
        WHERE c.user_id = ? AND c.is_archived = 0
        """,
        (user_id,),
    ).fetchone()

    return {
        "summaries_count": int(summaries_count or 0),
        "chat_threads_count": row_int(chats_row, "threads_count"),
        "chat_messages_count": row_int(chats_row, "messages_count"),
        "tests_count": row_int(tests_row, "tests_count"),
        "test_attempts_count": row_int(tests_row, "attempts_count"),
        "test_best_score": row_int(tests_row, "best_score"),
        "test_avg_score": row_int(tests_row, "avg_score"),
        "contests_count": row_int(contests_row, "contests_count"),
        "contest_attempts_count": row_int(contests_row, "attempts_count"),
        "contest_best_score": row_int(contests_row, "best_score"),
        "contest_solved_total": row_int(contests_row, "solved_total"),
    }


app.register_blueprint(
    create_contest_blueprint(
        model,
        save_contest_callback=save_contest_for_user,
        current_user_id_callback=current_user_id,
    )
)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if g.get("current_user") is not None:
        return redirect(url_for("home"))

    error = None
    username = ""
    next_url = (request.args.get("next") or "").strip()

    if request.method == 'POST':
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        password_confirm = request.form.get("password_confirm") or ""
        next_url = (request.form.get("next") or "").strip()

        error = validate_username(username) or validate_password(password)

        if not error and password != password_confirm:
            error = "Пароли не совпадают."

        if not error:
            try:
                db = get_db()
                db.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password))
                )
                db.commit()
            except sqlite3.IntegrityError:
                error = "Пользователь с таким логином уже существует."

        if not error:
            user = get_user_by_username(username)
            session.clear()
            session["user_id"] = int(user["id"])
            if is_safe_next_url(next_url):
                return redirect(next_url)
            return redirect(url_for("home"))

    return render_template(
        "register.html",
        error=error,
        username=username,
        next_url=next_url
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.get("current_user") is not None:
        return redirect(url_for("home"))

    error = None
    username = ""
    next_url = (request.args.get("next") or "").strip()

    if request.method == 'POST':
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        next_url = (request.form.get("next") or "").strip()

        if not username or not password:
            error = "Введите логин и пароль."
        else:
            user = get_user_by_username(username)
            if user is None or not check_password_hash(user["password_hash"], password):
                error = "Неверный логин или пароль."
            else:
                session.clear()
                session["user_id"] = int(user["id"])
                if is_safe_next_url(next_url):
                    return redirect(next_url)
                return redirect(url_for("home"))

    return render_template(
        "login.html",
        error=error,
        username=username,
        next_url=next_url
    )


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route('/make_summary')
@login_required
def make_summary():
    return render_template('generate_summary.html')


@app.route('/help', methods=['GET'])
def info():
    return render_template('help.html')


@app.route('/chat', methods=['GET'])
@login_required
def chat():
    thread_id = request.args.get('thread_id', type=int)
    initial_thread_id = None
    initial_messages = []
    chat_welcome_message = "Привет! Я помогу с учебой: объясню тему, дам план подготовки, сделаю краткий разбор и примеры. С чего начнем?"

    if thread_id:
        thread = get_chat_thread_for_user(current_user_id(), thread_id)
        if thread:
            initial_thread_id = int(thread['id'])
            summary_id = thread["summary_id"]
            if summary_id:
                summary = get_summary_for_user(int(summary_id), current_user_id())
                if summary:
                    chat_welcome_message = build_summary_chat_welcome(summary)
            rows = get_chat_messages(initial_thread_id, limit=250)
            initial_messages = [
                {
                    "role": row["role"],
                    "content": row["content"]
                }
                for row in rows
            ]

    return render_template(
        'chat.html',
        initial_thread_id=initial_thread_id,
        initial_messages=initial_messages,
        chat_welcome_message=chat_welcome_message
    )


@app.route('/make_test', methods=['GET'])
@login_required
def make_test():
    mode = str(request.args.get('mode', 'test')).strip().lower()
    initial_mode = 'contest' if mode == 'contest' else 'test'
    initial_test_id = request.args.get('test_id', type=int)
    initial_contest_id = request.args.get('contest_id', type=int)
    return render_template(
        'generate_test.html',
        initial_mode=initial_mode,
        initial_test_id=initial_test_id,
        initial_contest_id=initial_contest_id,
    )


@app.route('/make_contest', methods=['GET'])
@login_required
def make_contest():
    return redirect(url_for('make_test', mode='contest'))


@app.route('/profile', methods=['GET'])
@login_required
def profile():
    tab = (request.args.get('tab') or 'summaries').strip().lower()
    if tab not in {'summaries', 'chats', 'tests', 'contests'}:
        tab = 'summaries'

    user_id = current_user_id()
    deleted = str(request.args.get("deleted", "")).strip() == "1"
    delete_error = str(request.args.get("delete_error", "")).strip() == "1"

    contests_rows = get_profile_contests(user_id)
    contests = [serialize_contest_row(row) for row in contests_rows]

    return render_template(
        'profile.html',
        active_tab=tab,
        deleted=deleted,
        delete_error=delete_error,
        profile_stats=get_profile_stats(user_id),
        summaries=get_profile_summaries(user_id),
        chats=get_profile_chat_threads(user_id),
        tests=get_profile_tests(user_id),
        contests=contests,
    )


@app.route('/profile/delete_data', methods=['POST'])
@login_required
def delete_profile_data():
    confirmation = collapse_spaces(request.form.get("confirm_text", "")).lower()
    if confirmation not in {"удалить", "delete"}:
        return redirect(url_for("profile", tab="summaries", delete_error=1))

    user_id = current_user_id()
    db = get_db()
    db.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM chat_threads WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM tests WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM contests WHERE user_id = ?", (user_id,))
    db.commit()
    return redirect(url_for("profile", tab="summaries", deleted=1))


@app.route('/profile/summary/<int:summary_id>', methods=['GET'])
@login_required
def view_summary(summary_id):
    summary = get_summary_for_user(summary_id, current_user_id())
    if summary is None:
        return redirect(url_for('profile', tab='summaries'))

    return render_template('profile_summary.html', summary=summary)


@app.route('/profile/summary/<int:summary_id>/download_pdf', methods=['GET'])
@login_required
def download_summary_pdf(summary_id):
    summary = get_summary_for_user(summary_id, current_user_id())
    if summary is None:
        return redirect(url_for('profile', tab='summaries'))

    try:
        pdf_bytes = build_summary_pdf(summary)
    except Exception as error:
        print(f"Ошибка генерации PDF: {error}")
        return redirect(url_for('view_summary', summary_id=summary_id))

    filename = f"summary_{int(summary_id)}.pdf"
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@app.route('/profile/summary/<int:summary_id>/chat', methods=['GET'])
@login_required
def open_summary_chat(summary_id):
    user_id = current_user_id()
    summary = get_summary_for_user(summary_id, user_id)
    if summary is None:
        return redirect(url_for('profile', tab='summaries'))

    thread = get_summary_chat_thread_for_user(user_id, summary_id)
    if thread:
        thread_id = int(thread["id"])
    else:
        thread_id = create_chat_thread_for_user(
            user_id=user_id,
            summary_id=summary_id,
            title_override=build_summary_chat_title(summary),
        )
        append_chat_message(thread_id, 'assistant', build_summary_chat_welcome(summary))

    return redirect(url_for('chat', thread_id=thread_id))


@app.route('/profile/test/<int:test_id>', methods=['GET'])
@login_required
def view_test(test_id):
    user_id = current_user_id()
    test = get_test_for_user(test_id, user_id)
    if test is None:
        return redirect(url_for('profile', tab='tests'))

    attempts = get_test_attempts_for_user(test_id, user_id)
    return render_template('profile_test.html', test=test, attempts=attempts)


@app.route('/profile/contest/<int:contest_id>', methods=['GET'])
@login_required
def view_contest(contest_id):
    user_id = current_user_id()
    contest = get_contest_for_user(contest_id, user_id)
    if contest is None:
        return redirect(url_for('profile', tab='contests'))

    attempts = get_contest_attempts_for_user(contest_id, user_id)
    return render_template('profile_contest.html', contest=serialize_contest_row(contest), attempts=attempts)


@app.route('/api/question', methods=['POST'])
@login_required
def asking():
    data = request.get_json(silent=True) or {}
    subject = collapse_spaces(data.get('subject'))
    theme = collapse_spaces(data.get('theme'))
    question = collapse_spaces(data.get('question'))
    history = data.get('message_history', [])

    if not question:
        return jsonify({'error': 'Пустой вопрос'}), 400

    try:
        klass = int(data.get('klass', 6))
    except (TypeError, ValueError):
        klass = 6

    user_id = current_user_id()
    thread_id = data.get('thread_id')
    try:
        thread_id = int(thread_id) if thread_id is not None else None
    except (TypeError, ValueError):
        thread_id = None

    thread = get_chat_thread_for_user(user_id, thread_id) if thread_id else None
    if thread is None:
        thread_id = create_chat_thread_for_user(user_id, question)
    else:
        if not thread["summary_id"]:
            rename_chat_thread_if_default(thread_id, question)

    append_chat_message(thread_id, 'user', question)

    try:
        history_for_model = history if isinstance(history, list) else []
        if thread and thread["summary_id"]:
            summary = get_summary_for_user(int(thread["summary_id"]), user_id)
            if summary:
                subject = summary["subject"]
                theme = summary["theme"]
                try:
                    klass = int(summary["klass"])
                except (TypeError, ValueError):
                    klass = 6
                summary_system_context = build_summary_system_context(summary)
                history_for_model = [{"role": "system", "content": summary_system_context}] + history_for_model

        answer = model.answer_question(subject, klass, theme, question, history_for_model)
    except Exception as e:
        print(f"Ошибка чата: {str(e)}")
        return jsonify({
            'error': 'Ошибка при получении ответа от модели.',
            'thread_id': thread_id
        }), 500

    append_chat_message(thread_id, 'assistant', answer)

    return jsonify({
        'answer': answer,
        'thread_id': thread_id
    })


@app.route('/api/test')
@login_required
def generate_test():
    subject = request.args.get('subject')
    theme = request.args.get('theme')

    try:
        klass = int(request.args.get('class'))
    except (TypeError, ValueError):
        return "Некорректный класс", 400

    if klass < 1 or klass > 11:
        return f"Некорректный класс (должен быть от 1 до 11), у вас {klass}", 400

    try:
        generated_html = model.create_test(subject, klass, theme).lstrip('```html\n').rstrip('\n```')
        test_id = save_test_for_user(current_user_id(), subject, theme, klass, generated_html)
        response = make_response(generated_html)
        response.headers['X-Test-Id'] = str(test_id)
        return response
    except Exception as e:
        print(f"Ошибка генерации теста: {str(e)}")
        return "<p>Ошибка при генерации теста.</p>", 500


@app.route('/api/test_saved/<int:test_id>', methods=['GET'])
@login_required
def get_saved_test(test_id):
    test = get_test_for_user(test_id, current_user_id())
    if test is None:
        return jsonify({"error": "Тест не найден"}), 404
    return jsonify(
        {
            "id": int(test["id"]),
            "subject": test["subject"],
            "theme": test["theme"],
            "klass": int(test["klass"]),
            "generated_html": test["generated_html"],
            "created_at": test["created_at"],
        }
    )


@app.route('/api/test_attempt', methods=['POST'])
@login_required
def save_attempt():
    data = request.get_json(silent=True) or {}

    try:
        test_id = int(data.get('test_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Некорректный test_id'}), 400

    user_id = current_user_id()
    test = get_test_for_user(test_id, user_id)
    if test is None:
        return jsonify({'error': 'Тест не найден'}), 404

    try:
        score = int(data.get('score', 0))
        total_questions = int(data.get('total_questions', 0))
        correct_count = int(data.get('correct_count', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Некорректные числовые значения'}), 400

    if total_questions <= 0:
        return jsonify({'error': 'total_questions должен быть больше 0'}), 400
    if correct_count < 0 or correct_count > total_questions:
        return jsonify({'error': 'correct_count вне диапазона'}), 400

    score = max(0, min(score, 100))

    duration_sec = data.get('duration_sec')
    if duration_sec is not None:
        try:
            duration_sec = int(duration_sec)
        except (TypeError, ValueError):
            duration_sec = None
        if duration_sec is not None and duration_sec < 0:
            duration_sec = None
    is_final_raw = data.get('is_final', False)
    if isinstance(is_final_raw, str):
        is_final = is_final_raw.strip().lower() in {"1", "true", "yes", "y", "final"}
    else:
        is_final = bool(is_final_raw)

    # Не считаем попытку, пока тест не завершён пользователем.
    if not is_final:
        return jsonify({'ok': True, 'attempt_id': None, 'skipped': 'non_final'})

    attempt_id = save_test_attempt(
        test_id=test_id,
        user_id=user_id,
        score=score,
        total_questions=total_questions,
        correct_count=correct_count,
        duration_sec=duration_sec,
        is_final=is_final
    )

    return jsonify({'ok': True, 'attempt_id': attempt_id})


@app.route('/api/contest_saved/<int:contest_id>', methods=['GET'])
@login_required
def get_saved_contest(contest_id):
    contest = get_contest_for_user(contest_id, current_user_id())
    if contest is None:
        return jsonify({"error": "Контест не найден"}), 404

    try:
        payload = json.loads(contest["payload_json"] or "{}")
    except json.JSONDecodeError:
        payload = {}

    payload["contest_id"] = int(contest["id"])
    payload["contest_title"] = collapse_spaces(payload.get("contest_title") or contest["title"])
    payload["difficulty_label"] = normalize_contest_difficulty_label(contest["difficulty"])
    return jsonify(
        {
            "contest": payload,
            "duration_minutes": int(contest["duration_minutes"] or 60),
            "title": contest["title"],
            "difficulty_label": normalize_contest_difficulty_label(contest["difficulty"]),
            "created_at": contest["created_at"],
        }
    )


@app.route('/api/contest_attempt', methods=['POST'])
@login_required
def save_contest_attempt_api():
    data = request.get_json(silent=True) or {}

    try:
        contest_id = int(data.get('contest_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Некорректный contest_id'}), 400

    user_id = current_user_id()
    contest = get_contest_for_user(contest_id, user_id)
    if contest is None:
        return jsonify({'error': 'Контест не найден'}), 404

    required_int_fields = [
        "score",
        "solved_count",
        "total_tasks",
        "partial_count",
        "failed_count",
        "attempts_count",
        "time_used_sec",
    ]
    parsed = {}
    for field in required_int_fields:
        try:
            parsed[field] = int(data.get(field, 0))
        except (TypeError, ValueError):
            return jsonify({'error': f'Некорректное поле: {field}'}), 400

    if parsed["total_tasks"] <= 0:
        return jsonify({'error': 'total_tasks должен быть больше 0'}), 400
    if parsed["solved_count"] < 0 or parsed["solved_count"] > parsed["total_tasks"]:
        return jsonify({'error': 'solved_count вне диапазона'}), 400
    if parsed["partial_count"] < 0 or parsed["failed_count"] < 0:
        return jsonify({'error': 'partial_count/failed_count не могут быть отрицательными'}), 400
    if parsed["score"] < 0:
        parsed["score"] = 0
    if parsed["score"] > 100:
        parsed["score"] = 100
    if parsed["time_used_sec"] < 0:
        parsed["time_used_sec"] = 0

    attempt_id = save_contest_attempt(
        contest_id=contest_id,
        user_id=user_id,
        score=parsed["score"],
        solved_count=parsed["solved_count"],
        total_tasks=parsed["total_tasks"],
        partial_count=parsed["partial_count"],
        failed_count=parsed["failed_count"],
        attempts_count=parsed["attempts_count"],
        time_used_sec=parsed["time_used_sec"],
    )
    return jsonify({"ok": True, "attempt_id": attempt_id})


@app.route('/api/check_answer', methods=['POST'])
@login_required
def check_answer():
    try:
        data = request.get_json(silent=True) or {}
        question = collapse_spaces(data.get('question'))
        user_answer = collapse_spaces(data.get('answer'))
        subject = collapse_spaces(data.get('subject'))
        theme = collapse_spaces(data.get('theme'))

        try:
            klass = int(data.get('klass', 6))
        except (TypeError, ValueError):
            klass = 6

        if not question or not user_answer:
            return jsonify({
                "is_correct": False,
                "feedback": "Вопрос и ответ не могут быть пустыми.",
                "correct_answer": ""
            })

        checked_answer = model.check_answer_with_ai(
            subject, question, user_answer, klass, theme)

        if not all(key in checked_answer for key in ['is_correct', 'feedback', 'correct_answer']):
            raise ValueError("Некорректный формат ответа API")

        return jsonify(checked_answer)

    except Exception as e:
        print(f"Ошибка проверки ответа: {str(e)}")
        return jsonify({
            "is_correct": False,
            "feedback": "Ошибка проверки. Попробуйте ещё раз.",
            "correct_answer": ""
        })


@app.route('/api/summary')
@login_required
def action():
    subject = request.args.get('subject')
    theme = request.args.get('theme')

    try:
        klass = int(request.args.get('klass'))
    except (TypeError, ValueError):
        return "Некорректный класс", 400

    if klass < 1 or klass > 11:
        return f"Некорректный класс (должен быть от 1 до 11), у вас {klass}", 400

    try:
        generated_html = model.generaty_summary(subject, klass, theme)
        summary_id = save_summary_for_user(current_user_id(), subject, theme, klass, generated_html)
        response = make_response(generated_html)
        response.headers['X-Summary-Id'] = str(summary_id)
        return response
    except Exception as e:
        print(f"Ошибка генерации конспекта: {str(e)}")
        return render_template('summary_example_snippet.html')


with app.app_context():
    init_db()

if __name__ == '__main__':
    app.run(
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=env_int("APP_PORT", 4000),
        debug=env_bool("FLASK_DEBUG", True),
    )
