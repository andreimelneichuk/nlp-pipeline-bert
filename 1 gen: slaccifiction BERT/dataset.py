import asyncio
import csv
import os
import logging
import time
from urllib.parse import urljoin, urlparse, urlunparse
from playwright.async_api import async_playwright, Error as PlaywrightError
from bs4 import BeautifulSoup

# === Настройки логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("docs_crawler")

# === Настройки ===
BASE_URL = "https://docs.example.com/index.html"
OUTPUT_FILE = "output/documentation_dataset.csv"
ERRORS_FILE = "output/crawler_errors.csv"
MAX_CONCURRENT = 1  # Уменьшаем для надёжности
HEADLESS = True

# Игнорируемые расширения
IGNORED_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip',
    '.exe', '.mp4', '.css', '.js', '.ico', '.svg'
}

visited_normalized = set()

def normalize_url(url):
    try:
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = urljoin(BASE_URL, url)
        parsed = urlparse(url)
        clean_url = urlunparse(parsed._replace(fragment="", query=""))
        return clean_url.rstrip('/').lower()
    except:
        return None

def is_valid_doc_url(url, base_domain):
    normalized = normalize_url(url)
    if not normalized:
        return False
    if not (normalized.startswith(f"https://{base_domain}") or normalized.startswith(f"http://{base_domain}")):
        return False
    if any(normalized.endswith(ext) for ext in IGNORED_EXTENSIONS):
        return False
    return True

async def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    selectors = ['main', 'article', '.content', '.main-content', '#main', '.page-content']
    for sel in selectors:
        element = soup.select_one(sel)
        if element:
            for nav in element.select('nav, .navigation, .sidebar, .toc, .footer'):
                nav.decompose()
            return element.get_text(' ', strip=True)
    return soup.get_text(' ', strip=True)

def classify_document(url):
    path = urlparse(url).path.strip('/').split('/')
    class_mapping = {
        'api': 'API Документация',
        'guide': 'Руководства',
        'tutorial': 'Обучающие материалы',
        'reference': 'Справочник',
        'faq': 'Частые вопросы',
        'getting-started': 'Начало работы',
        'install': 'Установка'
    }
    for segment in path:
        for k, v in class_mapping.items():
            if k in segment.lower():
                return v
    return f"Раздел: {path[0].capitalize()}" if path else "Главная"

def is_already_saved(normalized):
    if not os.path.exists(OUTPUT_FILE):
        return False
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return any(normalize_url(row['final_url']) == normalized for row in reader)

def save_to_csv(data):
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'url', 'final_url', 'redirected_from', 'title', 'content', 'class', 'word_count', 'timestamp'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

async def handle_page(context, url, base_domain):
    normalized_start = normalize_url(url)
    if not normalized_start or normalized_start in visited_normalized or is_already_saved(normalized_start):
        return []

    page = None
    try:
        logger.info(f"🌐 Загрузка: {url}")
        
        page = await context.new_page()
        
        # Увеличенный таймаут
        page.set_default_navigation_timeout(60000)
        
        # Переходим на страницу
        await page.goto(url, wait_until="networkidle", timeout=60000)
        
        # ДОПОЛНИТЕЛЬНЫЕ ДЕЙСТВИЯ ДЛЯ ИЗВЛЕЧЕНИЯ ССЫЛОК
        await asyncio.sleep(2)  # Даём время на выполнение JS
        
        # Прокручиваем страницу, чтобы подгрузить контент
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        await asyncio.sleep(1)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        await page.evaluate("window.scrollTo(0, 0)")
        
        # Кликаем по возможным кнопкам меню
        for selector in ['button.nav-toggle', '.menu-button', '#menu-toggle', '.hamburger']:
            try:
                await page.wait_for_selector(selector, timeout=3000)
                await page.click(selector)
                await asyncio.sleep(1)
            except:
                pass
        
        # Ждём появления навигации
        for selector in ['nav', '.navigation', '.sidebar', '.toc']:
            try:
                await page.wait_for_selector(selector, timeout=5000)
            except:
                pass
        
        current_url = str(page.url)
        final_normalized = normalize_url(current_url)
        
        if final_normalized in visited_normalized or is_already_saved(final_normalized):
            return []
        
        title = await page.title()
        content_html = await page.content()
        content_text = await extract_content(content_html)
        doc_class = classify_document(current_url)
        word_count = len(content_text.split())
        
        if word_count > 10:  # Минимальный объём контента
            save_to_csv({
                'url': url,
                'final_url': current_url,
                'redirected_from': url if url != current_url else '',
                'title': title,
                'content': content_text[:50000],
                'class': doc_class,
                'word_count': word_count,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            visited_normalized.add(final_normalized)
            logger.info(f"✅ Сохранено: {current_url}")
        
        # === КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: улучшенное извлечение ссылок ===
        all_links = []
        
        # 1. Извлекаем из обычных ссылок
        try:
            links = await page.eval_on_selector_all("a[href]", """
                anchors => anchors
                    .map(a => a.href)
                    .filter(href => href && !href.startsWith('mailto:') && !href.startsWith('tel:'))
                    .slice(0, 1000)  // Ограничиваем, чтобы не перегружать
            """)
            all_links.extend(links)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка извлечения обычных ссылок: {e}")
        
        # 2. Ищем в data-атрибутах (часто используют для SPA)
        try:
            data_links = await page.eval_on_selector_all("*[data-url], *[data-href], *[href]", """
                elements => elements
                    .map(el => el.getAttribute('data-url') || el.getAttribute('data-href') || el.href)
                    .filter(href => href && href.startsWith('http'))
                    .slice(0, 500)
            """)
            all_links.extend(data_links)
        except:
            pass
        
        # 3. Ищем в JSON (если навигация в скриптах)
        try:
            json_links = await page.eval_on_selector_all("script[type='application/json']", """
                scripts => {
                    const links = [];
                    scripts.forEach(script => {
                        try {
                            const data = JSON.parse(script.textContent);
                            // Рекурсивный поиск URL в JSON
                            const extractUrls = (obj) => {
                                if (typeof obj === 'string' && (obj.startsWith('http://') || obj.startsWith('https://'))) {
                                    links.push(obj);
                                } else if (typeof obj === 'object' && obj !== null) {
                                    Object.values(obj).forEach(extractUrls);
                                }
                            };
                            extractUrls(data);
                        } catch {}
                    });
                    return links.slice(0, 500);
                }
            """)
            all_links.extend(json_links)
        except:
            pass
        
        # Уникализируем и фильтруем
        new_urls = []
        seen = set()
        for link in all_links:
            if is_valid_doc_url(link, base_domain):
                norm = normalize_url(link)
                if norm and norm not in visited_normalized and norm not in seen:
                    seen.add(norm)
                    new_urls.append(link)
        
        logger.info(f"🔗 Найдено {len(new_urls)} новых ссылок на {current_url}")
        
        if len(new_urls) == 0:
            # Дополнительная диагностика
            content = await page.content()
            if "bot" in content.lower() or "security" in content.lower() or "captcha" in content.lower():
                logger.critical("🚨 Подозревается блокировка бота. Проверьте страницу вручную.")
            elif len(content) < 1000:
                logger.warning("⚠️ Страница слишком короткая — возможно, защита.")
            else:
                logger.warning(f"⚠️ Не найдено ссылок. Проверьте структуру страницы: {current_url}")
        
        await page.close()
        return new_urls

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке {url}: {e}")
        if page:
            await page.close()
        return []

async def crawl_with_playwright():
    global visited_normalized
    domain = urlparse(BASE_URL).netloc
    queue = [BASE_URL]
    os.makedirs("output", exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm = normalize_url(row['final_url'])
                if norm:
                    visited_normalized.add(norm)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        while queue:
            current_batch = queue[:MAX_CONCURRENT]
            queue = queue[len(current_batch):]

            tasks = [handle_page(context, url, domain) for url in current_batch]
            results = await asyncio.gather(*tasks)

            for new_links in results:
                queue.extend(new_links)

            # Уникализация очереди
            unique_queue = []
            seen = set()
            for url in queue:
                norm = normalize_url(url)
                if norm and norm not in seen:
                    seen.add(norm)
                    unique_queue.append(url)
            queue = unique_queue

            logger.info(f"🔍 Очередь: {len(queue)} ссылок")

        await browser.close()
        logger.info(f"\n✅ Готово! Сохранено в {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(crawl_with_playwright())