# =========================== Imports ===========================
import requests
import time
import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =========================== Konfiguration ===========================

# Trading-Symbol
SYMBOL = "BTCUSDT"

# Unterstützte Zeitintervalle
INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

# Maximale Kerzen pro API-Abfrage
MAX_CANDLES = 1000

# Anzahl Abfragen pro Batch (für Zeitfenster-Generierung)
REQUESTS_PER_BATCH = 1100

# Maximale Anzahl paralleler Threads
THREADS = 150

# Wartezeit zwischen Batches in Sekunden
SLEEP_AFTER_BATCH = 61

# Zielordner für CSV-Ausgabe
OUTPUT_FOLDER = "data"
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mapping der Intervalle auf Minutenanzahl für Zeitfenster-Berechnung
INTERVAL_TO_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200  # 1 Monat ≈ 30 Tage
}


# =========================== Funktionen ===========================

def get_klines(symbol, interval, start_ms, end_ms, limit=MAX_CANDLES):
    """
    Holt historische Kerzen-Daten (Klines) von der Binance API.

    :param symbol: Trading-Paar (z.B. BTCUSDT)
    :param interval: Zeitintervall (z.B. 1m, 1h, 1d)
    :param start_ms: Startzeitpunkt in Millisekunden
    :param end_ms: Endzeitpunkt in Millisekunden
    :param limit: Maximale Anzahl Kerzen pro Abfrage
    :return: JSON-Daten der Kerzen oder leere Liste bei Fehler
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit
    }

    for attempt in range(5):  # Bis zu 5 Versuche bei Fehlern
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 418]:
                # Rate Limit erreicht
                wait = int(response.headers.get("Retry-After", 60))
                tqdm.write(f"Rate Limit erreicht. Warte {wait}s.")
                time.sleep(wait)
            else:
                tqdm.write(f"HTTP-Fehler {response.status_code}: {response.text}")
                time.sleep(2)
        except Exception as e:
            tqdm.write(f"Verbindungsfehler: {str(e)}")
            time.sleep(5)

    return []  # Bei allen Fehlversuchen leere Liste zurückgeben


def fetch_window(symbol, interval, start_ms, end_ms):
    """
    Wrapper-Funktion für die Abfrage eines Zeitfensters.

    :return: Kerzen-Daten für das Zeitfenster
    """
    return get_klines(symbol, interval, start_ms, end_ms)


def generate_time_windows(end_time, total_windows, candle_minutes, candles_per_request=MAX_CANDLES):
    """
    Erzeugt Zeitfenster für die API-Abfragen, beginnend vom aktuellen Endzeitpunkt rückwärts.

    :param end_time: Endzeitpunkt als datetime-Objekt
    :param total_windows: Anzahl gewünschter Zeitfenster
    :param candle_minutes: Dauer einer Kerze in Minuten
    :param candles_per_request: Anzahl Kerzen pro Zeitfenster
    :return: Liste von Tupeln (start_ms, end_ms)
    """
    ranges = []
    ms_per_request = candles_per_request * candle_minutes * 60 * 1000

    for i in range(total_windows):
        end_ms = int(end_time.timestamp() * 1000) - i * ms_per_request
        start_ms = end_ms - ms_per_request + 1

        # Binance-Daten sind erst ab 17.08.2017 verfügbar
        if start_ms < int(datetime(2017, 8, 17).timestamp() * 1000):
            break

        ranges.append((start_ms, end_ms))

    return ranges


def download_interval(symbol, interval):
    """
    Lädt vollständige historische Daten für ein bestimmtes Intervall herunter
    und speichert diese als CSV-Datei.

    :param symbol: Trading-Paar
    :param interval: Zeitintervall
    """
    all_data = []
    tqdm.write(f"\n=== Starte Download für {symbol} - {interval} ===")

    end_time = datetime.utcnow() - timedelta(days=2)  # Sicherheitspuffer für aktuelle Daten
    start_limit = datetime(2017, 8, 17)
    candle_minutes = INTERVAL_TO_MINUTES[interval]

    while True:
        batch_ranges = generate_time_windows(end_time, REQUESTS_PER_BATCH, candle_minutes)

        if not batch_ranges:
            tqdm.write("Alle Daten für dieses Intervall heruntergeladen.\n")
            break

        # Parallele Abfragen über mehrere Threads
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = {
                executor.submit(fetch_window, symbol, interval, start_ms, end_ms): (start_ms, end_ms)
                for start_ms, end_ms in batch_ranges
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Batch Fortschritt"):
                result = future.result()
                if result:
                    all_data.extend(result)

        # Nächsten Endzeitpunkt berechnen (rückwärts)
        earliest_ms_in_batch = min(start for (start, end) in batch_ranges)
        end_time = datetime.utcfromtimestamp((earliest_ms_in_batch - 1) / 1000)

        if end_time < start_limit:
            tqdm.write("Früheste verfügbare Binance-Daten erreicht.")
            break

        tqdm.write(f"Batch fertig. Nächster Endzeitpunkt: {end_time}")
        tqdm.write(f"Schlafe {SLEEP_AFTER_BATCH} Sekunden...")
        time.sleep(SLEEP_AFTER_BATCH)

    if not all_data:
        tqdm.write("Keine Daten heruntergeladen.")
        return

    # Umwandlung in DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
    ])

    # Formatierung & Duplikat-Entfernung
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.drop_duplicates(subset=['open_time'])
    df = df.sort_values(by='open_time')

    # CSV-Speicherung
    output_path = os.path.join(OUTPUT_FOLDER, f"{symbol}_{interval}.csv")
    df.to_csv(output_path, index=False)
    tqdm.write(f"Daten gespeichert unter {output_path}\n")


# =========================== Hauptprogramm ===========================

def main():
    """
    Hauptfunktion, die für alle definierten Intervalle die Daten abruft.
    """
    for interval in INTERVALS:
        download_interval(SYMBOL, interval)


# =========================== Einstiegspunkt ===========================

if __name__ == "__main__":
    main()
