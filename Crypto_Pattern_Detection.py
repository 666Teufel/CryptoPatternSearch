import os
import glob
import pandas as pd
import torch
import mplfinance as mpf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Konfiguration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PATTERNS = {
    # 1-Kerzen-Muster
    'hammer': {
        'function': lambda w: (
            (w[0, 3] > w[0, 0]) and  # Grüne Kerze
            (w[0, 0] > w[0, 3] - (w[0, 1] - w[0, 3]) * 2) and  # Oberer Schatten < 1/3 des Körpers
            (w[0, 2] < w[0, 0] - (w[0, 0] - w[0, 2]) * 2) and  # Unterer Schatten > 2x Körperlänge
            (w[0, 1] - w[0, 3]) < (w[0, 0] - w[0, 2])  # Unterer Schatten dominierend
        ),
        'window': 1,
        'description': 'Hammer (bullish reversal)',
        'type': 'bullish'
    },
    'inverse_hammer': {
        'function': lambda w: (
            (w[0, 3] > w[0, 0]) and  # Grüne Kerze
            (w[0, 1] - w[0, 3]) > (w[0, 0] - w[0, 2]) * 2 and  # Oberer Schatten > 2x Körperlänge
            (w[0, 0] - w[0, 2]) < (w[0, 1] - w[0, 3]) / 2  # Unterer Schatten klein
        ),
        'window': 1,
        'description': 'Inverse Hammer (bullish reversal)',
        'type': 'bullish'
    },
    'hanging_man': {
        'function': lambda w: (
            (w[0, 0] > w[0, 3]) and  # Rote Kerze
            (w[0, 2] < w[0, 3] - (w[0, 0] - w[0, 3]) * 2) and  # Langer unterer Schatten
            (w[0, 1] - w[0, 0]) < (w[0, 3] - w[0, 2]) / 3  # Kleiner oberer Schatten
        ),
        'window': 1,
        'description': 'Hanging Man (bearish reversal)',
        'type': 'bearish'
    },
    'shooting_star': {
        'function': lambda w: (
            (w[0, 0] > w[0, 3]) and  # Rote Kerze
            (w[0, 1] - w[0, 0]) > (w[0, 3] - w[0, 2]) * 2 and  # Langer oberer Schatten
            (w[0, 3] - w[0, 2]) < (w[0, 1] - w[0, 0]) / 3  # Kleiner unterer Schatten
        ),
        'window': 1,
        'description': 'Shooting Star (bearish reversal)',
        'type': 'bearish'
    },
    'doji': {
        'function': lambda w: (
            abs(w[0, 0] - w[0, 3]) < 0.01 * (w[0, 1] - w[0, 2])  # Kein Körper
        ),
        'window': 1,
        'description': 'Doji (indecision)',
        'type': 'neutral'
    },
    'marubozu': {
        'function': lambda w: (
            (abs(w[0, 0] - w[0, 3]) > 0.95 * (w[0, 1] - w[0, 2])) and  # Kaum Schatten
            (min(w[0, 0], w[0, 3]) == w[0, 2]) and  # Tief = Open/Close
            (max(w[0, 0], w[0, 3]) == w[0, 1])  # Hoch = Open/Close
        ),
        'window': 1,
        'description': 'Marubozu (starker Trend)',
        'type': 'continuation'
    },
    
    # 2-Kerzen-Muster
    'bullish_engulfing': {
        'function': lambda w: (
            (w[1, 0] > w[1, 3]) and  # Tag 1 rot
            (w[0, 3] > w[0, 0]) and  # Tag 2 grün
            (w[0, 0] < w[1, 3]) and  # Open Tag 2 < Close Tag 1
            (w[0, 3] > w[1, 0])  # Close Tag 2 > Open Tag 1
        ),
        'window': 2,
        'description': 'Bullish Engulfing (starker Aufwärtstrend)',
        'type': 'bullish'
    },
    'bearish_engulfing': {
        'function': lambda w: (
            (w[1, 3] > w[1, 0]) and  # Tag 1 grün
            (w[0, 0] > w[0, 3]) and  # Tag 2 rot
            (w[0, 0] > w[1, 3]) and  # Open Tag 2 > Close Tag 1
            (w[0, 3] < w[1, 0])  # Close Tag 2 < Open Tag 1
        ),
        'window': 2,
        'description': 'Bearish Engulfing (starker Abwärtstrend)',
        'type': 'bearish'
    },
    'piercing_line': {
        'function': lambda w: (
            (w[1, 0] > w[1, 3]) and  # Tag 1 rot
            (w[0, 3] > w[0, 0]) and  # Tag 2 grün
            (w[0, 0] < w[1, 2]) and  # Open unter Tief von Tag 1
            (w[0, 3] > (w[1, 0] + w[1, 3]) / 2)  # Close über 50% des Körpers von Tag 1
        ),
        'window': 2,
        'description': 'Piercing Line (bullish reversal)',
        'type': 'bullish'
    },
    'dark_cloud_cover': {
        'function': lambda w: (
            (w[1, 3] > w[1, 0]) and  # Tag 1 grün
            (w[0, 0] > w[0, 3]) and  # Tag 2 rot
            (w[0, 0] > w[1, 1]) and  # Open über Hoch von Tag 1
            (w[0, 3] < (w[1, 0] + w[1, 3]) / 2)  # Close unter 50% des Körpers von Tag 1
        ),
        'window': 2,
        'description': 'Dark Cloud Cover (bearish reversal)',
        'type': 'bearish'
    },
    'bullish_harami': {
        'function': lambda w: (
            (w[1, 0] > w[1, 3]) and  # Tag 1 rot (groß)
            (w[0, 3] > w[0, 0]) and  # Tag 2 grün (klein)
            (w[0, 0] > w[1, 3]) and  # Open innerhalb Körper Tag 1
            (w[0, 3] < w[1, 0])  # Close innerhalb Körper Tag 1
        ),
        'window': 2,
        'description': 'Bullish Harami (bullish reversal)',
        'type': 'bullish'
    },
    'bearish_harami': {
        'function': lambda w: (
            (w[1, 3] > w[1, 0]) and  # Tag 1 grün (groß)
            (w[0, 0] > w[0, 3]) and  # Tag 2 rot (klein)
            (w[0, 0] < w[1, 3]) and  # Open innerhalb Körper Tag 1
            (w[0, 3] > w[1, 0])  # Close innerhalb Körper Tag 1
        ),
        'window': 2,
        'description': 'Bearish Harami (bearish reversal)',
        'type': 'bearish'
    },
    'tweezer_bottom': {
        'function': lambda w: (
            (w[1, 0] > w[1, 3]) and  # Tag 1 rot
            (w[0, 3] > w[0, 0]) and  # Tag 2 grün
            abs(w[1, 2] - w[0, 2]) < 0.01 * w[1, 2] and  # Gleiches Tief
            (w[0, 1] > w[1, 1])  # Tag 2 hat höheres Hoch
        ),
        'window': 2,
        'description': 'Tweezer Bottom (bullish reversal)',
        'type': 'bullish'
    },
    'tweezer_top': {
        'function': lambda w: (
            (w[1, 3] > w[1, 0]) and  # Tag 1 grün
            (w[0, 0] > w[0, 3]) and  # Tag 2 rot
            abs(w[1, 1] - w[0, 1]) < 0.01 * w[1, 1] and  # Gleiches Hoch
            (w[0, 2] < w[1, 2])  # Tag 2 hat tieferes Tief
        ),
        'window': 2,
        'description': 'Tweezer Top (bearish reversal)',
        'type': 'bearish'
    },
    
    # 3-Kerzen-Muster
    'morning_star': {
        'function': lambda w: (
            (w[2, 0] > w[2, 3]) and  # Tag 1 rot
            (w[1, 3] < w[1, 0]) and  # Tag 2 Doji/kleiner Körper
            (w[0, 3] > w[0, 0]) and  # Tag 3 grün
            (w[1, 2] < w[2, 2]) and  # Tief Tag 2 < Tief Tag 1
            (w[0, 3] > (w[2, 0] + w[2, 3]) / 2)  # Close Tag 3 > Mitte Körper Tag 1
        ),
        'window': 3,
        'description': 'Morning Star (starker bullish reversal)',
        'type': 'bullish'
    },
    'evening_star': {
        'function': lambda w: (
            (w[2, 3] > w[2, 0]) and  # Tag 1 grün
            (w[1, 3] < w[1, 0]) and  # Tag 2 Doji/kleiner Körper
            (w[0, 0] > w[0, 3]) and  # Tag 3 rot
            (w[1, 1] > w[2, 1]) and  # Hoch Tag 2 > Hoch Tag 1
            (w[0, 3] < (w[2, 0] + w[2, 3]) / 2)  # Close Tag 3 < Mitte Körper Tag 1
        ),
        'window': 3,
        'description': 'Evening Star (starker bearish reversal)',
        'type': 'bearish'
    },
    'three_white_soldiers': {
        'function': lambda w: (
            all(w[i, 3] > w[i, 0] for i in range(3)) and  # Drei grüne Kerzen
            all(w[i, 0] > w[i-1, 0] for i in range(1,3)) and  # Höhere Opens
            all(w[i, 3] > w[i-1, 3] for i in range(1,3)) and  # Höhere Closes
            all((w[i, 3] - w[i, 0]) > 0.7 * (w[i, 1] - w[i, 2]) for i in range(3))  # Lange Körper
        ),
        'window': 3,
        'description': 'Three White Soldiers (starkes bullisches Momentum)',
        'type': 'bullish'
    },
    'three_black_crows': {
        'function': lambda w: (
            all(w[i, 0] > w[i, 3] for i in range(3)) and  # Drei rote Kerzen
            all(w[i, 0] < w[i-1, 0] for i in range(1,3)) and  # Tiefere Opens
            all(w[i, 3] < w[i-1, 3] for i in range(1,3)) and  # Tiefere Closes
            all((w[i, 0] - w[i, 3]) > 0.7 * (w[i, 1] - w[i, 2]) for i in range(3))  # Lange Körper
        ),
        'window': 3,
        'description': 'Three Black Crows (starkes bärisches Momentum)',
        'type': 'bearish'
    },
    'bullish_abandoned_baby': {
        'function': lambda w: (
            (w[2, 0] > w[2, 3]) and  # Tag 1 rot
            (w[1, 1] < w[2, 2]) and  # Lücke nach unten
            (w[1, 3] < w[1, 0]) and  # Tag 2 Doji
            (w[0, 3] > w[0, 0]) and  # Tag 3 grün
            (w[0, 0] > w[1, 1]) and  # Lücke nach oben
            (w[0, 3] > w[2, 0])  # Close über Tag 1 Open
        ),
        'window': 3,
        'description': 'Bullish Abandoned Baby (seltenes starkes Reversal)',
        'type': 'bullish'
    },
    'bearish_abandoned_baby': {
        'function': lambda w: (
            (w[2, 3] > w[2, 0]) and  # Tag 1 grün
            (w[1, 1] > w[2, 1]) and  # Lücke nach oben
            (w[1, 3] < w[1, 0]) and  # Tag 2 Doji
            (w[0, 0] > w[0, 3]) and  # Tag 3 rot
            (w[0, 1] < w[1, 2]) and  # Lücke nach unten
            (w[0, 3] < w[2, 3])  # Close unter Tag 1 Close
        ),
        'window': 3,
        'description': 'Bearish Abandoned Baby (seltenes starkes Reversal)',
        'type': 'bearish'
    }
}

def setup_directories():
    """Erstelle notwendige Verzeichnisse"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Data directory '{DATA_DIR}' created. Please add your CSV files there and run again.")
        return False
    return True

def load_data(file_path):
    """Lade und bereite Daten vor"""
    try:
        # Versuche verschiedene Trennzeichen
        df = pd.read_csv(file_path, delimiter=';|,', engine='python')
        
        # Finde die richtigen Spalten (case insensitive)
        columns_map = {
            'open_time': ['open_time', 'timestamp', 'date', 'time'],
            'open': ['open', 'opening', 'open_price'],
            'high': ['high', 'high_price', 'max'],
            'low': ['low', 'low_price', 'min'],
            'close': ['close', 'closing', 'close_price']
        }
        
        # Mapping der tatsächlichen Spaltennamen
        actual_columns = {}
        for standard_col, possible_cols in columns_map.items():
            for col in possible_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    actual_columns[standard_col] = col
                    break
            else:
                raise ValueError(f"Required column '{standard_col}' not found in {file_path}")
        
        # Wähle und benenne Spalten um
        df = df.rename(columns={v: k for k, v in actual_columns.items()})
        df = df[list(columns_map.keys())]
        
        # Konvertiere Zeitstempel
        df['open_time'] = pd.to_datetime(df['open_time'])
        df.set_index('open_time', inplace=True)
        
        # Umbenennen für mplfinance
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        })
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {str(e)}")

def detect_patterns_gpu(data, patterns):
    """GPU-beschleunigte Mustererkennung"""
    try:
        # Extrahiere Werte in der richtigen Reihenfolge
        values = data[['Open', 'High', 'Low', 'Close']].values
        tensor_data = torch.tensor(values, dtype=torch.float32).cuda()
        results = {}
        
        for pattern_name, config in patterns.items():
            window_size = config['window']
            pattern_func = config['function']
            matches = []
            
            for i in range(len(data) - window_size + 1):
                window = tensor_data[i:i+window_size]
                try:
                    if pattern_func(window):
                        matches.append(i + window_size - 1)
                except Exception as e:
                    print(f"Error evaluating {pattern_name} at index {i}: {str(e)}")
                    continue
            
            results[pattern_name] = matches
        
        torch.cuda.empty_cache()
        return results
    
    except Exception as e:
        torch.cuda.empty_cache()
        raise RuntimeError(f"GPU processing error: {str(e)}")

def save_pattern_results(data, results, symbol, interval):
    """Speichere Ergebnisse für jedes Muster"""
    pattern_stats = []
    
    for pattern_name, indices in results.items():
        pattern_dir = os.path.join(RESULTS_DIR, f'{symbol}_{interval}', pattern_name)
        os.makedirs(pattern_dir, exist_ok=True)
        
        # Speichere jedes Vorkommen als CSV und Bild
        for idx in indices:
            try:
                start_idx = max(0, idx-10)
                end_idx = min(len(data), idx+10)
                sample = data.iloc[start_idx:end_idx]
                
                # Speichere CSV
                sample.to_csv(os.path.join(pattern_dir, f'{idx}.csv'))
                
                # Candlestick Chart
                mpf.plot(
                    sample,
                    type='candle',
                    style='charles',
                    title=f'{pattern_name} at {data.index[idx]}',
                    savefig=os.path.join(pattern_dir, f'{idx}.png'),
                    close=True
                )
            except Exception as e:
                print(f"Error saving {pattern_name} at index {idx}: {str(e)}")
                continue
        
        # Sammle Statistiken
        pattern_stats.append({
            'pattern': pattern_name,
            'count': len(indices),
            'frequency': len(indices)/len(data) if len(data) > 0 else 0,
            'description': PATTERNS[pattern_name]['description'],
            'type': PATTERNS[pattern_name]['type']
        })
    
    return pd.DataFrame(pattern_stats)

def analyze_pattern_accuracy(data, results, lookahead=5):
    """Analysiere die Genauigkeit der erkannten Muster"""
    accuracy_stats = []
    close_prices = data['Close'].values

    for pattern_name, indices in results.items():
        correct = 0
        total = 0
        pattern_type = PATTERNS[pattern_name]['type']
        
        for idx in indices:
            if idx + lookahead < len(close_prices):
                current_price = close_prices[idx]
                future_price = close_prices[idx + lookahead]
                
                if pattern_type == 'bullish':
                    if future_price > current_price:
                        correct += 1
                    total += 1
                elif pattern_type == 'bearish':
                    if future_price < current_price:
                        correct += 1
                    total += 1
                elif pattern_type == 'neutral':
                    # Bei neutralen Mustern Richtung des vorherigen Trends prüfen
                    if idx > 0:
                        prev_price = close_prices[idx-1]
                        if (future_price > current_price and current_price > prev_price) or \
                           (future_price < current_price and current_price < prev_price):
                            correct += 1
                        total += 1
                elif pattern_type == 'continuation':
                    # Prüfe ob Trend fortgesetzt wurde
                    if idx > 0:
                        prev_price = close_prices[idx-1]
                        if (future_price > current_price and current_price > prev_price) or \
                           (future_price < current_price and current_price < prev_price):
                            correct += 1
                        total += 1
        
        accuracy = correct / total if total > 0 else None
        success_rate = correct / total if total > 0 else 0
        accuracy_stats.append({
            'pattern': pattern_name,
            'accuracy': accuracy,
            'success_rate': success_rate,
            'tested': total,
            'correct': correct
        })
    
    return pd.DataFrame(accuracy_stats)

def visualize_stats(stats_df, accuracy_df, symbol, interval):
    """Visualisiert die Musterstatistiken und Genauigkeit"""
    if stats_df.empty or accuracy_df.empty:
        print(f"No data to visualize for {symbol} {interval}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle(f'{symbol} {interval} - Pattern Analysis', fontsize=16)
    
    # Häufigkeitsdiagramm
    stats_df.set_index('pattern')['count'].plot.bar(ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Pattern Occurrences')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Genauigkeitsdiagramm
    accuracy_df.set_index('pattern')['success_rate'].plot.bar(ax=axes[0, 1], color='orange')
    axes[0, 1].set_title('Pattern Success Rate (5 periods)')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Erfolgsrate nach Mustertyp
    if 'type' in stats_df.columns:
        type_stats = stats_df.merge(accuracy_df, on='pattern')
        for t in ['bullish', 'bearish', 'neutral', 'continuation']:
            type_data = type_stats[type_stats['type'] == t]
            if not type_data.empty:
                type_data.set_index('pattern')['success_rate'].plot.bar(
                    ax=axes[1, 0], 
                    color={'bullish': 'green', 'bearish': 'red'}.get(t, 'gray'),
                    alpha=0.7,
                    label=t
                )
        axes[1, 0].set_title('Success Rate by Pattern Type')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Korrelation zwischen Häufigkeit und Erfolgsrate
    if 'success_rate' in accuracy_df.columns and 'count' in stats_df.columns:
        merged = stats_df.merge(accuracy_df, on='pattern')
        axes[1, 1].scatter(merged['count'], merged['success_rate'], s=100, alpha=0.7)
        for i, row in merged.iterrows():
            axes[1, 1].annotate(row['pattern'], (row['count'], row['success_rate']))
        axes[1, 1].set_title('Frequency vs. Success Rate')
        axes[1, 1].set_xlabel('Frequency (Count)')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.join(RESULTS_DIR, f'{symbol}_{interval}')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'pattern_stats.png'))
    plt.close()

def process_all_files():
    """Verarbeite alle CSV-Dateien im Datenordner"""
    if not setup_directories():
        return
    
    files = glob.glob(os.path.join(DATA_DIR, 'BTCUSDT_*.csv'))
    
    if not files:
        available_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
        if available_files:
            print(f"Found these CSV files but they don't match BTCUSDT pattern: {available_files}")
        print(f"No BTCUSDT_*.csv files found in {DATA_DIR}")
        return
    
    all_stats = []
    summary_stats = []
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            file_name = os.path.basename(file_path)
            symbol, interval = file_name.split('_')[:2]
            interval = interval.split('.')[0]
            
            print(f"\nProcessing {file_name}...")
            data = load_data(file_path)
            
            # GPU-Mustererkennung
            print("Detecting patterns with GPU...")
            results = detect_patterns_gpu(data, PATTERNS)
            
            # Ergebnisse speichern
            print("Saving pattern results...")
            stats_df = save_pattern_results(data, results, symbol, interval)
            
            # Genauigkeit analysieren
            print("Analyzing pattern accuracy...")
            accuracy_df = analyze_pattern_accuracy(data, results)
            
            # Statistik kombinieren
            if not stats_df.empty and not accuracy_df.empty:
                combined_stats = stats_df.merge(accuracy_df, on='pattern')
                combined_stats['symbol'] = symbol
                combined_stats['interval'] = interval
                all_stats.append(combined_stats)
                
                # Zusammenfassung für Gesamtstatistik
                summary = combined_stats.copy()
                summary['file'] = file_name
                summary_stats.append(summary)
            
            # Visualisierung
            print("Generating visualizations...")
            visualize_stats(stats_df, accuracy_df, symbol, interval)
            
            # Musterzusammenfassung ausgeben
            print("\nPattern Summary:")
            print(combined_stats[['pattern', 'count', 'success_rate', 'tested']])
            
        except Exception as e:
            print(f"\nError processing {file_name}: {str(e)}\n")
            continue
    
    if all_stats:
        # Gesamtstatistik speichern
        full_stats = pd.concat(all_stats)
        full_stats.to_csv(os.path.join(RESULTS_DIR, 'full_statistics.csv'), index=False)
        
        # Zusammenfassungsstatistik
        if summary_stats:
            summary_df = pd.concat(summary_stats)
            summary_df.to_csv(os.path.join(RESULTS_DIR, 'summary_statistics.csv'), index=False)
            
            # Visualisiere Gesamthäufigkeit
            plt.figure(figsize=(14, 8))
            summary_df.groupby('pattern')['count'].sum().sort_values().plot.bar()
            plt.title('Total Pattern Occurrences Across All Intervals')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'total_patterns.png'))
            plt.close()
            
            # Visualisiere Erfolgsraten
            plt.figure(figsize=(14, 8))
            success_rates = summary_df.groupby('pattern')['success_rate'].mean().sort_values()
            success_rates.plot.bar(color='orange')
            plt.title('Average Success Rate by Pattern')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'success_rates.png'))
            plt.close()

if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        else:
            raise RuntimeError("CUDA GPU not available - this script requires GPU acceleration")
        
        process_all_files()
        print(f"\nAnalysis complete. Results saved in '{RESULTS_DIR}' directory.")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()