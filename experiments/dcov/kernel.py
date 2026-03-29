import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import struct
import os

DATA_DIR = os.path.join("data", "stock_market", "matlab_data")

def load_tensor_from_binary() -> np.ndarray:
    """
    Loads a 3D tensor from a binary file exported from MATLAB.
    """
    binary_path = os.path.join(DATA_DIR, 'log_return_tensor.dat')
    try:
        with open(binary_path, 'rb') as f:
            # Read 3 int32 values for dimensions
            dims_data = f.read(3 * 4)  
            dims = struct.unpack('III', dims_data)

            # Each double is 8 bytes
            data_size = dims[0] * dims[1] * dims[2]
            data_bytes = f.read(data_size * 8)  
            data = struct.unpack('d' * data_size, data_bytes)

            # Use Fortran order ('F') because MATLAB is column-major
            tensor = np.array(data).reshape(dims, order='F')  
            return tensor
    except Exception as e:
        print(f"Failed to load from binary, attempting text file... Error: {e}")
        return load_tensor_from_text()

def load_sp500_data() -> Tuple[np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Loads data from both CSV and binary files.
    """
    tensor_data = load_tensor_from_binary()

    sector_info = pd.read_csv(os.path.join(DATA_DIR, 'python_sector_info.csv'))
    stock_info = pd.read_csv(os.path.join(DATA_DIR, 'python_stock_info.csv'))

    sector_names = sector_info['Sector'].tolist()
    return tensor_data, sector_names, sector_info, stock_info

def get_true_data_matlab() -> Tuple[List[str], List[np.ndarray]]:
    """
    Processes the raw tensor data into a list of valid data arrays per sector.
    """
    tensor_data, sector_names, _, _ = load_sp500_data()

    # Combine sector_names with indices and sort by name
    sector_indices_sorted = sorted(enumerate(sector_names), key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sector_indices_sorted]
    sorted_sector_names = [name for _, name in sector_indices_sorted]

    processed_datas = []

    for i in sorted_indices:
        # Transpose to get (stocks, time) format
        sector_data = tensor_data[i, :, :].T
        
        # Identify columns (stocks) that are entirely NaN
        all_nan_cols = np.all(np.isnan(sector_data), axis=0)

        if np.any(all_nan_cols):
            first_all_nan_col = np.argmax(all_nan_cols)
        else:
            first_all_nan_col = sector_data.shape[1]

        # Slice to keep only valid stock columns
        valid_data = sector_data[:, :first_all_nan_col]
        processed_datas.append(valid_data)

    return sorted_sector_names, processed_datas

def get_stock_info(sector_idx: int, stock_idx: int) -> Dict:
    """
    Retrieves metadata for a stock based on its tensor position.
    """
    _, _, sector_info, stock_info = load_sp500_data()
    sector_name = sector_info.loc[sector_idx, 'Sector']

    # Adjust for 1-based indexing used in the CSV mapping
    stock_query = stock_info[
        (stock_info['SectorIndex'] == sector_idx + 1) &
        (stock_info['StockIndex'] == stock_idx + 1)
    ]

    stock_symbol = stock_query.iloc[0]['StockSymbol'] if not stock_query.empty else 'Invalid'

    return {
        'sector_idx': sector_idx,
        'stock_idx': stock_idx,
        'sector_name': sector_name,
        'stock_symbol': stock_symbol,
    }

def find_stock_position(stock_symbol: str) -> Tuple[int, int]:
    """
    Finds the (sector_idx, stock_idx) for a given stock symbol.
    """
    _, _, _, stock_info = load_sp500_data()

    stock_query = stock_info[stock_info['StockSymbol'] == stock_symbol]
    if stock_query.empty:
        return -1, -1

    # Convert 1-based index back to 0-based
    sector_idx = stock_query.iloc[0]['SectorIndex'] - 1
    stock_idx = stock_query.iloc[0]['StockIndex'] - 1
    return sector_idx, stock_idx

def get_sector_stocks(sector_name: str) -> List[str]:
    """
    Returns a list of all stock symbols belonging to a specific sector.
    """
    _, _, _, stock_info = load_sp500_data()
    return stock_info[stock_info['Sector'] == sector_name]['StockSymbol'].tolist()

if __name__ == "__main__":
    print("=== Loading S&P 500 Data ===")
    unique_sectors, processed_datas = get_true_data_matlab()

    print(f"Number of sectors: {len(unique_sectors)}")
    print(f"Sectors: {unique_sectors}")

    for i, (sector, data) in enumerate(zip(unique_sectors, processed_datas)):
        print(f"{i}: {sector} - Data shape: {data.shape}")

    print("\n=== Lookup Example ===")
    info = get_stock_info(0, 1)
    print(f"Tensor position (0,1) corresponds to:")
    print(f"  Sector: {info['sector_name']}")
    print(f"  Stock: {info['stock_symbol']}")

    if processed_datas and processed_datas[0].shape[1] > 0:
        first_sector_stocks = get_sector_stocks(unique_sectors[0])
        if first_sector_stocks:
            example_stock = first_sector_stocks[0]
            sector_idx, stock_idx = find_stock_position(example_stock)
            print(f"\nStock {example_stock} tensor position: ({sector_idx}, {stock_idx})")

    print(f"\nAll stocks in the first sector {unique_sectors[0]}:")
    first_sector_stocks = get_sector_stocks(unique_sectors[0])
    for i, stock in enumerate(first_sector_stocks):
        print(f"  {i}: {stock}")