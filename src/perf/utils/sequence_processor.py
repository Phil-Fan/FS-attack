
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class SequenceProcessor:

    def __init__(
        self,
        time_limit: int = 1200,
        test_ratio: float = 0.2,
        num_classes: int = 10,
        random_seed: int = 42,
        window_size_min: int = 30,
        window_size_max: int = 50,
        overlap_ratio: float = 0.5,
    ):
        self.time_limit = time_limit
        self.test_ratio = test_ratio
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.window_size_min = window_size_min
        self.window_size_max = window_size_max
        self.overlap_ratio = overlap_ratio
        self.kv_bins = self._create_kv_bins()
        # Use an independent RNG to avoid affecting the global random state
        self.rng = np.random.default_rng(random_seed)

    def _create_kv_bins(self) -> np.ndarray:
        return np.linspace(0, 100, self.num_classes + 1)

    def _sliding_windows(
        self,
        itl_values: np.ndarray,
        kv_values: np.ndarray
    ) -> list:
        windows = []
        pos = 0

        while pos < len(itl_values):
            window_size = int(self.rng.integers(self.window_size_min, self.window_size_max + 1))

            if pos + window_size > len(itl_values):
                break

            itl_window = itl_values[pos:pos + window_size]
            kv_window = kv_values[pos:pos + window_size]

            windows.append({
                'itl': itl_window,
                'kv_usage': kv_window.mean()
            })

            stride = int(window_size * self.overlap_ratio)
            if stride == 0:
                stride = 1
            pos += stride

        return windows

    def process(
        self,
        input_csv: Path,
        output_dir: Path,
        dataset_name: str = "dataset"
    ) -> Dict[str, pd.DataFrame]:
        logger.info("=" * 60)
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info("=" * 60)

        logger.info(f"Reading data: {input_csv}")
        df = pd.read_csv(input_csv)
        logger.info(f"Number of raw records: {len(df)}")

        if df['kv_usage'].isna().any():
            nan_count = df['kv_usage'].isna().sum()
            logger.warning(f"Found {nan_count} NaN values, filling with 0")
            df['kv_usage'] = df['kv_usage'].fillna(0)

        df['timestamp'] = pd.to_datetime(df['time'])
        df['time_seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        df = df[df['time_seconds'] <= self.time_limit].copy()
        logger.info(f"Number of records after filtering: {len(df)} (<= {self.time_limit}s)")

        logger.info("Grouping by (user_id, num_run)...")
        req_groups = df.groupby(['user_id', 'num_run'])
        logger.info(f"Generated {len(req_groups)} request sequences")

        sequences = []
        total_windows = 0
        skipped_requests = 0

        for (user_id, num_run), group in req_groups:
            group = group.sort_values('time_seconds')

            itl_values = group['itl_ms'].values
            kv_values = group['kv_usage'].values

            windows = self._sliding_windows(itl_values, kv_values)

            if len(windows) == 0:
                skipped_requests += 1
                continue

            for window_id, window in enumerate(windows):
                sequences.append({
                    'user_id': user_id,
                    'num_req': num_run,
                    'window_id': window_id,
                    'itl_sequence': window['itl'],
                    'kv_usage': window['kv_usage'],
                    'seq_length': len(window['itl']),
                })
                total_windows += 1

        logger.info(f"Generated {total_windows} windows (skipped {skipped_requests} requests with too few windows)")

        seq_df = pd.DataFrame(sequences)
        logger.info(f"Sequence statistics: min length={seq_df['seq_length'].min()}, "
                   f"max length={seq_df['seq_length'].max()}, "
                   f"mean length={seq_df['seq_length'].mean():.2f}")

        seq_df['kv_label'] = pd.cut(
            seq_df['kv_usage'],
            bins=self.kv_bins,
            labels=range(self.num_classes),
            include_lowest=True
        ).astype(int)

        logger.info(f"KV Label distribution: {np.bincount(seq_df['kv_label'])}")

        np.random.seed(self.random_seed)
        unique_users = seq_df['user_id'].unique()
        np.random.shuffle(unique_users)

        n_test = int(len(unique_users) * self.test_ratio)
        test_users = unique_users[:n_test]
        train_users = unique_users[n_test:]

        train_df = seq_df[seq_df['user_id'].isin(train_users)]
        eval_df = seq_df[seq_df['user_id'].isin(test_users)]

        logger.info(f"Training set: {len(train_df)} sequences, {len(train_users)} users")
        logger.info(f"Evaluation set: {len(eval_df)} sequences, {len(test_users)} users")

        output_dir = Path(output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        results['train'] = self._save_dataset_csv(train_df, output_dir / 'train.csv')
        results['eval'] = self._save_dataset_csv(eval_df, output_dir / 'eval.csv')

        logger.info(f"Data saved to: {output_dir}")
        logger.info("=" * 60)

        return results

    def _save_dataset_csv(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        data_df = pd.DataFrame({
            'userid': df['user_id'].values,
            'numreq': df['num_req'].values,
            'window_id': df['window_id'].values,
            'itl': [','.join(map(str, seq)) for seq in df['itl_sequence'].values],
            'kv_label': df['kv_label'].values,
            'kv_usage': df['kv_usage'].values
        })
        path_obj = Path(path)
        data_df.to_csv(path, index=False)
        logger.info(f"  Saved: {path_obj.name} ({len(data_df)} rows)")
        return data_df


class SequenceDataset:

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        max_seq_length: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_length = max_seq_length

        self.data_df = pd.read_csv(self.data_dir / f'{split}.csv')

        self.sequences = [
            np.array([float(x) for x in seq_str.split(',')])
            for seq_str in self.data_df['itl'].values
        ]

        if max_seq_length is not None:
            self.sequences = self._pad_sequences(self.sequences, max_seq_length)

        self.labels = self.data_df['kv_label'].values
        self.kv_usages = self.data_df['kv_usage'].values

        logger.info(f"Loaded {split} dataset: {len(self.sequences)} sequences")

    def _pad_sequences(self, sequences: list, max_length: int) -> list:
        padded = []
        for seq in sequences:
            if len(seq) > max_length:
                padded.append(seq[:max_length])
            else:
                padded_seq = np.zeros(max_length)
                padded_seq[:len(seq)] = seq
                padded.append(padded_seq)
        return padded

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'userid': self.data_df.iloc[idx]['userid'],
            'numreq': self.data_df.iloc[idx]['numreq'],
            'window_id': self.data_df.iloc[idx]['window_id'],
            'sequence': self.sequences[idx],
            'kv_label': self.labels[idx],
            'kv_usage': self.kv_usages[idx]
        }


if __name__ == "__main__":
    setup_logging()
    processor = SequenceProcessor(
        time_limit=1200,
        test_ratio=0.2,
        num_classes=10
    )

    input_csv = Path("data/data_special/data_ratio_0.2/itl_aligned.csv")
    output_dir = Path("data/test_dataset")

    results = processor.process(input_csv, output_dir, "test_dataset")

    logger.info("Generated files:")
    for key, df in results.items():
        logger.info("  %s: %s", key, df.shape)
