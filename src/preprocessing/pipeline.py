from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.engineer import FeatureEngineer
from src.preprocessing.cleaning import SurfaceCleaner
from src.preprocessing.normalization import SurfaceNormalizer
from src.preprocessing.regime import RegimeTagger


@dataclass
class PreprocessingArtifacts:
    cleaned_chain: pd.DataFrame
    normalized_surfaces: np.ndarray
    regimes: np.ndarray


class PreprocessingPipeline:
    def __init__(self):
        self.cleaner = SurfaceCleaner()
        self.normalizer = SurfaceNormalizer()
        self.regime_tagger = RegimeTagger()
        self.features = FeatureEngineer()

    def run(self, chain_df: pd.DataFrame, surfaces: np.ndarray) -> PreprocessingArtifacts:
        cleaned = self.cleaner.clean_chain(chain_df)
        normalized = self.normalizer.fit_transform(surfaces)
        regimes = self.regime_tagger.fit_predict_from_surfaces(surfaces)
        return PreprocessingArtifacts(cleaned_chain=cleaned, normalized_surfaces=normalized, regimes=regimes)
