#from molevaluate import Screener
import subprocess

__all__ = ["molevaluatescorer"]
from dataclasses import dataclass
from typing import List
from rdkit import Chem
import glob
import os
import yaml
import pandas as pd
import numpy as np

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults

@add_tag("__parameters")
@dataclass
class Parameters:
    scoring_yaml: str = None
    metric: str = None

@add_tag("__component")
class molevaluatescorer:

    def __init__(self, params: Parameters):

        self.moleval_input = {}
        if params.metric is None:
            raise ValueError("No metric provided, please specify a metric in the parameters.")
        else:
            self.metric = params.metric[0]
        
        # Load scoring options from a YAML file
        if params.scoring_yaml is None:
            raise ValueError("No scoring YAML file provided, please specify a scoring YAML file in the parameters.")
        else:
            self.scoring_yaml = params.scoring_yaml[0]
            with open(self.scoring_yaml, 'r') as file:
                self.moleval_input = yaml.safe_load(file)


    def __call__(self, mols: List[str]) -> ComponentResults:
        # Input validation
        assert mols is not None and len(mols) > 0, "Input SMILES list cannot be empty"
        assert all(isinstance(mol, str) for mol in mols), "All molecules must be SMILES strings"
        
        # Ensure the necessary keys exist in the scoring_options dictionary
        self.moleval_input['molecules'] = {}
        self.moleval_input['molecules']['reinvent-molecules'] = {}
        
        # Set the smiles list
        self.moleval_input['molecules']["reinvent-molecules"]['smiles'] = mols

        with open('molevaluate_input.yaml', 'w') as file:
            yaml.dump(self.moleval_input, file)

        cmd = ['conda', 'run', '-n', 'molevaluate',
                    'moleval', '-y', 'molevaluate_input.yaml']
        timeout = 999999
        try:
            output = subprocess.run(cmd, timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired as err:
            LOGGER.error(f"Molevaluate crashed or overran using command: {' '.join(cmd)}")
            raise err

        # Check subprocess execution
        assert output.returncode == 0, f"Molevaluate failed with return code {output.returncode}. stderr: {output.stderr.decode()}"
        
        print(output.stdout.decode())
        print(output.stderr.decode())

        # Find and validate output file
        output_files = glob.glob('./outputs/*/*/*.csv')
        assert len(output_files) > 0, "No CSV output files found from molevaluate"
        output_df = output_files[0]
        assert os.path.isfile(output_df), f"Output file does not exist: {output_df}"
        
        # Load and validate DataFrame
        df = pd.read_csv(output_df)
        assert not df.empty, "Output CSV file is empty"
        assert self.metric in df.columns.to_list(), f"Metric '{self.metric}' not found in output columns: {df.columns.tolist()}"
        
        # Extract and validate scores
        scores = df[self.metric].to_numpy().flatten()
        assert len(scores) > 0, "No scores extracted from output"
        assert scores.ndim == 1, f"Scores should be 1D array, got shape: {scores.shape}"
        
        # Data quality check
        nan_count = np.isnan(scores).sum()
        if nan_count == len(scores):
            print("WARNING: All scores are NaN - check molevaluate configuration")
        
        return ComponentResults([scores])