"""
================================================================================
MAIN PIPELINE: Broadband Demand Elasticity Analysis
================================================================================
Purpose: Orchestrate complete data pipeline from collection to analysis
Author: Samir Orujov
Date: November 27, 2025

Pipeline Stages:
1. Data Collection: Download ITU and World Bank data
2. Data Preparation: Clean, transform, and prepare analysis-ready dataset
3. Analysis: Run model selection and main specifications

Usage:
    python code/main.py                    # Run full pipeline
    python code/main.py --skip-collection  # Skip data download
    python code/main.py --skip-preparation # Skip data preparation
    python code/main.py --analysis-only    # Run analysis only

================================================================================
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


class PipelineRunner:
    """Orchestrates the complete data analysis pipeline"""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.code_dir = self.project_root / 'code'
        self.data_dir = self.project_root / 'data'
        self.results_dir = self.project_root / 'results'
        
    def print_section(self, title):
        """Print formatted section header"""
        print("\n" + "="*80)
        print(f"{title}")
        print("="*80 + "\n")
        
    def run_script(self, script_path, description):
        """Run a Python script and handle errors"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {description}")
        print(f"   Script: {script_path.relative_to(self.project_root)}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"   ✓ Completed successfully\n")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Error occurred!")
            print(f"\n{e.stderr}")
            return False
            
    def stage_1_data_collection(self):
        """Stage 1: Download raw data from ITU and World Bank"""
        self.print_section("STAGE 1: DATA COLLECTION")

        scripts = [
            (self.code_dir / 'data_collection' / 'step1_download_itu.py',
             "Download ITU telecommunications data"),
            (self.code_dir / 'data_collection' / 'step2_download_worldbank.py',
             "Download World Bank economic indicators"),
            (self.code_dir / 'data_collection' / 'step3_process_raw_data.py',
             "Process raw data files"),
            (self.code_dir / 'data_collection' / 'step4_merge_datasets.py',
             "Merge ITU and World Bank datasets"),
        ]

        for script_path, description in scripts:
            if not script_path.exists():
                print(f"   Warning: Script not found: {script_path.name}")
                continue

            success = self.run_script(script_path, description)
            if not success:
                print(f"\nPipeline stopped at: {description}")
                return False

        print("Stage 1 completed: Raw data collected and merged")
        return True
        
    def stage_2_data_preparation(self):
        """Stage 2: Clean and prepare analysis-ready dataset"""
        self.print_section("STAGE 2: DATA PREPARATION")

        # Note: 01_analysis.py is an optional diagnostic tool, not part of main pipeline
        scripts = [
            (self.code_dir / 'data_preparation' / '02_prepare_data.py',
             "Clean, transform, and prepare final dataset"),
        ]

        for script_path, description in scripts:
            if not script_path.exists():
                print(f"   Warning: Script not found: {script_path.name}")
                continue

            success = self.run_script(script_path, description)
            if not success:
                print(f"\nPipeline stopped at: {description}")
                return False

        # Verify analysis-ready data exists
        analysis_ready = self.data_dir / 'processed' / 'analysis_ready_data.csv'
        if analysis_ready.exists():
            print(f"\nStage 2 completed: Analysis-ready data created")
            print(f"   Output: {analysis_ready.relative_to(self.project_root)}")
            return True
        else:
            print(f"\nError: Expected output file not found: {analysis_ready}")
            return False
            
    def stage_3_analysis(self):
        """Stage 3: Run econometric analysis"""
        self.print_section("STAGE 3: ECONOMETRIC ANALYSIS")

        scripts = [
            (self.code_dir / 'analysis' / 'pre_covid' / 'two_way_fe.py',
             "Pre-COVID analysis (2010-2019): 8 control specs × 3 price measures"),
            (self.code_dir / 'analysis' / 'full_sample' / 'two_way_fe_full_sample.py',
             "Full sample analysis (2010-2024): COVID interactions"),
            (self.code_dir / 'analysis' / 'full_sample' / 'covid_diagnostics.py',
             "COVID diagnostics: year-by-year elasticities and placebo tests"),
            (self.code_dir / 'analysis' / 'analysis_visualizations.py',
             "Generate publication-quality figures (6 figures, 300 DPI)"),
        ]

        for script_path, description in scripts:
            if not script_path.exists():
                print(f"   Warning: Script not found: {script_path.name}")
                continue

            success = self.run_script(script_path, description)
            if not success:
                print(f"\nPipeline stopped at: {description}")
                return False

        print("\nStage 3 completed: All analyses finished")
        print(f"   Results:")
        print(f"      - Excel files: results/regression_output/")
        print(f"      - Figures: results/figures/")
        return True
            
    def run_full_pipeline(self, skip_collection=False, skip_preparation=False, analysis_only=False):
        """Run the complete pipeline with optional stage skipping"""
        start_time = datetime.now()
        
        print("\n" + "="*80)
        print("BROADBAND DEMAND ELASTICITY ANALYSIS - FULL PIPELINE")
        print("="*80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project root: {self.project_root}")
        
        # Determine which stages to run
        stages = []
        if not (skip_collection or skip_preparation or analysis_only):
            stages = [
                (self.stage_1_data_collection, "Data Collection"),
                (self.stage_2_data_preparation, "Data Preparation"),
                (self.stage_3_analysis, "Analysis")
            ]
        elif analysis_only:
            stages = [(self.stage_3_analysis, "Analysis")]
        else:
            if not skip_collection:
                stages.append((self.stage_1_data_collection, "Data Collection"))
            if not skip_preparation:
                stages.append((self.stage_2_data_preparation, "Data Preparation"))
            stages.append((self.stage_3_analysis, "Analysis"))
        
        # Run selected stages
        for stage_func, stage_name in stages:
            success = stage_func()
            if not success:
                print("\n" + "="*80)
                print(f"✗ PIPELINE FAILED at stage: {stage_name}")
                print("="*80)
                return False
        
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")

        print(f"\nOUTPUT LOCATIONS:")
        print(f"   Processed data: {self.data_dir / 'processed' / 'analysis_ready_data.csv'}")
        print(f"   Regression output: {self.results_dir / 'regression_output'}")
        print(f"      - Pre-COVID: regression_output/pre_covid_analysis/")
        print(f"      - Full sample: regression_output/full_sample_covid_analysis/")
        print(f"   Figures: {self.results_dir / 'figures'}")
        print(f"      - Analysis figures: figures/analysis_figures/")
        print(f"      - Diagnostics: figures/covid_diagnostics/")

        return True


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run the complete broadband demand elasticity analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code/main.py                     # Run full pipeline
  python code/main.py --skip-collection   # Skip data download
  python code/main.py --analysis-only     # Run analysis only
        """
    )
    
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection stage (use existing raw data)'
    )
    
    parser.add_argument(
        '--skip-preparation',
        action='store_true',
        help='Skip data preparation stage (use existing processed data)'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Run analysis stage only (skip collection and preparation)'
    )
    
    args = parser.parse_args()
    
    # Get project root (2 levels up from this script)
    project_root = Path(__file__).resolve().parents[1]
    
    # Create and run pipeline
    pipeline = PipelineRunner(project_root)
    success = pipeline.run_full_pipeline(
        skip_collection=args.skip_collection,
        skip_preparation=args.skip_preparation,
        analysis_only=args.analysis_only
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
