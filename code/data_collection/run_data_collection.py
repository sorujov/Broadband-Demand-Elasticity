"""
================================================================================
MASTER DATA COLLECTION PIPELINE
================================================================================
Purpose: Orchestrate complete data collection and merge workflow
Author: Samir Orujov
Date: December 11, 2025

This script runs all data collection steps in the correct sequence:
  Step 1: Download ITU telecommunications data
  Step 2: Download World Bank economic indicators
  Step 3: Process raw data and preserve metadata
  Step 4: Merge datasets into analysis-ready format

Usage:
    python code/data_collection/run_data_collection.py
    
    Optional flags:
    --skip-download    Skip download steps (use existing raw data)
    --itu-only         Only download ITU data
    --wb-only          Only download World Bank data
    --verbose          Show detailed output from each step
================================================================================
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


class DataCollectionPipeline:
    """Master orchestrator for data collection workflow."""
    
    def __init__(self, verbose=False):
        self.script_dir = Path(__file__).parent
        self.verbose = verbose
        self.steps = {
            1: ("Download ITU Data", "step1_download_itu.py"),
            2: ("Download World Bank Data", "step2_download_worldbank.py"),
            3: ("Process Raw Data", "step3_process_raw_data.py"),
            4: ("Merge Datasets", "step4_merge_datasets.py")
        }
        self.results = {}
        
    def print_header(self):
        """Print pipeline header."""
        print("\n" + "="*80)
        print("DATA COLLECTION PIPELINE - MASTER ORCHESTRATOR")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Working Directory: {self.script_dir}")
        print("="*80 + "\n")
        
    def print_step_header(self, step_num, step_name):
        """Print step header."""
        print("\n" + "-"*80)
        print(f"STEP {step_num}: {step_name.upper()}")
        print("-"*80)
        
    def run_step(self, step_num, script_name):
        """Run a single pipeline step."""
        step_name = self.steps[step_num][0]
        self.print_step_header(step_num, step_name)
        
        script_path = self.script_dir / script_name
        
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            self.results[step_num] = "FAILED - Script not found"
            return False
            
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Always show output (not just in verbose mode)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
                
            print(f"\n[OK] Step {step_num} completed successfully")
            self.results[step_num] = "SUCCESS"
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Step {step_num} failed with exit code {e.returncode}")
            if e.stdout and not self.verbose:
                print("\nOutput:")
                print(e.stdout)
            if e.stderr:
                print("\nError details:")
                print(e.stderr)
            self.results[step_num] = f"FAILED - Exit code {e.returncode}"
            return False
            
        except Exception as e:
            print(f"\n[ERROR] Unexpected error in step {step_num}: {str(e)}")
            self.results[step_num] = f"FAILED - {str(e)}"
            return False
            
    def print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        all_success = True
        has_failures = False
        for step_num, (step_name, _) in self.steps.items():
            status = self.results.get(step_num, "NOT RUN")
            
            # Determine symbol based on status
            if status == "SUCCESS":
                symbol = "[OK]"
            elif status == "NOT RUN":
                symbol = "-"
            else:
                symbol = "[FAILED]"
                has_failures = True
                
            print(f"  {symbol} Step {step_num}: {step_name:<30} [{status}]")
            
            # Only count as failure if it was run and failed
            if status.startswith("FAILED"):
                all_success = False
                
        print("="*80)
        
        if has_failures:
            print("\n[ERROR] Pipeline completed with failures")
            print("Review the output above to identify issues")
        elif all_success and any(s == "NOT RUN" for s in self.results.values()):
            print("\n[SUCCESS] Requested pipeline steps completed successfully!")
            print("\nOutput files updated:")
            if 3 in self.results and self.results[3] == "SUCCESS":
                print("  - data/interim/*_processed.xlsx (processed data)")
            if 4 in self.results and self.results[4] == "SUCCESS":
                print("  - data/processed/data_merged_with_series.xlsx")
        elif all_success:
            print("\n[SUCCESS] All pipeline steps completed successfully!")
            print("\nOutput files created:")
            print("  - data/raw/*.csv (6 ITU files + 1 World Bank file)")
            print("  - data/interim/*_processed.xlsx (7 processed files)")
            print("  - data/processed/data_merged_with_series.xlsx (462 obs × 33 vars)")
            print("\nNext step: Run data preparation scripts")
            print("  python code/data_preparation/01_analysis.py")
        else:
            print("\n[WARNING] Pipeline completed with errors")
            print("Review the output above to identify issues")
            
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        return all_success
        
    def run_full_pipeline(self):
        """Run complete data collection pipeline."""
        self.print_header()
        
        # Run all steps in sequence
        for step_num in sorted(self.steps.keys()):
            script_name = self.steps[step_num][1]
            success = self.run_step(step_num, script_name)
            
            if not success:
                print(f"\n[WARNING] Step {step_num} failed. Continuing with remaining steps...")
                
        return self.print_summary()
        
    def run_partial_pipeline(self, skip_download=False, itu_only=False, wb_only=False):
        """Run partial pipeline based on flags."""
        self.print_header()
        
        if skip_download:
            print("[INFO] Skipping download steps (using existing raw data)\n")
            steps_to_run = [3, 4]
        elif itu_only:
            print("[INFO] ITU-only mode: Steps 1, 3, 4\n")
            steps_to_run = [1, 3, 4]
        elif wb_only:
            print("[INFO] World Bank-only mode: Steps 2, 3, 4\n")
            steps_to_run = [2, 3, 4]
        else:
            steps_to_run = [1, 2, 3, 4]
            
        for step_num in steps_to_run:
            script_name = self.steps[step_num][1]
            success = self.run_step(step_num, script_name)
            
            if not success:
                print(f"\n[WARNING] Step {step_num} failed. Continuing with remaining steps...")
                
        return self.print_summary()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run complete data collection and merge pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_data_collection.py
  
  # Skip downloads, only process and merge existing data
  python run_data_collection.py --skip-download
  
  # Only download and process ITU data
  python run_data_collection.py --itu-only
  
  # Show detailed output from each step
  python run_data_collection.py --verbose
        """
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download steps and use existing raw data'
    )
    
    parser.add_argument(
        '--itu-only',
        action='store_true',
        help='Only download ITU data (skip World Bank)'
    )
    
    parser.add_argument(
        '--wb-only',
        action='store_true',
        help='Only download World Bank data (skip ITU)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output from each step'
    )
    
    args = parser.parse_args()
    
    # Check for conflicting flags
    if sum([args.skip_download, args.itu_only, args.wb_only]) > 1:
        print("[ERROR] Cannot combine --skip-download, --itu-only, and --wb-only")
        sys.exit(1)
        
    # Create and run pipeline
    pipeline = DataCollectionPipeline(verbose=args.verbose)
    
    if args.skip_download or args.itu_only or args.wb_only:
        success = pipeline.run_partial_pipeline(
            skip_download=args.skip_download,
            itu_only=args.itu_only,
            wb_only=args.wb_only
        )
    else:
        success = pipeline.run_full_pipeline()
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
