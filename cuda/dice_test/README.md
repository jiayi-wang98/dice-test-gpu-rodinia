# DICE Benchmark Suite

This repository contains the DICE benchmark suite for evaluating performance on DICE and GPU platforms. The suite includes configuration files, metadata, and scripts to run benchmarks, collect results, and generate visualizations.

## Directory Structure

- **`./cfg`**: Configuration files for DICE and GPU platforms.
  - `gpgpusim_dice.config`: Configuration for DICE runs.
  - `gpgpusim_gpu.config`: Configuration for GPU runs.
  - `gpuwattch_gtx480.xml` and `config_fermi_islip.icnt`: Additional configuration files used for both platforms.
- **`./sw`**: Metadata and PPTX files for DICE benchmarks (e.g., `<app>.1.sm_52.meta`, `<app>.1.sm_52.pptx`).
- **`./result_summary`**: Directory storing result summaries for each benchmark, organized by application (e.g., `result_summary/<app>`).
- **`./benchmarks.lst`**: List of benchmark applications to run with `make` or `make test_all`.
- **`./benchmarks_spec.lst`**: List of benchmark applications to run with `make test_specific`.

## Prerequisites

- Ensure you have `make`, `python3`, and a compatible EDA tool environment set up.
- The benchmark application directories (e.g., `../<app>`) must exist and contain the necessary source code.
- Python scripts (`collect_data.py` and `create_timeline.py`) are Saved: System: are required for result collection and timeline visualization.

## Usage

The Makefile provides targets to configure, build, run, and analyze benchmarks on DICE or GPU platforms. Below are the available commands and their usage.

### Key Variables

You can override these variables on the command line:
- **`app`**: The benchmark application to run (default: `nn_cuda`).
- **`test`**: The platform to test (`test_dice` or `test_gpu`, default: `test_dice`).
- **`logfile`**: Custom log file name for result collection (optional).

### Available Commands

- **`make test_all`**:
  - Runs both DICE and GPU tests for all benchmarks listed in `benchmarks.lst`.
  - Example: `make test_all`

- **`make test_specific`**:
  - Runs both DICE and GPU tests for benchmarks listed in `benchmarks_spec.lst`.
  - Example: `make test_specific`

- **`make test_dice app=<app>`**:
  - Runs the full DICE test sequence (copy config, copy metadata, build, run, collect results) for the specified `<app>`.
  - Example: `make test_dice app=streamcluster`

- **`make test_gpu app=<app>`**:
  - Runs the full GPU test sequence (copy config, build, run, collect results) for the specified `<app>`.
  - Example: `make test_gpu app=streamcluster`

- **`make timeline app=<app>`**:
  - Generates a pipeline diagram (timeline visualization) for the specified `<app>` using the latest DICE test log in `result_summary/<app>`.
  - Example: `make timeline app=streamcluster`

- **`make copy_config app=<app> test=<test>`**:
  - Copies the appropriate configuration files (`gpgpusim_dice.config` or `gpgpusim_gpu.config`, plus `gpuwattch_gtx480.xml` and `config_fermi_islip.icnt`) to the benchmark directory (`../<app>`).
  - Example: `make copy_config app=nn_cuda test=test_dice`

- **`make copy_meta app=<app> test=test_dice`**:
  - Copies DICE-specific metadata and PPTX files (e.g., `<app>.1.sm_52.meta`, `<app>.1.sm_52.pptx`) to the benchmark directory. Skipped for GPU tests.
  - Example: `make copy_meta app=nn_cuda test=test_dice`

- **`make build app=<app>`**:
  - Builds the specified `<app>` in its directory (`../<app>`).
  - Example: `make build app=nn_cuda`

- **`make run app=<app> test=<test>`**:
  - Runs the specified `<app>` on the chosen platform (`test_dice` or `test_gpu`). Outputs logs to `../<app>/<test>_<timestamp>.log` and `../<app>/<test>_<timestamp>_asan.log`, and copies them to `result_summary/<app>`.
  - Example: `make run app=nn_cuda test=test_dice`

- **`make collect app=<app> test=<test> [logfile=<logfile>]`**:
  - Collects results from the specified log file (or the default `<test>_<timestamp>.log`) and stores them in `result_summary/<app>`.
  - Example: `make collect app=nn_cuda test=test_dice`

- **`make clean app=<app>`**:
  - Removes configuration files, logs, metadata, PPTX files, build artifacts, and the `result_summary/<app>` directory for the specified `<app>`.
  - Example: `make clean app=nn_cuda`

- **`make clean_all`**:
  - Cleans up all benchmarks listed in `benchmarks.lst`.
  - Example: `make clean_all`

### Examples

- **Run the `streamcluster` benchmark on DICE**:
  ```bash
  make test_dice app=streamcluster
  ```

- **Run the `streamcluster` benchmark on GPU**:
  ```bash
  make test_gpu app=streamcluster
  ```

- **Visualize the pipeline diagram for `streamcluster` on DICE**:
  ```bash
  make timeline app=streamcluster
  ```

### Notes

- **Log Files**: Each run generates two logs in `../<app>` and `result_summary/<app>`:
  - Standard output: `<test>_<timestamp>.log`
  - AddressSanitizer output: `<test>_<timestamp>_asan.log`
- **Top Cell Name Warning**: When working with GDSII files (e.g., for DRC results), ensure the top cell name in output files does not match the original layout to avoid conflicts during processing (e.g., streaming in for metal fill).
- **Custom Log Files**: Use the `logfile` variable with `make collect` to specify a custom log file if needed.
- **Benchmark Directory**: The benchmark application must exist in `../<app>` relative to the `dice_test` directory.
- **Timeline Visualization**: The `timeline` target requires a valid DICE test log in `result_summary/<app>`. If none is found, it will report an error.

## Troubleshooting

- **Missing Benchmark Directory**: Ensure `../<app>` exists and contains the necessary source code.
- **Log File Not Found**: Verify that `result_summary/<app>` contains the expected log files for `timeline` or `collect` targets.
- **Configuration Issues**: Check that `cfg/` and `sw/` contain the required configuration and metadata files for your benchmark.
- **Density Violations**: If integrating with DRC flows (e.g., metal fill), ensure the GDSII output files (like `DOD.gds`) have distinct top cell names to avoid conflicts during stream-in.

For further assistance, consult the Makefile or contact the CAD team for specific configuration details.