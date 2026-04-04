# Makefile for dice_test
# Usage:
#   make copy_config    app=nn_cuda
#   make copy_meta      app=nn_cuda
#   make build          app=nn_cuda
#   make run            app=nn_cuda test=test_dice
#   make collect        app=nn_cuda logfile=...
#   make test_dice      app=nn_cuda
#   make test_gpu       app=nn_cuda
#   make test_all

# Variables you can override on the command line
app          ?= nn_cuda
test         ?= test_dice     # or test_gpu
RUN_SCRIPT   ?= run
BENCH_DIR    := ../$(app)
CFG_DIR      := cfg
SW_DIR       := sw
SUMMARY_DIR  := result_summary/$(app)

# Config files for each test type
CFG_gpu      := $(CFG_DIR)/gpgpusim_gpu_rtx2060s.config

# Metadata & pptx (only for dice)
META_SRC     := $(SW_DIR)/$(app).1.sm_52.meta
PPTX_SRC     := $(SW_DIR)/$(app).1.sm_52.pptx

# Timestamp for logs
NOW          := $(shell date +%Y%m%d_%H%M%S)
LOG_SUFFIX   := $(strip $(test))_$(NOW)

STD_LOGFILE  := $(LOG_SUFFIX).log
STDERR_LOGFILE := $(LOG_SUFFIX)_stderr.log

POWER_LOG_NAME := accelwattch_power_report.log
POWER_SRC      := $(BENCH_DIR)/$(POWER_LOG_NAME)

POWER_OUTDIR   := $(SUMMARY_DIR)/power_result
POWER_DST      := $(POWER_OUTDIR)/$(LOG_SUFFIX)_$(POWER_LOG_NAME)

POWER_SCRIPT   := extract_power.py

.PHONY: all test_all copy_config build run collect test_gpu power

all: test_all

# Composite: run both tests for all apps in a list file
# Usage: make test_all
# Expects a file "benchmarks.lst" in dice_test/ listing one app per line
test_all:
	@for b in $$(cat benchmarks.lst); do \
	  $(MAKE) test_gpu  app=$$b; \
	done

test_specific:l
	@for b in $$(cat benchmarks_spec.lst); do \
	  $(MAKE) test_gpu  app=$$b; \
	done

# Copy the appropriate config into the benchmark dir
copy_config:
	@echo "==> Copying config for $(test) into $(BENCH_DIR)..."
	@cp $(CFG_gpu) $(BENCH_DIR)/gpgpusim.config
	@cp $(CFG_DIR)/accelwattch_ptx_sim.xml $(BENCH_DIR)/accelwattch_ptx_sim.xml
	@cp $(CFG_DIR)/accelwattch_ptx_sim_alt.xml $(BENCH_DIR)/accelwattch_ptx_sim_alt.xml
	@cp $(CFG_DIR)/config_turing_islip.icnt $(BENCH_DIR)/config_turing_islip.icnt


# Build the benchmark
build:
	@echo "==> Building $(app) in $(BENCH_DIR)..."
	@$(MAKE) clean -C $(BENCH_DIR)
	@$(MAKE) -C $(BENCH_DIR)

# Run the benchmark
run:
	@echo "==> Running $(app) ($(test))..."
	@if [ ! -f "$(BENCH_DIR)/$(RUN_SCRIPT)" ]; then \
		echo "[ERROR] Missing run script: $(BENCH_DIR)/$(RUN_SCRIPT)"; \
		exit 1; \
	fi
	-@cd $(BENCH_DIR) && . ./$(RUN_SCRIPT) >$(STD_LOGFILE) 2>$(STDERR_LOGFILE)
	@echo "    Stdout log: $(BENCH_DIR)/$(STD_LOGFILE)"
	@echo "    Stderr log: $(BENCH_DIR)/$(STDERR_LOGFILE)"
	@mkdir -p $(SUMMARY_DIR)
	@cp $(BENCH_DIR)/$(STD_LOGFILE) $(SUMMARY_DIR)/$(STD_LOGFILE)
	@cp $(BENCH_DIR)/$(STDERR_LOGFILE) $(SUMMARY_DIR)/$(STDERR_LOGFILE)

# Collect data from the copied log
collect:
	@echo "==> Collecting data for $(app)..."
	@python3 collect_data.py $(SUMMARY_DIR)/$(STD_LOGFILE) $(SUMMARY_DIR) --$(test)

power:
	@mkdir -p $(POWER_OUTDIR)
	@if [ -f "$(POWER_SRC)" ]; then \
		echo "==> Found power log: $(POWER_SRC)"; \
		cp "$(POWER_SRC)" "$(POWER_DST)"; \
		echo "==> Processing power log into $(POWER_OUTDIR)"; \
		python3 "$(POWER_SCRIPT)" "$(POWER_DST)" --out "$(POWER_OUTDIR)" --no-metric-column; \
	else \
		echo "==> WARNING: $(POWER_SRC) not found (did AccelWattch run?)"; \
	fi

power_all:
	@for b in $$(cat benchmarks.lst); do \
	  $(MAKE) power app=$$b; \
	done

timeline:
	@echo "==> Generating timeline for $(app)..."
	@log_file=$$(find $(SUMMARY_DIR) -type f -name "test_dice_*.log" ! -name "*_stderr.log" | head -n 1); \
	if [ -z "$$log_file" ]; then \
		echo "No suitable log file found for test_dice."; \
	else \
		echo "Using log file: $$log_file"; \
		python3 create_timeline.py $$log_file; \
	fi


# Clean up configs, logs, builds, and rundir for the given app
clean:
	@echo "==> Cleaning up for $(app)..."
	-@rm -f $(BENCH_DIR)/gpgpusim.config
	-@rm -f $(BENCH_DIR)/*.log
	-@rm -rf $(BENCH_DIR)/build
	-@rm -rf $(RUNDIR)
	-@rm -rf $(SUMMARY_DIR)
	@echo "==> Clean complete."

clean_all:
	@for b in $$(cat benchmarks.lst); do \
	  $(MAKE) clean app=$$b; \
	done

# Full sequence for GPU test
test_gpu:
	@stamp=$$(date +%Y%m%d_%H%M%S); \
	log_suffix=test_gpu_$$stamp; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix copy_config; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix build; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix run; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix collect
