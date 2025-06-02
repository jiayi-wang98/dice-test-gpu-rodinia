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
BENCH_DIR    := ../$(app)
CFG_DIR      := cfg
SW_DIR       := sw
SUMMARY_DIR  := result_summary/$(app)

# Config files for each test type
CFG_gpu      := $(CFG_DIR)/gpgpusim_gpu.config
CFG_dice     := $(CFG_DIR)/gpgpusim_dice.config

# Metadata & pptx (only for dice)
META_SRC     := $(SW_DIR)/$(app).1.sm_52.meta
PPTX_SRC     := $(SW_DIR)/$(app).1.sm_52.pptx

# Timestamp for logs
NOW          := $(shell date +%Y%m%d_%H%M%S)
LOG_SUFFIX   := $(strip $(test))_$(NOW)

STD_LOGFILE  := $(LOG_SUFFIX).log
ASAN_LOGFILE := $(LOG_SUFFIX)_asan.log

.PHONY: all test_all copy_config copy_meta build run collect test_dice test_gpu

all: test_all

# Composite: run both tests for all apps in a list file
# Usage: make test_all
# Expects a file "benchmarks.lst" in dice_test/ listing one app per line
test_all:
	@for b in $$(cat benchmarks.lst); do \
	  $(MAKE) test_dice app=$$b; \
	  $(MAKE) test_gpu  app=$$b; \
	done

test_specific:
	@for b in $$(cat benchmarks_spec.lst); do \
	  $(MAKE) test_dice app=$$b; \
	  $(MAKE) test_gpu  app=$$b; \
	done

# Copy the appropriate config into the benchmark dir
copy_config:
	@echo "==> Copying config for $(test) into $(BENCH_DIR)..."
ifeq ($(test),test_gpu)
	@cp $(CFG_gpu) $(BENCH_DIR)/gpgpusim.config
else
	@cp $(CFG_dice) $(BENCH_DIR)/gpgpusim.config
endif
	@cp $(CFG_DIR)/gpuwattch_gtx480.xml $(BENCH_DIR)/gpuwattch_gtx480.xml
	@cp $(CFG_DIR)/config_fermi_islip.icnt $(BENCH_DIR)/config_fermi_islip.icnt

# Copy metadata & pptx (only for dice)
copy_meta:
ifeq ($(test),test_dice)
	@echo "==> Copying metadata and PPTX into $(BENCH_DIR)..."
	@cp $(META_SRC) $(BENCH_DIR)/
	@cp $(PPTX_SRC) $(BENCH_DIR)/
else
	@echo "==> Skipping metadata/pptx for GPU test"
endif

# Build the benchmark
build:
	@echo "==> Building $(app) in $(BENCH_DIR)..."
	@$(MAKE) clean -C $(BENCH_DIR)
	@$(MAKE) -C $(BENCH_DIR)

# Run the benchmark
run:
	@echo "==> Running $(app) ($(test))..."
	-@cd $(BENCH_DIR) && . ./run >$(STD_LOGFILE) 2>$(ASAN_LOGFILE)
	@echo "    Stdout log: $(BENCH_DIR)/$(STD_LOGFILE)"
	@echo "    ASan   log: $(BENCH_DIR)/$(ASAN_LOGFILE)"
	@mkdir -p $(SUMMARY_DIR)
	@cp $(BENCH_DIR)/$(STD_LOGFILE) $(SUMMARY_DIR)/$(STD_LOGFILE)
	@cp $(BENCH_DIR)/$(ASAN_LOGFILE) $(SUMMARY_DIR)/$(ASAN_LOGFILE)

# Collect data from the copied log
collect:
	@echo "==> Collecting data for $(app)..."
	@python3 collect_data.py $(SUMMARY_DIR)/$(STD_LOGFILE) $(SUMMARY_DIR) --$(test)

time_line:
	@echo "==> Generating timeline for $(app)..."
	@log_file=$$(find $(SUMMARY_DIR) -type f -name "test_dice_*.log" ! -name "*_asan.log" | head -n 1); \
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
	-@rm -f $(BENCH_DIR)/*.1.sm_52.meta
	-@rm -f $(BENCH_DIR)/*.1.sm_52.pptx
	-@rm -rf $(BENCH_DIR)/build
	-@rm -rf $(RUNDIR)
	-@rm -rf $(SUMMARY_DIR)
	@echo "==> Clean complete."

clean_all:
	@for b in $$(cat benchmarks.lst); do \
	  $(MAKE) clean app=$$b; \
	done

# Full sequence for dice test
test_dice:
	$(MAKE) test=test_dice copy_config copy_meta build run collect

# Full sequence for GPU test
test_gpu:
	$(MAKE) test=test_gpu copy_config build run collect

